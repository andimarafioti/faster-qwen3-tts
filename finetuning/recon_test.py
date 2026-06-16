"""Teacher-forced check: does the trained model actually predict codec0 on a TRAINING clip?

High accuracy  -> model learned; garbled free-run is AR/sampling/EOS, not learning.
Near-random    -> training isn't learning the mapping despite falling loss.
Also decodes (predicted codec0 + GT residuals) so we can HEAR codec0 quality.
"""
import json

import argparse

import soundfile as sf
import torch
from transformers import AutoConfig

from dataset import TTSDataset
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="output/checkpoint-epoch-0", help="path to a finetuned checkpoint")
parser.add_argument("--jsonl", default="train_with_codes.jsonl", help="prepared training jsonl (uses first line)")
args = parser.parse_args()
MODEL = args.model
JSONL = args.jsonl

q = Qwen3TTSModel.from_pretrained(MODEL, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="flash_attention_2")
model = q.model
config = AutoConfig.from_pretrained(MODEL)

line = json.loads(open(JSONL, encoding="utf-8").readline())
ds = TTSDataset([line], q.processor, config)
batch = ds.collate_fn([ds[0]])

dev = model.device
g = lambda k: batch[k].to(dev)
input_ids = g("input_ids")
codec_ids = g("codec_ids")
ref_mels = batch["ref_mels"].to(dev).to(model.dtype)
text_embedding_mask = g("text_embedding_mask")
codec_embedding_mask = g("codec_embedding_mask")
attention_mask = g("attention_mask")
codec_0_labels = g("codec_0_labels")
codec_mask = g("codec_mask")

with torch.no_grad():
    # real custom_voice inference uses codec_embedding[spk_id=3000], not the speaker_encoder
    speaker_embedding = model.talker.model.codec_embedding.weight[3000].unsqueeze(0).to(model.dtype)
    ite = model.talker.text_projection(model.talker.model.text_embedding(input_ids[:, :, 0])) * text_embedding_mask
    ice = model.talker.model.codec_embedding(input_ids[:, :, 1]) * codec_embedding_mask
    ice[:, 6, :] = speaker_embedding
    emb = ite + ice
    for i in range(1, 16):
        emb = emb + model.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i]) * codec_mask.unsqueeze(-1)

    out = model.talker(inputs_embeds=emb[:, :-1, :], attention_mask=attention_mask[:, :-1], output_hidden_states=True)
    logits = out.logits  # [1, T-1, vocab]
    pred0 = logits.argmax(-1)[0]          # aligned to codec_0_labels[:,1:]
    gt0 = codec_0_labels[0, 1:]
    valid = gt0 != -100
    acc = (pred0[valid] == gt0[valid]).float().mean().item()
    print(f"codec0 teacher-forced argmax accuracy: {acc:.3f}  over {int(valid.sum())} frames")

    frame_pos = torch.where(codec_mask[0])[0]                 # positions of audio frames (full seq)
    gt_codes = codec_ids[0, frame_pos].clone()                # [C,16] ground truth

    def decode_and_write(codes, name):
        audio, sr = q.model.speech_tokenizer.decode({"audio_codes": codes.unsqueeze(0)})
        a = audio[0]
        a = a.detach().float().cpu().numpy() if hasattr(a, "detach") else a
        sf.write(name, a, sr)
        print("wrote", name)

    # (1) predicted codec0 + GT residuals
    pred_codes = gt_codes.clone()
    pred_codes[:, 0] = pred0[frame_pos - 1]
    decode_and_write(pred_codes, "recon_predcodec0.wav")

    # (2) GT codec0 + SUB-TALKER-predicted residuals -- isolates the residual head
    talker_hidden = out.hidden_states[0][-1]                  # [1, T-1, D]
    frame_hidden = talker_hidden[0][codec_mask[0, :-1]]       # [C, D]
    sub_logits, _ = model.talker.forward_sub_talker_finetune(gt_codes, frame_hidden)  # [C, 15, V]
    pred_res = sub_logits.argmax(-1)                          # [C, 15]
    res_acc = (pred_res == gt_codes[:, 1:]).float().mean().item()
    print(f"sub-talker residual teacher-forced acc: {res_acc:.3f}")
    codes_predres = gt_codes.clone()
    codes_predres[:, 1:] = pred_res
    decode_and_write(codes_predres, "recon_predresiduals.wav")
