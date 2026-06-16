"""Manual greedy AR with NO KV cache, reusing the training-style prefill that recon proves correct.

Reproduces clean audio  -> generate()'s own prefill/KV-cache/position path is the bug.
Still noise             -> deeper issue in the shared forward.
"""
import argparse
import json

import soundfile as sf
import torch
from transformers import AutoConfig

from dataset import TTSDataset
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="output/checkpoint-epoch-0", help="path to a finetuned checkpoint")
parser.add_argument("--jsonl", default="train_with_codes.jsonl", help="prepared training jsonl (uses first line)")
parser.add_argument("--max_frames", type=int, default=200)
parser.add_argument("--use_gt_residuals", action="store_true",
                    help="feed back GT residuals (isolate codec0 AR robustness from residual feedback)")
args = parser.parse_args()
MODEL = args.model
JSONL = args.jsonl
MAX_FRAMES = args.max_frames
USE_GT_RESIDUALS = args.use_gt_residuals

q = Qwen3TTSModel.from_pretrained(MODEL, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="flash_attention_2")
model = q.model
talker = model.talker
config = AutoConfig.from_pretrained(MODEL)
EOS = config.talker_config.codec_eos_token_id

line = json.loads(open(JSONL, encoding="utf-8").readline())
ds = TTSDataset([line], q.processor, config)
batch = ds.collate_fn([ds[0]])
dev = model.device

input_ids = batch["input_ids"].to(dev)
codec_ids = batch["codec_ids"].to(dev)
text_embedding_mask = batch["text_embedding_mask"].to(dev)
codec_embedding_mask = batch["codec_embedding_mask"].to(dev)
codec_mask = batch["codec_mask"].to(dev)

with torch.no_grad():
    spk = talker.model.codec_embedding.weight[3000].unsqueeze(0).to(model.dtype)
    ite = talker.text_projection(talker.model.text_embedding(input_ids[:, :, 0])) * text_embedding_mask
    ice = talker.model.codec_embedding(input_ids[:, :, 1]) * codec_embedding_mask
    ice[:, 6, :] = spk
    emb_full = ite + ice
    for i in range(1, 16):
        emb_full = emb_full + talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i]) * codec_mask.unsqueeze(-1)

    first_frame = int(torch.where(codec_mask[0])[0][0])     # position where audio frames start
    prefill = emb_full[:, :first_frame, :].clone()          # known-good prefill (through codec_bos)

    # text contribution that training adds at every frame position (text channel = tts_pad there)
    pad_id = torch.tensor([[config.tts_pad_token_id]], device=dev)
    pad_text_emb = talker.text_projection(talker.model.text_embedding(pad_id))  # [1,1,D]

    seq = prefill
    out_codes = []
    for step in range(MAX_FRAMES):
        o = talker(inputs_embeds=seq, attention_mask=torch.ones(seq.shape[:2], device=dev, dtype=torch.long), output_hidden_states=True)
        c0 = o.logits[0, -1].argmax()
        if int(c0) == EOS:
            print(f"EOS at frame {step}")
            break
        hidden = o.hidden_states[0][-1][:, -1:, :]                 # talker hidden at last pos
        c0_emb = talker.get_input_embeddings()(c0.view(1, 1))
        pred = talker.code_predictor.generate(
            inputs_embeds=torch.cat([hidden, c0_emb], dim=1),
            max_new_tokens=15, do_sample=False, return_dict_in_generate=True,
        )
        res = pred.sequences                                       # [1,15]
        gt_frames = codec_ids[0, codec_mask[0]]                     # [GT_T,16]
        if USE_GT_RESIDUALS and step < gt_frames.shape[0]:
            res = gt_frames[step, 1:].view(1, 15)                  # ground-truth residuals for this frame
        out_codes.append(torch.cat([c0.view(1, 1), res], dim=-1))  # [1,16]
        fe = c0_emb + pad_text_emb
        for i in range(1, 16):
            fe = fe + talker.code_predictor.get_input_embeddings()[i - 1](res[:, i - 1:i])
        seq = torch.cat([seq, fe], dim=1)

    codes = torch.cat(out_codes, dim=0)                            # [T,16]
    print("generated frames:", codes.shape[0], "(GT was", int(codec_mask[0].sum()), ")")
    # compare to GT codec0
    gt0 = codec_ids[0, codec_mask[0]][:, 0]
    n = min(len(gt0), codes.shape[0])
    print("codec0 match vs GT (first", n, "frames):", (codes[:n, 0] == gt0[:n]).float().mean().item())
    audio, sr = q.model.speech_tokenizer.decode({"audio_codes": codes.unsqueeze(0)})
    a = audio[0]
    a = a.detach().float().cpu().numpy() if hasattr(a, "detach") else a
    sf.write("manual_ar.wav", a, sr)
    print("wrote manual_ar.wav")
