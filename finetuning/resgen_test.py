"""Measure code_predictor.generate() residual accuracy in isolation (GT talker hidden + GT codec0).

Compares the AR within-frame residual generator against the teacher-forced 0.92 baseline.
  ~0.9  -> residual generation is fine; talker is hypersensitive to residual feedback (overfit/brittle).
  low   -> code_predictor.generate path is buggy (differs from forward_finetune).
"""
import json

import argparse

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
talker = model.talker
config = AutoConfig.from_pretrained(MODEL)

line = json.loads(open(JSONL, encoding="utf-8").readline())
ds = TTSDataset([line], q.processor, config)
batch = ds.collate_fn([ds[0]])
dev = model.device
input_ids = batch["input_ids"].to(dev); codec_ids = batch["codec_ids"].to(dev)
text_embedding_mask = batch["text_embedding_mask"].to(dev)
codec_embedding_mask = batch["codec_embedding_mask"].to(dev)
codec_mask = batch["codec_mask"].to(dev)

with torch.no_grad():
    spk = talker.model.codec_embedding.weight[3000].unsqueeze(0).to(model.dtype)
    ite = talker.text_projection(talker.model.text_embedding(input_ids[:, :, 0])) * text_embedding_mask
    ice = talker.model.codec_embedding(input_ids[:, :, 1]) * codec_embedding_mask
    ice[:, 6, :] = spk
    emb = ite + ice
    for i in range(1, 16):
        emb = emb + talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i]) * codec_mask.unsqueeze(-1)
    out = talker(inputs_embeds=emb[:, :-1, :], attention_mask=batch["attention_mask"][:, :-1].to(dev), output_hidden_states=True)
    hidden = out.hidden_states[0][-1]
    frame_hidden = hidden[0][codec_mask[0, :-1]]          # [C, D]
    gt = codec_ids[0, codec_mask[0]]                       # [C,16]

    C = frame_hidden.shape[0]
    pos_correct = torch.zeros(15)
    for k in range(C):
        h = frame_hidden[k].view(1, 1, -1)
        c0_emb = talker.get_input_embeddings()(gt[k, 0].view(1, 1))
        pred = talker.code_predictor.generate(
            inputs_embeds=torch.cat([h, c0_emb], dim=1),
            max_new_tokens=15, do_sample=False, return_dict_in_generate=True,
        )
        res = pred.sequences[0].cpu()                      # [15]
        pos_correct += (res == gt[k, 1:].cpu()).float()
    pos_acc = (pos_correct / C)
    print(f"overall AR residual acc: {pos_acc.mean():.3f}  (teacher-forced baseline 0.924)")
    print("per-residual-position acc (group1..group15):")
    print("  " + " ".join(f"{a:.2f}" for a in pos_acc.tolist()))
