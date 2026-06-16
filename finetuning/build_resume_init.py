"""Build an init dir to RESUME training from a saved checkpoint.

Saved checkpoints have speaker_encoder.* stripped (not needed for inference). But training
needs it. speaker_encoder is never trained (detached in the loop), so it's identical to base --
just copy the checkpoint and merge base's speaker_encoder weights back in.

Usage:
  python build_resume_init.py --ckpt output/checkpoint-epoch-5 --base model --out resume_init
  python sft_12hz.py --init_model_path resume_init --output_model_path output2 \
                     --train_jsonl train_with_codes.jsonl --batch_size 6 --lr 2e-5 \
                     --num_epochs 300 --speaker_name <your_speaker>
"""
import argparse
import os
import shutil

from safetensors.torch import load_file, save_file

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", default="output/checkpoint-epoch-5", help="checkpoint to resume from")
parser.add_argument("--base", default="model", help="base model dir (source of speaker_encoder weights)")
parser.add_argument("--out", default="resume_init", help="output init dir to create")
args = parser.parse_args()
CKPT = args.ckpt
BASE = args.base
OUT = args.out

# 1) copy the checkpoint dir (config, tokenizer, weights) as the base of the init
if os.path.exists(OUT):
    shutil.rmtree(OUT)
shutil.copytree(CKPT, OUT)

# 2) merge base speaker_encoder.* weights back into the safetensors
ckpt_sd = load_file(os.path.join(CKPT, "model.safetensors"))
base_sd = load_file(os.path.join(BASE, "model.safetensors"))
added = 0
for k, v in base_sd.items():
    if k.startswith("speaker_encoder"):
        ckpt_sd[k] = v
        added += 1
save_file(ckpt_sd, os.path.join(OUT, "model.safetensors"))
print(f"merged {added} speaker_encoder tensors; wrote {OUT}\\model.safetensors")
print("now run sft_12hz.py with --init_model_path resume_init --output_model_path output2")
