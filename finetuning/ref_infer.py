"""Reference inference using the BASE repo's Qwen3TTSModel (no faster-repo CUDA graphs).

Isolates: is garbled/never-stopping output caused by the fine-tune, or by the
faster-qwen3-tts optimized generation path?

  base-repo output GOOD  -> faster-repo generate path is the bug.
  base-repo output BAD   -> the fine-tune / data is the bug.
"""
import argparse

import soundfile as sf
import torch

from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="output/checkpoint-epoch-0", help="path to a finetuned checkpoint")
parser.add_argument("--text", default="Hello, this is a test of the finetuned voice.", help="text to synthesize")
parser.add_argument("--speaker", default="speaker_test", help="speaker name registered during training (--speaker_name)")
parser.add_argument("--language", default="Auto", help="must match the training prefix; keep 'Auto'")
parser.add_argument("--max_new_tokens", type=int, default=512)
args = parser.parse_args()

m = Qwen3TTSModel.from_pretrained(args.model, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="flash_attention_2")

wavs, sr = m.generate_custom_voice(
    text=args.text,
    speaker=args.speaker,
    language=args.language,
    non_streaming_mode=True,
    max_new_tokens=args.max_new_tokens,   # cap so a non-stopping model doesn't run for minutes
    do_sample=False,      # GREEDY: isolates sampling/exposure-bias from a deeper AR bug
    subtalker_dosample=False,
)
sf.write("ref_infer.wav", wavs[0], sr)
print(f"wrote ref_infer.wav  ({len(wavs[0])/sr:.2f}s @ {sr}Hz)")
