#!/usr/bin/env python3
"""Play streaming TTS chunks locally without gaps between chunks.

Usage:
    python examples/streaming_playback.py \
      --model Qwen/Qwen3-TTS-12Hz-0.6B-Base \
      --ref-audio ref_audio.wav \
      --ref-text "Hello from the reference clip." \
      --text "What do you mean that I'm not real?" \
      --language English

Requires:
    pip install sounddevice
"""

from __future__ import annotations

import argparse
import sys

import torch

sys.path.insert(0, ".")

from examples.audio import StreamPlayer
from faster_qwen3_tts import FasterQwen3TTS


def main():
    parser = argparse.ArgumentParser(description="Stream TTS audio to your speakers")
    parser.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-0.6B-Base", help="Model id or local path")
    parser.add_argument("--ref-audio", required=True, help="Reference audio path")
    parser.add_argument("--ref-text", required=True, help="Reference transcript")
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--language", default="English", help="Target language")
    parser.add_argument("--device", default="cuda", help="Device")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"], help="Model dtype")
    parser.add_argument("--chunk-size", type=int, default=8, help="Codec steps per chunk")
    args = parser.parse_args()

    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }

    model = FasterQwen3TTS.from_pretrained(
        args.model,
        device=args.device,
        dtype=dtype_map[args.dtype],
    )

    play = StreamPlayer()
    try:
        for audio_chunk, sr, timing in model.generate_voice_clone_streaming(
            text=args.text,
            language=args.language,
            ref_audio=args.ref_audio,
            ref_text=args.ref_text,
            chunk_size=args.chunk_size,
        ):
            print(
                f"chunk={timing['chunk_index']} "
                f"steps={timing['chunk_steps']} "
                f"prefill_ms={timing['prefill_ms']:.0f} "
                f"decode_ms={timing['decode_ms']:.0f}"
            )
            play(audio_chunk, sr)
    finally:
        play.close()


if __name__ == "__main__":
    main()
