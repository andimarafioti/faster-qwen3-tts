# TTS model roadmap for this Spark

Saved on 2026-08-19 so the evaluation plan survives conversation compaction.

## Current services

- Port 7860: Qwen3-TTS 0.6B Base using Torch, for reference-audio voice cloning.
- Port 7861: Qwen3-TTS 1.7B VoiceDesign using Torch, for instruction-designed voices and read-aloud work.
- The prebuilt GGML/qwentts.cpp wheel is not usable on this GB10: its CUDA kernels abort on this GPU architecture. Use Torch here unless a compatible wheel is built.

## Alternative model shortlist

1. Chatterbox Turbo 350M — first choice; English zero-shot voice cloning, lower latency, and paralinguistic tags such as `[laugh]` and `[cough]`.
2. Chatterbox Multilingual V3 500M — second choice for 23+ languages, cross-language cloning, and speaker-similarity comparisons.
3. Fish Speech S2 Pro — expressive multilingual and multi-speaker comparison; more complex deployment and a research-oriented model license.
4. CosyVoice — multilingual cloning and streaming comparison.
5. XTTS v2 — mature multilingual voice-cloning baseline.
6. Kokoro 82M — small, fast conventional-voice baseline rather than a direct cloning competitor.
7. F5-TTS — natural zero-shot cloning research comparison.

Chatterbox Nano 110M is also available when CPU inference or the smallest resource footprint matters most.

## Decision

Start with Chatterbox Turbo in its own environment and service. Keep it isolated from the Qwen environments and services. Compare voice similarity, expressiveness, time to first audio, real-time factor, long-form stability, and VRAM/unified-memory use with the same text and reference clips.

Official project: https://github.com/resemble-ai/chatterbox
