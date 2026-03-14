---
title: faster-qwen3-tts
author: andito
emoji: 🎙
tags: [text-to-speech, streaming, cuda-graphs]
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
preload_from_hub:
  - nvidia/parakeet-tdt-0.6b-v3
  - Qwen/Qwen3-TTS-12Hz-0.6B-Base
  - Qwen/Qwen3-TTS-12Hz-1.7B-Base
  - Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice
  - Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
  - Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign
---

# Faster Qwen3-TTS Demo

This Space hosts the operator-style demo UI for **faster-qwen3-tts** with streaming audio, TTFA/RTF metrics, voice clone, custom voices, and voice design.

## Highlights

- Control-room style console for model selection, queue visibility, and live run telemetry
- Voice clone workflow with upload, preset voices, in-browser recording, and on-demand transcription
- CustomVoice and VoiceDesign flows in the same interface, with compatible-model guidance
- Local recent-job history so you can restore prior settings quickly while iterating

## Run locally (no Docker)

```bash
pip install "faster-qwen3-tts[demo]"
python server.py --model Qwen/Qwen3-TTS-12Hz-0.6B-Base
# open http://localhost:7860
```

## Run with Docker

```bash
docker build -t faster-qwen3-tts-demo .
docker run --gpus all -p 7860:7860 faster-qwen3-tts-demo
```
