# Parity Sample Set

These samples compare **Qwen3TTS** (dynamic cache) against **FasterQwen3TTS** (static cache). The algorithms are equivalent, but the attention kernel choice differs, so outputs may not be bit-identical. Use these to compare subjective quality and sentence completion.

## CustomVoice Samples

- Model: `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` (1.7B)
- Language: `English`
- Speakers: `aiden`, `serena`
- Generation: `max_new_tokens=168` (~14s), `min_new_tokens=2`, `temperature=0.9`, `top_k=50`, `top_p=1.0`, `repetition_penalty=1.05`
- RNG seed: `1337`

## CustomVoice Prompts

1. "We met at the corner cafe after work and talked about weekend plans. The street was quiet, the lights were warm, and the time passed quickly. We stayed a bit longer."
2. "On Tuesday morning I missed the bus, so I walked home through the park. I took the long path and listened to the wind in the trees before heading back. I took my time."

## ICL Prompts

1. "We met at the corner cafe after work and talked about weekend plans. The street was quiet, the lights were warm, and the time passed quickly. We stayed a bit longer."
2. "On Tuesday morning I missed the bus, so I walked home through the park. I took the long path and listened to the wind in the trees before heading back. I took my time."

## Files

- Voice `aiden`, prompt 1:
  - `custom_aiden_gen1_static.wav`
  - `custom_aiden_gen1_dynamic.wav`
- Voice `aiden`, prompt 2:
  - `custom_aiden_gen2_static.wav`
  - `custom_aiden_gen2_dynamic.wav`
- Voice `serena`, prompt 1:
  - `custom_serena_gen1_static.wav`
  - `custom_serena_gen1_dynamic.wav`
- Voice `serena`, prompt 2:
  - `custom_serena_gen2_static.wav`
  - `custom_serena_gen2_dynamic.wav`

## ICL (Voice Clone) Samples

- Model: `Qwen/Qwen3-TTS-12Hz-1.7B-Base` (1.7B)
- Language: `English`
- Reference audios: `ref_audio.wav`, `ref_audio_2.wav`, `ref_audio_3.wav`
- Reference text: auto‑transcribed with `nano-parakeet` unless `PARITY_REF_TEXT(_2)` is provided. See `samples/parity/icl_transcripts.txt`.
- Generation: `max_new_tokens=168` (~14s), `min_new_tokens=2`, `temperature=0.9`, `top_k=50`, `top_p=1.0`, `repetition_penalty=1.05`
- RNG seed: `1337`

Files:

- Ref `ref_audio.wav`, prompt 1:
  - `icl_ref_audio_gen1_static.wav`
  - `icl_ref_audio_gen1_dynamic.wav`
- Ref `ref_audio.wav`, prompt 2:
  - `icl_ref_audio_gen2_static.wav`
  - `icl_ref_audio_gen2_dynamic.wav`
- Ref `ref_audio_2.wav`, prompt 1:
  - `icl_ref_audio_2_gen1_static.wav`
  - `icl_ref_audio_2_gen1_dynamic.wav`
- Ref `ref_audio_2.wav`, prompt 2:
  - `icl_ref_audio_2_gen2_static.wav`
  - `icl_ref_audio_2_gen2_dynamic.wav`
- Ref `ref_audio_3.wav`, prompt 1:
  - `icl_ref_audio_3_gen1_static.wav`
  - `icl_ref_audio_3_gen1_dynamic.wav`
- Ref `ref_audio_3.wav`, prompt 2:
  - `icl_ref_audio_3_gen2_static.wav`
  - `icl_ref_audio_3_gen2_dynamic.wav`

## Regenerate

```bash
source .venv/bin/activate
python benchmarks/generate_parity_samples.py
python benchmarks/generate_parity_samples_icl.py
```

You can override the model or speakers via environment variables:

```bash
QWEN_TTS_CUSTOM_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
PARITY_SPEAKERS=aiden,serena \
PARITY_MAX_NEW_TOKENS=168 \
PARITY_MIN_NEW_TOKENS=2 \
python benchmarks/generate_parity_samples.py
```

ICL regeneration (optional overrides):

```bash
QWEN_TTS_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-Base \
PARITY_REF_AUDIO=ref_audio.wav \
PARITY_REF_AUDIO_2=ref_audio_2.wav \
PARITY_REF_AUDIO_3=ref_audio_3.wav \
PARITY_REF_TEXT="A short reference transcript." \
PARITY_REF_TEXT_2="A short reference transcript." \
PARITY_REF_TEXT_3="A short reference transcript." \
PARITY_MAX_NEW_TOKENS=168 \
PARITY_MIN_NEW_TOKENS=2 \
python benchmarks/generate_parity_samples_icl.py
```
