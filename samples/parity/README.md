# Parity Sample Set

These samples compare the **static-cache fast path** (CUDA graphs + StaticCache) with the **dynamic-cache parity path** (no graphs, DynamicCache). The algorithms are equivalent, but the attention kernel choice differs, so outputs may not be bit-identical. Use these to compare subjective quality.

## CustomVoice Samples

- Model: `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`
- Language: `English`
- Speakers: `aiden`, `serena`
- Generation: `max_new_tokens=96`, `temperature=0.9`, `top_k=50`, `top_p=1.0`, `repetition_penalty=1.05`
- RNG seed: `1337`

## Prompts

1. "It is a bright morning, and the city is just waking up. Please keep a calm, clear tone for the first words."
2. "Please read this sentence with a steady pace, and pause briefly before the final word so the cadence is clear."

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

- Model: `Qwen/Qwen3-TTS-12Hz-1.7B-Base`
- Language: `English`
- Reference audios: `ref_audio.wav`, `ref_1.wav`
- Reference text: "A short reference transcript."
- Generation: `max_new_tokens=96`, `min_new_tokens=24`, `temperature=0.9`, `top_k=50`, `top_p=1.0`, `repetition_penalty=1.05`
- RNG seed: `1337`

Files:

- Ref `ref_audio.wav`, prompt 1:
  - `icl_ref_audio_gen1_static.wav`
  - `icl_ref_audio_gen1_dynamic.wav`
- Ref `ref_audio.wav`, prompt 2:
  - `icl_ref_audio_gen2_static.wav`
  - `icl_ref_audio_gen2_dynamic.wav`
- Ref `ref_1.wav`, prompt 1:
  - `icl_ref_1_gen1_static.wav`
  - `icl_ref_1_gen1_dynamic.wav`
- Ref `ref_1.wav`, prompt 2:
  - `icl_ref_1_gen2_static.wav`
  - `icl_ref_1_gen2_dynamic.wav`

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
PARITY_MAX_NEW_TOKENS=96 \
python benchmarks/generate_parity_samples.py
```

ICL regeneration (optional overrides):

```bash
QWEN_TTS_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-Base \
PARITY_REF_AUDIO=ref_audio.wav \
PARITY_REF_AUDIO_2=ref_1.wav \
PARITY_REF_TEXT="A short reference transcript." \
PARITY_REF_TEXT_2="A short reference transcript." \
PARITY_MAX_NEW_TOKENS=96 \
PARITY_MIN_NEW_TOKENS=24 \
python benchmarks/generate_parity_samples_icl.py
```
