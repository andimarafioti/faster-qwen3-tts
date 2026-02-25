# Parity Sample Set

These samples compare the **static-cache fast path** (CUDA graphs + StaticCache) with the **dynamic-cache parity path** (no graphs, DynamicCache). The algorithms are equivalent, but the attention kernel choice differs, so outputs may not be bit-identical. Use these to compare subjective quality.

## Model

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

## Regenerate

```bash
source .venv/bin/activate
python benchmarks/generate_parity_samples.py
```

You can override the model or speakers via environment variables:

```bash
QWEN_TTS_CUSTOM_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
PARITY_SPEAKERS=aiden,serena \
PARITY_MAX_NEW_TOKENS=96 \
python benchmarks/generate_parity_samples.py
```
