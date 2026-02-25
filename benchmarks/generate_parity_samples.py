#!/usr/bin/env python3
"""Generate static vs dynamic cache quality samples for README/BLOG."""
import os
import torch
import soundfile as sf

from faster_qwen3_tts import FasterQwen3TTS
from faster_qwen3_tts.generate import fast_generate

MODEL_ID = os.environ.get("QWEN_TTS_CUSTOM_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
OUT_DIR = os.environ.get("PARITY_SAMPLES_DIR", "samples/parity")
SEED = int(os.environ.get("PARITY_SEED", "1337"))

DEFAULT_SPEAKERS = "aiden,serena"
SPEAKERS = [s.strip() for s in os.environ.get("PARITY_SPEAKERS", DEFAULT_SPEAKERS).split(",") if s.strip()]

TEXTS = [
    "It is a bright morning, and the city is just waking up. Please keep a calm, clear tone for the first words.",
    "Please read this sentence with a steady pace, and pause briefly before the final word so the cadence is clear.",
]

MAX_NEW_TOKENS = int(os.environ.get("PARITY_MAX_NEW_TOKENS", "96"))
LANGUAGE = os.environ.get("PARITY_LANGUAGE", "English")

os.makedirs(OUT_DIR, exist_ok=True)

print(f"Loading model {MODEL_ID}...")
model = FasterQwen3TTS.from_pretrained(
    MODEL_ID,
    device="cuda",
    dtype=torch.bfloat16,
    attn_implementation="eager",
    max_seq_len=2048,
)

if model.model.model.tts_model_type != "custom_voice":
    raise RuntimeError("This script expects a CustomVoice model.")

available_speakers = sorted(model.model.model.config.talker_config.spk_id.keys())
missing = [s for s in SPEAKERS if s.lower() not in available_speakers]
if missing:
    raise RuntimeError(f"Speakers not found in model: {missing}. Available: {available_speakers}")


def _decode(codec_ids):
    speech_tokenizer = model.model.model.speech_tokenizer
    audio_list, sr = speech_tokenizer.decode({"audio_codes": codec_ids.unsqueeze(0)})
    audio = audio_list[0]
    if hasattr(audio, "cpu"):
        audio = audio.flatten().cpu().numpy()
    else:
        audio = audio.flatten() if hasattr(audio, "flatten") else audio
    return audio, sr


for speaker in SPEAKERS:
    for idx, text in enumerate(TEXTS, start=1):
        for mode in ("static", "dynamic"):
            torch.manual_seed(SEED)
            m, talker, config, tie, tam, tth, tpe = model._prepare_generation_custom(
                text=text,
                language=LANGUAGE,
                speaker=speaker,
                instruct=None,
            )

            codec_ids, timing = fast_generate(
                talker=talker,
                talker_input_embeds=tie,
                attention_mask=tam,
                trailing_text_hiddens=tth,
                tts_pad_embed=tpe,
                config=config,
                predictor_graph=model.predictor_graph,
                talker_graph=model.talker_graph,
                max_new_tokens=MAX_NEW_TOKENS,
                min_new_tokens=2,
                temperature=0.9,
                top_k=50,
                top_p=1.0,
                do_sample=True,
                repetition_penalty=1.05,
                parity_mode=(mode == "dynamic"),
            )

            if codec_ids is None:
                raise RuntimeError("Generation produced no tokens")

            audio, sr = _decode(codec_ids)

            filename = f"custom_{speaker.lower()}_gen{idx}_{mode}.wav"
            path = os.path.join(OUT_DIR, filename)
            sf.write(path, audio, sr)
            print(f"Wrote {path} ({len(audio) / sr:.2f}s, {timing['ms_per_step']:.1f} ms/step)")

print("Done.")
