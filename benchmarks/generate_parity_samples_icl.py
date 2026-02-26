#!/usr/bin/env python3
"""Generate ICL (voice clone) static vs dynamic cache samples."""
import os
import torch
import soundfile as sf

from faster_qwen3_tts import FasterQwen3TTS
from faster_qwen3_tts.generate import fast_generate

MODEL_ID = os.environ.get("QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
OUT_DIR = os.environ.get("PARITY_SAMPLES_DIR", "samples/parity")
SEED = int(os.environ.get("PARITY_SEED", "1337"))

REFS = [
    {
        "name": "ref_audio",
        "path": os.environ.get("PARITY_REF_AUDIO", "ref_audio.wav"),
        "text": os.environ.get("PARITY_REF_TEXT", ""),
    },
    {
        "name": "ref_audio_2",
        "path": os.environ.get("PARITY_REF_AUDIO_2", "ref_audio_2.wav"),
        "text": os.environ.get("PARITY_REF_TEXT_2", ""),
    },
    {
        "name": "ref_audio_3",
        "path": os.environ.get("PARITY_REF_AUDIO_3", "ref_audio_3.wav"),
        "text": os.environ.get("PARITY_REF_TEXT_3", ""),
    },
]

TEXTS = [
    "We met at the corner cafe after work and talked about weekend plans. The street was quiet, the lights were warm, and the time passed quickly. We stayed a bit longer.",
    "On Tuesday morning I missed the bus, so I walked home through the park. I took the long path and listened to the wind in the trees before heading back. I took my time.",
]

MAX_NEW_TOKENS = int(os.environ.get("PARITY_MAX_NEW_TOKENS", "168"))
MIN_NEW_TOKENS = int(os.environ.get("PARITY_MIN_NEW_TOKENS", "2"))
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
parakeet = None
transcripts = {}

def _load_parakeet():
    from nano_parakeet import from_pretrained as parakeet_from_pretrained
    parakeet = parakeet_from_pretrained(device="cuda")
    parakeet.warmup(duration_s=1.0)
    return parakeet


def _transcribe(parakeet, path: str) -> str:
    import soundfile as sf
    import torchaudio

    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != 16000:
        wav_t = torch.from_numpy(wav).unsqueeze(0)
        wav_t = torchaudio.functional.resample(wav_t, sr, 16000)
        wav = wav_t.squeeze().numpy()
    wav_tensor = torch.from_numpy(wav).cuda()
    return parakeet.transcribe(wav_tensor)


def _decode(codec_ids, ref_codes):
    speech_tokenizer = model.model.model.speech_tokenizer
    if ref_codes is not None:
        ref_codes = ref_codes.to(codec_ids.device)
        codes_for_decode = torch.cat([ref_codes, codec_ids], dim=0)
    else:
        codes_for_decode = codec_ids

    audio_list, sr = speech_tokenizer.decode({"audio_codes": codes_for_decode.unsqueeze(0)})
    audio = audio_list[0]
    if hasattr(audio, "cpu"):
        audio = audio.flatten().cpu().numpy()
    else:
        audio = audio.flatten() if hasattr(audio, "flatten") else audio

    if ref_codes is not None:
        ref_len = ref_codes.shape[0]
        total_len = codes_for_decode.shape[0]
        cut = int(ref_len / max(total_len, 1) * len(audio))
        audio = audio[cut:]

    return audio, sr


for ref in REFS:
    if not os.path.exists(ref["path"]):
        raise RuntimeError(f"Reference audio not found: {ref['path']}")

    transcript = ref["text"].strip()
    if not transcript:
        if parakeet is None:
            print("Loading nano-parakeet for reference transcription...")
            parakeet = _load_parakeet()
        transcript = _transcribe(parakeet, ref["path"]).strip()
        print(f"  Transcribed {ref['name']}: '{transcript}'")
    ref["text"] = transcript
    transcripts[ref["name"]] = transcript

    with torch.inference_mode():
        for idx, text in enumerate(TEXTS, start=1):
            for mode in ("static", "dynamic"):
                torch.manual_seed(SEED)
                m, talker, config, tie, tam, tth, tpe, ref_codes = model._prepare_generation(
                    text=text,
                    ref_audio=ref["path"],
                    ref_text=ref["text"],
                    language=LANGUAGE,
                    xvec_only=False,
                    non_streaming_mode=False,
                    append_silence=True,
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
                    min_new_tokens=MIN_NEW_TOKENS,
                    temperature=0.9,
                    top_k=50,
                    top_p=1.0,
                    do_sample=True,
                    repetition_penalty=1.05,
                    parity_mode=(mode == "dynamic"),
                )

                if codec_ids is None:
                    raise RuntimeError("Generation produced no tokens")

                audio, sr = _decode(codec_ids, ref_codes)

                filename = f"icl_{ref['name']}_gen{idx}_{mode}.wav"
                path = os.path.join(OUT_DIR, filename)
                sf.write(path, audio, sr)
                print(f"Wrote {path} ({len(audio) / sr:.2f}s, {timing['ms_per_step']:.1f} ms/step)")

if transcripts:
    transcript_path = os.path.join(OUT_DIR, "icl_transcripts.txt")
    with open(transcript_path, "w", encoding="utf-8") as f:
        for ref in REFS:
            name = ref["name"]
            path = ref["path"]
            text = transcripts.get(name, "")
            f.write(f"{name} ({path}): {text}\n")
    print(f"Wrote {transcript_path}")

print("Done.")
