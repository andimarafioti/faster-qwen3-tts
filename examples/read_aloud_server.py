#!/usr/bin/env python3
"""Small read-aloud web server for Faster Qwen3-TTS.

Usage:
    python examples/read_aloud_server.py --backend ggml --port 7861
    python examples/read_aloud_server.py --backend torch --model Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign
"""
from __future__ import annotations

import argparse
import io
import hmac
import os
import json
import re
import sys
import threading
import time
import uuid
import wave
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from faster_qwen3_tts import FasterQwen3TTS  # noqa: E402


BASE_DIR = Path(__file__).resolve().parent
INDEX_HTML = BASE_DIR / "read_aloud.html"
OUTPUT_DIR = Path("/tmp/faster-qwen3-tts-read-aloud")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_LOCK = threading.Lock()
MODEL = None
ARGS = None
STREAM_REQUESTS: dict[str, SpeakRequest] = {}
STREAM_REQUESTS_LOCK = threading.Lock()


class SpeakRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10000)
    language: str = "English"
    instruction: str = Field(
        default="Warm, natural narrator with clear diction and a steady pace.",
        max_length=600,
    )
    dialogue_mode: bool = False
    speaker_a_instruction: str = Field(
        default="Adult male speaker with a calm, grounded tone and clear diction.",
        max_length=600,
    )
    speaker_b_instruction: str = Field(
        default="Adult female speaker with a bright, conversational tone and clear diction.",
        max_length=600,
    )
    speaker_pause_ms: int = Field(default=250, ge=0, le=2000)
    temperature: float = Field(default=0.9, ge=0.1, le=2.0)
    top_k: int = Field(default=50, ge=1, le=200)
    top_p: float = Field(default=1.0, ge=0.05, le=1.0)
    repetition_penalty: float = Field(default=1.05, ge=0.8, le=2.0)
    chunk_size: int = Field(default=8, ge=2, le=24)
    max_new_tokens: int = Field(default=8192, ge=24, le=8192)
    seed: int | None = Field(default=None, ge=0, le=2**31 - 1)
    greedy: bool = False


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--backend", choices=("ggml", "torch"), default="ggml")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        help="VoiceDesign model is recommended for the customization panel.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--quant", default="BF16", help="GGML quantization, e.g. BF16.")
    parser.add_argument("--gguf-model")
    parser.add_argument("--gguf-codec")
    parser.add_argument("--qwentts-lib")
    parser.add_argument("--qwentts-ref-cache-dir")
    parser.add_argument("--max-seq-len", type=int, default=12288, help="Torch backend static cache length; raise for longer read-aloud renders.")
    parser.add_argument("--no-preload", action="store_true")
    return parser.parse_args()


def _torch_dtype(value: str):
    if value == "bf16":
        return torch.bfloat16
    if value == "fp16":
        return torch.float16
    return torch.float32


def _load_model():
    global MODEL
    if MODEL is not None:
        return MODEL

    with MODEL_LOCK:
        if MODEL is not None:
            return MODEL

        if ARGS.backend == "ggml":
            MODEL = FasterQwen3TTS.from_pretrained(
                ARGS.model,
                backend="ggml",
                quant=ARGS.quant,
                gguf_talker_path=ARGS.gguf_model,
                gguf_codec_path=ARGS.gguf_codec,
                qwentts_library_path=ARGS.qwentts_lib,
                qwentts_ref_cache_dir=ARGS.qwentts_ref_cache_dir,
            )
        else:
            MODEL = FasterQwen3TTS.from_pretrained(
                ARGS.model,
                device=ARGS.device,
                dtype=_torch_dtype(ARGS.dtype),
                attn_implementation="sdpa",
                max_seq_len=ARGS.max_seq_len,
            )
        return MODEL


def _normalize_audio(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    return np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)


def _pcm16(audio: np.ndarray) -> bytes:
    audio = np.clip(_normalize_audio(audio), -1.0, 1.0)
    return (audio * 32767.0).astype("<i2").tobytes()


def _wav_header(sample_rate: int, channels: int = 1, bits_per_sample: int = 16) -> bytes:
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    data_size = 0x7FFFFFF0
    riff_size = min(36 + data_size, 0xFFFFFFFF)
    return (
        b"RIFF"
        + riff_size.to_bytes(4, "little")
        + b"WAVEfmt "
        + (16).to_bytes(4, "little")
        + (1).to_bytes(2, "little")
        + channels.to_bytes(2, "little")
        + sample_rate.to_bytes(4, "little")
        + byte_rate.to_bytes(4, "little")
        + block_align.to_bytes(2, "little")
        + bits_per_sample.to_bytes(2, "little")
        + b"data"
        + data_size.to_bytes(4, "little")
    )



SPEAKER_LINE_RE = re.compile(
    r"^\s*(?:\*\*)?([A-Za-z][A-Za-z0-9 _-]{0,30}|Speaker\s+[A-Za-z0-9]+)(?:\*\*)?\s*:\s*(.+?)\s*$",
    re.IGNORECASE,
)


def _speaker_key(label: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", label.lower()).strip()
    parts = normalized.split()
    if parts and parts[0] == "speaker" and len(parts) > 1:
        return parts[1]
    return parts[0] if parts else "a"


def _parse_dialogue_segments(text: str) -> list[tuple[str, str]]:
    segments: list[tuple[str, str]] = []
    current_speaker: str | None = None
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_speaker, current_lines
        if current_speaker and current_lines:
            segment_text = " ".join(line.strip() for line in current_lines if line.strip())
            if segment_text:
                if segments and segments[-1][0] == current_speaker:
                    previous_speaker, previous_text = segments[-1]
                    segments[-1] = (previous_speaker, f"{previous_text} {segment_text}")
                else:
                    segments.append((current_speaker, segment_text))
        current_lines = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = SPEAKER_LINE_RE.match(line)
        if match:
            flush()
            current_speaker = _speaker_key(match.group(1))
            current_lines = [match.group(2).strip()]
        elif current_speaker:
            current_lines.append(line)
        else:
            return []
    flush()
    return segments


def _speaker_instruction(req: SpeakRequest, speaker: str) -> str:
    if speaker in {"a", "1", "one"}:
        specific = req.speaker_a_instruction.strip()
    elif speaker in {"b", "2", "two"}:
        specific = req.speaker_b_instruction.strip()
    else:
        specific = req.instruction.strip()
    general = req.instruction.strip()
    if general and specific and general != specific:
        return f"{specific} {general}"
    return specific or general


def _generate_voice_design_audio(model, req: SpeakRequest, text: str, instruct: str) -> tuple[np.ndarray, int]:
    audio_list, sample_rate = model.generate_voice_design(
        text=text,
        instruct=instruct,
        language=req.language,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
        do_sample=not req.greedy,
        repetition_penalty=req.repetition_penalty,
    )
    return _normalize_audio(audio_list[0]), int(sample_rate)


def _render_dialogue_file(req: SpeakRequest, model, started: float) -> dict | None:
    segments = _parse_dialogue_segments(req.text)
    if not segments:
        return None

    rendered: list[np.ndarray] = []
    sample_rate: int | None = None
    for index, (speaker, segment_text) in enumerate(segments):
        instruct = _speaker_instruction(req, speaker)
        audio, sr = _generate_voice_design_audio(model, req, segment_text, instruct)
        if sample_rate is None:
            sample_rate = sr
        elif sr != sample_rate:
            raise RuntimeError(f"Dialogue segment sample rate changed from {sample_rate} to {sr}")
        rendered.append(audio)
        if index < len(segments) - 1 and req.speaker_pause_ms > 0:
            rendered.append(np.zeros(int(sr * req.speaker_pause_ms / 1000), dtype=np.float32))

    audio = np.concatenate(rendered) if rendered else np.zeros(1, dtype=np.float32)
    audio_id = uuid.uuid4().hex
    output_path = OUTPUT_DIR / f"{audio_id}.wav"
    sf.write(output_path, audio, int(sample_rate or 24000))
    duration = len(audio) / float(sample_rate or 24000)
    elapsed = time.perf_counter() - started
    return {
        "id": audio_id,
        "url": f"/audio/{audio_id}.wav",
        "sample_rate": int(sample_rate or 24000),
        "duration_s": duration,
        "elapsed_s": elapsed,
        "rtf": duration / elapsed if elapsed > 0 else 0.0,
        "segments": len(segments),
    }


def _stream_chunks(req: SpeakRequest) -> Iterable[bytes]:
    if req.seed is not None:
        torch.manual_seed(req.seed)

    model = _load_model()
    first = True
    for audio_chunk, sample_rate, _timing in model.generate_voice_design_streaming(
        text=req.text,
        instruct=req.instruction,
        language=req.language,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
        do_sample=not req.greedy,
        repetition_penalty=req.repetition_penalty,
        chunk_size=req.chunk_size,
    ):
        if first:
            yield _wav_header(int(sample_rate))
            first = False
        pcm = _pcm16(audio_chunk)
        if pcm:
            yield pcm

    if first:
        yield _wav_header(24000)


def _render_file(req: SpeakRequest) -> dict:
    if req.seed is not None:
        torch.manual_seed(req.seed)

    model = _load_model()
    started = time.perf_counter()
    if req.dialogue_mode:
        dialogue_result = _render_dialogue_file(req, model, started)
        if dialogue_result is not None:
            return dialogue_result

    audio, sample_rate = _generate_voice_design_audio(
        model,
        req,
        req.text,
        req.instruction,
    )
    audio_id = uuid.uuid4().hex
    output_path = OUTPUT_DIR / f"{audio_id}.wav"
    sf.write(output_path, audio, int(sample_rate))
    duration = len(audio) / float(sample_rate) if sample_rate else 0.0
    elapsed = time.perf_counter() - started
    return {
        "id": audio_id,
        "url": f"/audio/{audio_id}.wav",
        "sample_rate": int(sample_rate),
        "duration_s": duration,
        "elapsed_s": elapsed,
        "rtf": duration / elapsed if elapsed > 0 else 0.0,
    }


def _require_passcode(x_read_aloud_passcode: str | None = Header(default=None)) -> None:
    expected = os.environ.get("READ_ALOUD_PASSCODE", "")
    if not expected:
        return
    if not x_read_aloud_passcode or not hmac.compare_digest(x_read_aloud_passcode, expected):
        raise HTTPException(status_code=401, detail="Passcode required")


app = FastAPI(title="Faster Qwen3-TTS Read Aloud")


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(INDEX_HTML.read_text(encoding="utf-8"))


@app.get("/api/status")
def status():
    return {
        "backend": ARGS.backend,
        "model": ARGS.model,
        "loaded": MODEL is not None,
        "mode": "voice_design",
    }


@app.post("/api/speak")
def speak(req: SpeakRequest, _auth: None = Depends(_require_passcode)):
    try:
        return JSONResponse(_render_file(req))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/speak/stream")
def speak_stream(req: SpeakRequest, _auth: None = Depends(_require_passcode)):
    try:
        return StreamingResponse(_stream_chunks(req), media_type="audio/wav")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/speak/stream-url")
def speak_stream_url(req: SpeakRequest, _auth: None = Depends(_require_passcode)):
    stream_id = uuid.uuid4().hex
    with STREAM_REQUESTS_LOCK:
        STREAM_REQUESTS[stream_id] = req
    return {"url": f"/api/speak/stream/{stream_id}"}


@app.get("/api/speak/stream/{stream_id}")
def speak_stream_get(stream_id: str):
    with STREAM_REQUESTS_LOCK:
        req = STREAM_REQUESTS.pop(stream_id, None)
    if req is None:
        raise HTTPException(status_code=404, detail="Stream request not found")
    try:
        return StreamingResponse(_stream_chunks(req), media_type="audio/wav")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/audio/{name}")
def audio_file(name: str):
    if not name.endswith(".wav"):
        name = f"{name}.wav"
    path = OUTPUT_DIR / name
    if not path.exists() or path.parent != OUTPUT_DIR:
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(path, media_type="audio/wav", filename=name)


@app.get("/api/request-template")
def request_template():
    return JSONResponse(json.loads(SpeakRequest(text="Hello.").json()))


def main() -> None:
    global ARGS
    ARGS = _parse_args()
    if not ARGS.no_preload:
        print(f"Loading {ARGS.model} with {ARGS.backend} backend...")
        _load_model()
    print(f"Open http://{ARGS.host}:{ARGS.port}")
    uvicorn.run(app, host=ARGS.host, port=ARGS.port)


if __name__ == "__main__":
    main()
