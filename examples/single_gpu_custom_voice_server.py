#!/usr/bin/env python3
"""Single-GPU CustomVoice API server for faster-qwen3-tts.

This is the non-Ray backup companion for examples/ray_dual_worker_server.py.
It keeps the same HTTP surface used by local callers, but loads one hot model
instance on one CUDA device and serializes generation inside the process.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import math
import os
import re
import struct
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

try:
    from examples.tts_text_normalizer import has_readable_text, normalize_for_tts
except ModuleNotFoundError:
    from tts_text_normalizer import has_readable_text, normalize_for_tts

DEFAULT_SPEAKER = "Serena"


def _to_pcm16(audio: np.ndarray) -> bytes:
    arr = np.asarray(audio, dtype=np.float32).reshape(-1)
    arr = np.clip(arr, -1.0, 1.0)
    return (arr * 32767.0).astype("<i2").tobytes()


def _wav_header(sample_rate: int, data_len: int) -> bytes:
    byte_rate = sample_rate * 2
    block_align = 2
    riff_size = 36 + data_len
    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", riff_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, byte_rate, block_align, 16))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_len))
    return buf.getvalue()


def _to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    pcm = _to_pcm16(audio)
    return _wav_header(sample_rate, len(pcm)) + pcm


class SpeechRequest(BaseModel):
    model: str = "tts-1"
    input: str = Field(..., min_length=1)
    voice: str = DEFAULT_SPEAKER
    response_format: str = "wav"
    speed: float = 1.0
    instruction: str | None = None
    language: str | None = None
    max_new_tokens: int | None = Field(default=None, ge=1)


class JsonSpeechRequest(SpeechRequest):
    include_audio_b64: bool = True


class CapsWriterSpeakRequest(BaseModel):
    text: str = Field(..., min_length=1)
    speaker: str | None = None
    speed: float = 1.0
    instruction: str | None = None
    language: str | None = None
    max_new_tokens: int | None = Field(default=None, ge=1)


class TTSPlanRequest(BaseModel):
    text: str = Field(..., min_length=1)
    trace_id: str | None = None
    max_chars_per_chunk: int | None = None
    lang_hint: str | None = None


def _sanitize_tts_text(text: str, lang_hint: str | None = None) -> str:
    return normalize_for_tts(text, lang_hint=lang_hint).text


def _split_sentences(text: str) -> list[str]:
    normalized = _sanitize_tts_text(text)
    if not normalized:
        return []
    parts = re.split(r"(?<=[。.!！？?；;])\s*", normalized)
    return [part.strip() for part in parts if has_readable_text(part)]


def _estimate_audio_s(text: str) -> float:
    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text or ""))
    other_chars = max(0, len(text or "") - chinese_chars)
    return max(1.5, chinese_chars * 0.23 + other_chars * 0.08)


def _estimate_max_new_tokens(text: str, hard_cap: int) -> int:
    hard_cap = max(1, int(hard_cap))
    min_cap = min(64, hard_cap)
    estimated_steps = _estimate_audio_s(text) * 12.0
    value = int(math.ceil(estimated_steps * 1.35 + 24))
    return max(min_cap, min(hard_cap, value))


def _resolve_max_new_tokens(text: str, requested: int | None, hard_cap: int) -> int:
    hard_cap = max(1, int(hard_cap))
    min_cap = min(64, hard_cap)
    if requested is None:
        return _estimate_max_new_tokens(text, hard_cap)
    return max(min_cap, min(hard_cap, int(requested)))


def _split_tts_text_into_chunks(
    text: str,
    *,
    max_chars: int,
    min_chars: int,
    max_segments: int,
) -> tuple[list[str], bool]:
    content = _sanitize_tts_text(text)
    if not content:
        return [], False

    max_chars = max(20, int(max_chars))
    min_chars = max(1, min(int(min_chars), max_chars))
    max_segments = max(1, int(max_segments))
    if len(content) <= max_chars:
        return [content], False
    sentences = _split_sentences(content) or [content]

    chunks: list[str] = []
    current = ""
    truncated = False

    def split_long_text(value: str) -> tuple[str, str]:
        window = value[:max_chars]
        boundary = -1
        for pattern in (r"[。.!！？?；;，,]\s*", r"\s+"):
            matches = list(re.finditer(pattern, window))
            if matches:
                boundary = max(boundary, matches[-1].end())
        if boundary < max(12, min_chars // 2):
            boundary = max_chars
        return value[:boundary].strip(), value[boundary:].strip()

    def push(value: str) -> None:
        nonlocal truncated
        item = value.strip()
        if not has_readable_text(item):
            return
        if len(chunks) >= max_segments:
            truncated = True
            return
        chunks.append(item)

    for sentence in sentences:
        remaining = sentence
        while len(remaining) > max_chars:
            if current:
                push(current)
                current = ""
            head, remaining = split_long_text(remaining)
            push(head)
            if truncated:
                return chunks, True

        candidate = (current + remaining).strip()
        if not current:
            current = remaining
        elif len(candidate) <= max_chars:
            current = candidate
        else:
            push(current)
            current = remaining
            if truncated:
                return chunks, True

        if len(current) >= min_chars and _estimate_audio_s(current) >= 3.0:
            push(current)
            current = ""
            if truncated:
                return chunks, True

    if current:
        push(current)
    return chunks, truncated


class SingleGPUWorker:
    def __init__(
        self,
        model_path: str,
        attn: str,
        dtype_name: str,
        language: str,
        default_speaker: str,
        chunk_size: int,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
        max_seq_len: int,
        warmup_text: str,
        device: str,
    ) -> None:
        import torch
        from faster_qwen3_tts import FasterQwen3TTS

        self.worker_id = 0
        self.model_path = model_path
        self.attn = attn
        self.dtype_name = dtype_name
        self.language = language
        self.default_speaker = default_speaker
        self.chunk_size = chunk_size
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p
        self.max_seq_len = max_seq_len
        self.device = device
        self.cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        self._lock = threading.Lock()

        dtype = getattr(torch, dtype_name)
        t0 = time.perf_counter()
        self.model = FasterQwen3TTS.from_pretrained(
            model_path,
            device=device,
            dtype=dtype,
            attn_implementation=attn,
            max_seq_len=max_seq_len,
        )
        predictor_graph = getattr(self.model, "predictor_graph", None)
        if predictor_graph is not None:
            predictor_graph.do_sample = do_sample
            predictor_graph.temperature = temperature
            predictor_graph.top_p = top_p
        self.load_elapsed_s = time.perf_counter() - t0
        self.speakers = self._load_speakers()
        self.default_speaker = self._resolve_speaker(default_speaker)
        self.warmup = self.synthesize(
            text=warmup_text,
            speaker=self.default_speaker,
            language=language,
            instruction=None,
        )

    def _load_speakers(self) -> list[str]:
        try:
            raw = self.model.model.get_supported_speakers()
        except Exception:
            raw = []
        speakers = []
        for item in raw or []:
            value = str(item).strip()
            if value:
                speakers.append(value)
        return list(dict.fromkeys(speakers))

    def _resolve_speaker(self, requested: str | None) -> str:
        requested_value = (requested or "").strip()
        if not self.speakers:
            return requested_value
        lut = {speaker.lower(): speaker for speaker in self.speakers}
        if requested_value and requested_value.lower() in lut:
            return lut[requested_value.lower()]
        if self.default_speaker and self.default_speaker.lower() in lut:
            return lut[self.default_speaker.lower()]
        return self.speakers[0]

    def synthesize(
        self,
        text: str,
        speaker: str | None = None,
        language: str | None = None,
        instruction: str | None = None,
        max_new_tokens: int | None = None,
    ) -> dict[str, Any]:
        import torch

        content = (text or "").strip()
        if not content:
            raise ValueError("Text is empty")
        resolved_speaker = self._resolve_speaker(speaker)
        resolved_language = (language or self.language or "Auto").strip()
        effective_max_new_tokens = _resolve_max_new_tokens(content, max_new_tokens, self.max_new_tokens)

        with self._lock:
            chunks: list[np.ndarray] = []
            sr = 24000
            timings: list[dict[str, Any]] = []
            first_audio_s: float | None = None

            if self.device.startswith("cuda"):
                torch.cuda.synchronize()
            start = time.perf_counter()
            for chunk, sr, timing in self.model.generate_custom_voice_streaming(
                text=content,
                speaker=resolved_speaker,
                language=resolved_language,
                instruct=(instruction or None),
                max_new_tokens=effective_max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=self.do_sample,
                chunk_size=self.chunk_size,
            ):
                if first_audio_s is None:
                    if self.device.startswith("cuda"):
                        torch.cuda.synchronize()
                    first_audio_s = time.perf_counter() - start
                chunks.append(np.asarray(chunk, dtype=np.float32).reshape(-1))
                timings.append(dict(timing or {}))
            if self.device.startswith("cuda"):
                torch.cuda.synchronize()
            elapsed_s = time.perf_counter() - start

        if not chunks:
            raise RuntimeError("faster-qwen3-tts returned empty audio")

        audio = np.concatenate(chunks)
        audio_s = len(audio) / int(sr)
        rtf = audio_s / elapsed_s if elapsed_s > 0 else 0.0
        estimated_audio_s = _estimate_audio_s(content)
        expected_cap_s = effective_max_new_tokens / 12.0
        hit_token_cap = audio_s >= max(0.0, expected_cap_s - 0.08)
        suspicious_duration = audio_s > max(12.0, estimated_audio_s * 2.5)
        wav_bytes = _to_wav_bytes(audio, int(sr))
        print(
            "[TTS] "
            f"worker=0 gpu={self.cuda_visible_devices or self.device} "
            f"text_len={len(content)} speaker={resolved_speaker} "
            f"max_new_tokens={effective_max_new_tokens} "
            f"audio_s={audio_s:.3f} elapsed_s={elapsed_s:.3f} "
            f"rtf={rtf:.3f} ttfa_s={(first_audio_s or 0.0):.3f} "
            f"hit_token_cap={hit_token_cap} suspicious_duration={suspicious_duration}",
            flush=True,
        )

        return {
            "worker_id": 0,
            "gpu_ids": [self.cuda_visible_devices or self.device],
            "cuda_visible_devices": self.cuda_visible_devices,
            "speaker": resolved_speaker,
            "language": resolved_language,
            "sample_rate": int(sr),
            "audio_s": audio_s,
            "elapsed_s": elapsed_s,
            "ttfa_s": first_audio_s,
            "rtf": rtf,
            "text_len": len(content),
            "max_new_tokens": effective_max_new_tokens,
            "estimated_audio_s": estimated_audio_s,
            "hit_token_cap": hit_token_cap,
            "suspicious_duration": suspicious_duration,
            "bytes": wav_bytes,
            "timings": timings,
        }

    def health(self) -> dict[str, Any]:
        return {
            "worker_id": 0,
            "status": "ready",
            "gpu_ids": [self.cuda_visible_devices or self.device],
            "cuda_visible_devices": self.cuda_visible_devices,
            "model_path": self.model_path,
            "attn": self.attn,
            "dtype": self.dtype_name,
            "language": self.language,
            "default_speaker": self.default_speaker,
            "speakers": self.speakers,
            "chunk_size": self.chunk_size,
            "max_new_tokens": self.max_new_tokens,
            "load_elapsed_s": self.load_elapsed_s,
            "warmup_rtf": self.warmup.get("rtf"),
            "warmup_audio_s": self.warmup.get("audio_s"),
            "warmup_elapsed_s": self.warmup.get("elapsed_s"),
        }


@dataclass
class AppState:
    worker: SingleGPUWorker


app = FastAPI(title="Single-GPU faster-qwen3-tts")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        "X-TTS-Worker-Id",
        "X-TTS-GPU-Ids",
        "X-TTS-Audio-Seconds",
        "X-TTS-Elapsed-Seconds",
        "X-TTS-RTF",
        "X-TTS-TTFA-Seconds",
        "X-TTS-Speaker",
        "X-TTS-Normalizer",
        "X-TTS-Hit-Token-Cap",
        "X-TTS-Suspicious-Duration",
    ],
)
state: AppState | None = None


def _tts_response_headers(result: dict[str, Any]) -> dict[str, str]:
    headers = {
        "X-TTS-Worker-Id": str(result["worker_id"]),
        "X-TTS-GPU-Ids": ",".join(str(x) for x in result["gpu_ids"]),
        "X-TTS-Audio-Seconds": f"{result['audio_s']:.6f}",
        "X-TTS-Elapsed-Seconds": f"{result['elapsed_s']:.6f}",
        "X-TTS-RTF": f"{result['rtf']:.6f}",
        "X-TTS-Hit-Token-Cap": str(bool(result.get("hit_token_cap"))).lower(),
        "X-TTS-Suspicious-Duration": str(bool(result.get("suspicious_duration"))).lower(),
    }
    if result.get("ttfa_s") is not None:
        headers["X-TTS-TTFA-Seconds"] = f"{result['ttfa_s']:.6f}"
    if result.get("speaker"):
        headers["X-TTS-Speaker"] = result["speaker"]
    if result.get("normalizer"):
        headers["X-TTS-Normalizer"] = result["normalizer"]
    return headers


async def _synthesize(req: SpeechRequest) -> dict[str, Any]:
    if state is None:
        raise HTTPException(status_code=503, detail="Server not initialized")
    normalized = normalize_for_tts(req.input, lang_hint=req.language)
    content = normalized.text
    if not has_readable_text(content):
        raise HTTPException(status_code=400, detail="No readable TTS text after sanitization")
    try:
        result = await asyncio.to_thread(
            state.worker.synthesize,
            content,
            req.voice,
            req.language,
            req.instruction,
            req.max_new_tokens,
        )
        result["normalizer"] = normalized.normalizer
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc


@app.get("/health")
async def health() -> dict[str, Any]:
    if state is None:
        raise HTTPException(status_code=503, detail="Server not initialized")
    return {"status": "ok", "workers": [state.worker.health()]}


@app.get("/api/status")
async def api_status() -> dict[str, Any]:
    if state is None:
        return {
            "success": False,
            "status": "loading",
            "tts_enabled": True,
            "tts_model_loaded": False,
            "error": "Server not initialized",
        }
    row = state.worker.health()
    return {
        "success": True,
        "status": "ready",
        "tts_enabled": True,
        "tts_model_loaded": True,
        "tts_backend": "single_gpu",
        "default_speaker": row.get("default_speaker"),
        "workers_ready": 1,
        "workers": [row],
    }


@app.post("/api/tts/load")
async def tts_load() -> dict[str, Any]:
    if state is None:
        raise HTTPException(status_code=503, detail="Server not initialized")
    return {"success": True, "status": "ready", "already_loaded": True}


@app.post("/api/tts/plan")
async def tts_plan(req: TTSPlanRequest) -> JSONResponse:
    original_content = (req.text or "").strip()
    normalized = normalize_for_tts(original_content, lang_hint=req.lang_hint)
    content = normalized.text
    if not original_content:
        raise HTTPException(status_code=400, detail="No text provided")
    if not has_readable_text(content):
        return JSONResponse(
            {
                "success": True,
                "trace_id": req.trace_id or "",
                "text": "",
                "chunks": [],
                "total_chunks": 0,
                "truncated": False,
                "sanitized": True,
                "normalizer": normalized.normalizer,
            }
        )

    max_chars = req.max_chars_per_chunk or int(os.getenv("QWEN_TTS_PLAN_MAX_CHARS", "90"))
    max_chars = max(20, min(int(max_chars), int(os.getenv("QWEN_TTS_PLAN_MAX_CHARS_LIMIT", "240"))))
    min_chars = max(1, min(int(os.getenv("QWEN_TTS_PLAN_MIN_CHARS", "28")), max_chars))
    max_segments = max(1, int(os.getenv("QWEN_TTS_PLAN_MAX_SEGMENTS", "120")))

    chunks, truncated = _split_tts_text_into_chunks(
        content,
        max_chars=max_chars,
        min_chars=min_chars,
        max_segments=max_segments,
    )
    payload_chunks = [
        {
            "index": idx,
            "text": chunk,
            "est_chars": len(chunk),
            "estimated_audio_s": _estimate_audio_s(chunk),
            "max_new_tokens": _estimate_max_new_tokens(chunk, 512),
        }
        for idx, chunk in enumerate(chunks)
    ]
    longest = max((len(chunk) for chunk in chunks), default=0)
    print(
        "[TTS] "
        f"plan trace_id={req.trace_id or '-'} text_len={len(content)} "
        f"chunks={len(chunks)} max_chars={max_chars} longest={longest} "
        f"truncated={truncated} sanitized={normalized.changed} "
        f"normalizer={normalized.normalizer}",
        flush=True,
    )
    return JSONResponse(
        {
            "success": True,
            "trace_id": req.trace_id or "",
            "text": content,
            "chunks": payload_chunks,
            "total_chunks": len(payload_chunks),
            "truncated": truncated,
            "sanitized": normalized.changed,
            "normalizer": normalized.normalizer,
        }
    )


@app.post("/v1/audio/speech")
async def create_speech(req: SpeechRequest) -> Response:
    fmt = req.response_format.lower()
    if fmt != "wav":
        raise HTTPException(status_code=400, detail="Only response_format='wav' is supported")
    result = await _synthesize(req)
    return Response(content=result["bytes"], media_type="audio/wav", headers=_tts_response_headers(result))


@app.post("/api/tts/speak_json")
async def speak_json(req: JsonSpeechRequest) -> JSONResponse:
    result = await _synthesize(req)
    wav_bytes = result.pop("bytes")
    if req.include_audio_b64:
        result["audio_b64"] = base64.b64encode(wav_bytes).decode("ascii")
    result["wav_bytes"] = len(wav_bytes)
    return JSONResponse(result)


@app.post("/api/tts/speak")
async def capswriter_speak(req: CapsWriterSpeakRequest) -> Response:
    speech_req = SpeechRequest(
        input=req.text,
        voice=req.speaker or DEFAULT_SPEAKER,
        speed=req.speed,
        instruction=req.instruction,
        language=req.language,
        max_new_tokens=req.max_new_tokens,
    )
    result = await _synthesize(speech_req)
    return Response(content=result["bytes"], media_type="audio/wav", headers=_tts_response_headers(result))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.getenv("QWEN_TTS_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("QWEN_TTS_PORT", "8091")))
    parser.add_argument("--model", default=os.getenv("QWEN_TTS_MODEL", "/home/admin/models/Qwen3-TTS-12Hz-1.7B-CustomVoice"))
    parser.add_argument("--device", default=os.getenv("QWEN_TTS_DEVICE", "cuda:0"))
    parser.add_argument("--attn", default=os.getenv("QWEN_TTS_ATTN", "sdpa"), choices=["sdpa", "eager", "flash_attention_2"])
    parser.add_argument("--dtype", default=os.getenv("QWEN_TTS_DTYPE", "float16"), choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--language", default=os.getenv("QWEN_TTS_LANGUAGE", "Auto"))
    parser.add_argument("--speaker", default=os.getenv("QWEN_TTS_SPEAKER", DEFAULT_SPEAKER))
    parser.add_argument("--chunk-size", type=int, default=int(os.getenv("QWEN_TTS_CHUNK_SIZE", "8")))
    parser.add_argument("--max-new-tokens", type=int, default=int(os.getenv("QWEN_TTS_MAX_NEW_TOKENS", "512")))
    parser.add_argument("--max-seq-len", type=int, default=int(os.getenv("QWEN_TTS_MAX_SEQ_LEN", "2048")))
    parser.add_argument("--do-sample", action="store_true", default=os.getenv("QWEN_TTS_DO_SAMPLE", "0") == "1")
    parser.add_argument("--temperature", type=float, default=float(os.getenv("QWEN_TTS_TEMPERATURE", "0.8")))
    parser.add_argument("--top-p", type=float, default=float(os.getenv("QWEN_TTS_TOP_P", "1.0")))
    parser.add_argument("--warmup-text", default=os.getenv("QWEN_TTS_WARMUP_TEXT", "预热。"))
    return parser.parse_args()


def main() -> None:
    global state
    args = _parse_args()
    state = AppState(
        worker=SingleGPUWorker(
            model_path=args.model,
            attn=args.attn,
            dtype_name=args.dtype,
            language=args.language,
            default_speaker=args.speaker,
            chunk_size=args.chunk_size,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            max_seq_len=args.max_seq_len,
            warmup_text=args.warmup_text,
            device=args.device,
        )
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
