#!/usr/bin/env python3
"""Ray-backed dual-GPU OpenAI-compatible server for faster-qwen3-tts.

This server is designed for hosts with two usable CUDA GPUs. Start it with a
filtered CUDA device list, for example:

    CUDA_VISIBLE_DEVICES=1,2 python examples/ray_dual_worker_server.py \
      --model /home/ivan/models/Qwen3-TTS-12Hz-1.7B-CustomVoice

Each Ray actor reserves one GPU and loads one hot TTS model instance. Requests
are routed round-robin across actors to improve concurrent throughput.
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
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import ray
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from examples.tts_text_normalizer import has_readable_text, normalize_for_tts

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


def _wav_stream_header(sample_rate: int) -> bytes:
    return _wav_header(sample_rate, 0xFFFFFFFF)


def _to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    pcm = _to_pcm16(audio)
    return _wav_header(sample_rate, len(pcm)) + pcm


def _wav_payload(wav_bytes: bytes) -> bytes:
    if len(wav_bytes) < 44 or wav_bytes[:4] != b"RIFF" or wav_bytes[8:12] != b"WAVE":
        raise ValueError("Expected 16-bit PCM WAV bytes")
    data_pos = wav_bytes.find(b"data")
    if data_pos < 0:
        raise ValueError("WAV data chunk not found")
    return wav_bytes[data_pos + 8 :]


def _silence_pcm(sample_rate: int, milliseconds: int) -> bytes:
    samples = max(0, int(sample_rate * milliseconds / 1000))
    return b"\x00\x00" * samples


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


class PlaybackFirstRequest(SpeechRequest):
    include_audio_b64: bool = False
    head_target_audio_s: float = 12.0
    head_min_chars: int = 40
    head_max_chars: int = 140
    join_silence_ms: int = 180


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
    # Verified Chinese samples on this model are roughly 4-5 chars/s. Bias high
    # so the first segment is long enough to cover the tail's generation time.
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


def _split_playback_first(
    text: str,
    *,
    target_audio_s: float,
    min_chars: int,
    max_chars: int,
) -> list[dict[str, Any]]:
    content = _sanitize_tts_text(text)
    if not has_readable_text(content):
        return []
    if len(content) <= max(min_chars, 80):
        return [{"index": 0, "role": "full", "text": content}]

    sentences = _split_sentences(content)
    if len(sentences) < 2:
        split_at = min(len(content), max(min_chars, min(max_chars, len(content) // 2)))
        return [
            {"index": 0, "role": "head", "text": content[:split_at].strip()},
            {"index": 1, "role": "tail", "text": content[split_at:].strip()},
        ]

    head: list[str] = []
    for sentence in sentences:
        candidate = "".join(head + [sentence])
        head_text = "".join(head)
        head_is_usable = len(head_text) >= min_chars and _estimate_audio_s(head_text) >= target_audio_s * 0.8
        if head and len(candidate) > max_chars and head_is_usable:
            break
        head.append(sentence)
        if len(candidate) >= min_chars and _estimate_audio_s(candidate) >= target_audio_s:
            break

    if not head:
        head = [sentences[0]]
    if len(head) == len(sentences):
        head = sentences[: max(1, len(sentences) // 2)]

    head_text = "".join(head).strip()
    tail_text = "".join(sentences[len(head) :]).strip()
    if not tail_text:
        return [{"index": 0, "role": "full", "text": content}]
    return [
        {"index": 0, "role": "head", "text": head_text},
        {"index": 1, "role": "tail", "text": tail_text},
    ]


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


@ray.remote(num_gpus=1)
class TTSWorker:
    def __init__(
        self,
        worker_id: int,
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
    ) -> None:
        import torch
        from faster_qwen3_tts import FasterQwen3TTS

        self.worker_id = worker_id
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
        self.gpu_ids = ray.get_gpu_ids()
        self.cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")

        dtype = getattr(torch, dtype_name)
        t0 = time.perf_counter()
        self.model = FasterQwen3TTS.from_pretrained(
            model_path,
            device="cuda:0",
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
        self.warmup = self._synthesize(
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

    def _estimate_max_tokens(self, text: str) -> int:
        return _estimate_max_new_tokens(text, self.max_new_tokens)

    def _synthesize(
        self,
        text: str,
        speaker: str | None,
        language: str | None,
        instruction: str | None,
        max_new_tokens: int | None = None,
    ) -> dict[str, Any]:
        import torch

        content = (text or "").strip()
        if not content:
            raise ValueError("Text is empty")
        resolved_speaker = self._resolve_speaker(speaker)
        resolved_language = (language or self.language or "Auto").strip()

        chunks: list[np.ndarray] = []
        sr = 24000
        timings: list[dict[str, Any]] = []
        first_audio_s: float | None = None

        effective_max_new_tokens = _resolve_max_new_tokens(content, max_new_tokens, self.max_new_tokens)

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
                torch.cuda.synchronize()
                first_audio_s = time.perf_counter() - start
            chunks.append(np.asarray(chunk, dtype=np.float32).reshape(-1))
            timings.append(dict(timing or {}))
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
            f"worker={self.worker_id} gpu={','.join(str(x) for x in self.gpu_ids)} "
            f"text_len={len(content)} speaker={resolved_speaker} "
            f"max_new_tokens={effective_max_new_tokens} "
            f"audio_s={audio_s:.3f} elapsed_s={elapsed_s:.3f} "
            f"rtf={rtf:.3f} ttfa_s={(first_audio_s or 0.0):.3f} "
            f"hit_token_cap={hit_token_cap} suspicious_duration={suspicious_duration}",
            flush=True,
        )

        return {
            "worker_id": self.worker_id,
            "gpu_ids": self.gpu_ids,
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

    def synthesize(
        self,
        text: str,
        speaker: str | None = None,
        language: str | None = None,
        instruction: str | None = None,
        max_new_tokens: int | None = None,
    ) -> dict[str, Any]:
        return self._synthesize(text, speaker, language, instruction, max_new_tokens)

    def health(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "status": "ready",
            "gpu_ids": self.gpu_ids,
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
    workers: list[Any]
    next_worker: int = 0

    def pick_worker(self) -> Any:
        worker = self.workers[self.next_worker % len(self.workers)]
        self.next_worker += 1
        return worker

    def pick_two_workers(self) -> tuple[Any, Any]:
        first = self.pick_worker()
        if len(self.workers) == 1:
            return first, first
        second = self.pick_worker()
        return first, second


app = FastAPI(title="Ray dual-worker faster-qwen3-tts")
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


async def _ray_get(ref: Any) -> Any:
    return await asyncio.to_thread(ray.get, ref)


@app.get("/health")
async def health() -> dict[str, Any]:
    if state is None:
        raise HTTPException(status_code=503, detail="Server not initialized")
    rows = await asyncio.gather(*[_ray_get(worker.health.remote()) for worker in state.workers])
    return {"status": "ok", "workers": rows}


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
    rows = await asyncio.gather(*[_ray_get(worker.health.remote()) for worker in state.workers])
    return {
        "success": True,
        "status": "ready",
        "tts_enabled": True,
        "tts_model_loaded": True,
        "tts_backend": "ray_dual_worker",
        "default_speaker": rows[0].get("default_speaker") if rows else None,
        "workers_ready": len(rows),
        "workers": rows,
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
    if state is None:
        raise HTTPException(status_code=503, detail="Server not initialized")
    fmt = req.response_format.lower()
    if fmt != "wav":
        raise HTTPException(status_code=400, detail="Only response_format='wav' is supported")

    worker = state.pick_worker()
    normalized = normalize_for_tts(req.input, lang_hint=req.language)
    content = normalized.text
    if not has_readable_text(content):
        raise HTTPException(status_code=400, detail="No readable TTS text after sanitization")
    try:
        result = await _ray_get(
            worker.synthesize.remote(
                content,
                req.voice,
                req.language,
                req.instruction,
                req.max_new_tokens,
            )
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc

    result["normalizer"] = normalized.normalizer
    return Response(content=result["bytes"], media_type="audio/wav", headers=_tts_response_headers(result))


@app.post("/api/tts/speak_json")
async def speak_json(req: JsonSpeechRequest) -> JSONResponse:
    if state is None:
        raise HTTPException(status_code=503, detail="Server not initialized")
    worker = state.pick_worker()
    normalized = normalize_for_tts(req.input, lang_hint=req.language)
    content = normalized.text
    if not has_readable_text(content):
        raise HTTPException(status_code=400, detail="No readable TTS text after sanitization")
    try:
        result = await _ray_get(
            worker.synthesize.remote(
                content,
                req.voice,
                req.language,
                req.instruction,
                req.max_new_tokens,
            )
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc

    result["normalizer"] = normalized.normalizer
    wav_bytes = result.pop("bytes")
    if req.include_audio_b64:
        result["audio_b64"] = base64.b64encode(wav_bytes).decode("ascii")
    result["wav_bytes"] = len(wav_bytes)
    return JSONResponse(result)


@app.post("/api/tts/speak")
async def capswriter_speak(req: CapsWriterSpeakRequest) -> Response:
    if state is None:
        raise HTTPException(status_code=503, detail="Server not initialized")
    worker = state.pick_worker()
    normalized = normalize_for_tts(req.text, lang_hint=req.language)
    content = normalized.text
    if not has_readable_text(content):
        raise HTTPException(status_code=400, detail="No readable TTS text after sanitization")
    try:
        result = await _ray_get(
            worker.synthesize.remote(
                content,
                req.speaker or DEFAULT_SPEAKER,
                req.language,
                req.instruction,
                req.max_new_tokens,
            )
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc

    result["normalizer"] = normalized.normalizer
    return Response(content=result["bytes"], media_type="audio/wav", headers=_tts_response_headers(result))


async def _run_playback_first(req: PlaybackFirstRequest) -> dict[str, Any]:
    if state is None:
        raise HTTPException(status_code=503, detail="Server not initialized")

    normalized = normalize_for_tts(req.input, lang_hint=req.language)
    content = normalized.text
    if not has_readable_text(content):
        raise HTTPException(status_code=400, detail="No readable TTS text after sanitization")
    segments = _split_playback_first(
        content,
        target_audio_s=req.head_target_audio_s,
        min_chars=req.head_min_chars,
        max_chars=req.head_max_chars,
    )
    hard_cap = 512
    for segment in segments:
        segment["estimated_audio_s"] = _estimate_audio_s(segment["text"])
        segment["max_new_tokens"] = _estimate_max_new_tokens(segment["text"], hard_cap)

    start = time.perf_counter()
    if len(segments) == 1:
        worker = state.pick_worker()
        result = await _ray_get(
            worker.synthesize.remote(
                segments[0]["text"],
                req.voice,
                req.language,
                req.instruction,
                segments[0]["max_new_tokens"],
            )
        )
        result["segment_index"] = 0
        result["role"] = "full"
        results = [result]
        head_ready_wall_s = result["elapsed_s"]
        tail_ready_wall_s = None
    else:
        first_worker, second_worker = state.pick_two_workers()
        first_ref = first_worker.synthesize.remote(
            segments[0]["text"],
            req.voice,
            req.language,
            req.instruction,
            segments[0]["max_new_tokens"],
        )
        second_ref = second_worker.synthesize.remote(
            segments[1]["text"],
            req.voice,
            req.language,
            req.instruction,
            segments[1]["max_new_tokens"],
        )
        first_task = asyncio.create_task(_ray_get(first_ref))
        second_task = asyncio.create_task(_ray_get(second_ref))
        first_result = await first_task
        head_ready_wall_s = time.perf_counter() - start
        second_result = await second_task
        tail_ready_wall_s = time.perf_counter() - start
        first_result["segment_index"] = 0
        first_result["role"] = "head"
        second_result["segment_index"] = 1
        second_result["role"] = "tail"
        results = [first_result, second_result]

    sample_rate = int(results[0]["sample_rate"])
    join_silence = _silence_pcm(sample_rate, req.join_silence_ms) if len(results) > 1 else b""
    pcm_parts: list[bytes] = []
    for idx, result in enumerate(results):
        if idx:
            pcm_parts.append(join_silence)
        pcm_parts.append(_wav_payload(result["bytes"]))
    pcm = b"".join(pcm_parts)
    wav_bytes = _wav_header(sample_rate, len(pcm)) + pcm

    wall_s = time.perf_counter() - start
    audio_s = sum(float(item["audio_s"]) for item in results)
    if len(results) > 1:
        audio_s += req.join_silence_ms / 1000.0
    metadata = {
        "mode": "playback_first_two_segment",
        "segments_requested": len(segments),
        "head_ready_wall_s": head_ready_wall_s,
        "tail_ready_wall_s": tail_ready_wall_s,
        "wall_s": wall_s,
        "audio_s": audio_s,
        "rtf": audio_s / wall_s if wall_s > 0 else 0.0,
        "sample_rate": sample_rate,
        "join_silence_ms": req.join_silence_ms,
        "tail_ready_before_head_played_out": (
            True
            if len(results) == 1
            else bool((tail_ready_wall_s or wall_s) <= head_ready_wall_s + results[0]["audio_s"])
        ),
        "segments": [
            {
                "index": segment["index"],
                "role": result["role"],
                "text_len": len(segment["text"]),
                "estimated_audio_s": segment["estimated_audio_s"],
                "max_new_tokens": segment["max_new_tokens"],
                "worker_id": result["worker_id"],
                "gpu_ids": result["gpu_ids"],
                "audio_s": result["audio_s"],
                "elapsed_s": result["elapsed_s"],
                "rtf": result["rtf"],
                "ttfa_s": result["ttfa_s"],
            }
            for segment, result in zip(segments, results)
        ],
        "bytes": wav_bytes,
    }
    return metadata


@app.post("/api/tts/playback_first_json")
async def playback_first_json(req: PlaybackFirstRequest) -> JSONResponse:
    try:
        result = await _run_playback_first(req)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc
    wav_bytes = result.pop("bytes")
    if req.include_audio_b64:
        result["audio_b64"] = base64.b64encode(wav_bytes).decode("ascii")
    result["wav_bytes"] = len(wav_bytes)
    return JSONResponse(result)


@app.post("/api/tts/playback_first_wav")
async def playback_first_wav(req: PlaybackFirstRequest) -> StreamingResponse:
    async def stream():
        if state is None:
            raise HTTPException(status_code=503, detail="Server not initialized")

        normalized = normalize_for_tts(req.input, lang_hint=req.language)
        content = normalized.text
        if not has_readable_text(content):
            raise HTTPException(status_code=400, detail="No readable TTS text after sanitization")
        segments = _split_playback_first(
            content,
            target_audio_s=req.head_target_audio_s,
            min_chars=req.head_min_chars,
            max_chars=req.head_max_chars,
        )
        hard_cap = 512
        for segment in segments:
            segment["max_new_tokens"] = _estimate_max_new_tokens(segment["text"], hard_cap)

        sample_rate = 24000
        yield _wav_stream_header(sample_rate)

        if len(segments) == 1:
            worker = state.pick_worker()
            result = await _ray_get(
                worker.synthesize.remote(
                    segments[0]["text"],
                    req.voice,
                    req.language,
                    req.instruction,
                    segments[0]["max_new_tokens"],
                )
            )
            yield _wav_payload(result["bytes"])
            return

        first_worker, second_worker = state.pick_two_workers()
        first_task = asyncio.create_task(
            _ray_get(
                first_worker.synthesize.remote(
                    segments[0]["text"],
                    req.voice,
                    req.language,
                    req.instruction,
                    segments[0]["max_new_tokens"],
                )
            )
        )
        second_task = asyncio.create_task(
            _ray_get(
                second_worker.synthesize.remote(
                    segments[1]["text"],
                    req.voice,
                    req.language,
                    req.instruction,
                    segments[1]["max_new_tokens"],
                )
            )
        )
        first_result = await first_task
        sample_rate = int(first_result["sample_rate"])
        yield _wav_payload(first_result["bytes"])
        if req.join_silence_ms > 0:
            yield _silence_pcm(sample_rate, req.join_silence_ms)
        second_result = await second_task
        yield _wav_payload(second_result["bytes"])

    return StreamingResponse(stream(), media_type="audio/wav")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.getenv("QWEN_TTS_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("QWEN_TTS_PORT", "8091")))
    parser.add_argument("--model", default=os.getenv("QWEN_TTS_MODEL", "/home/ivan/models/Qwen3-TTS-12Hz-1.7B-CustomVoice"))
    parser.add_argument("--workers", type=int, default=int(os.getenv("QWEN_TTS_WORKERS", "2")))
    parser.add_argument("--attn", default=os.getenv("QWEN_TTS_ATTN", "sdpa"), choices=["sdpa", "eager", "flash_attention_2"])
    parser.add_argument("--dtype", default=os.getenv("QWEN_TTS_DTYPE", "bfloat16"), choices=["bfloat16", "float16", "float32"])
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
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")

    ray.init(ignore_reinit_error=True)
    workers = [
        TTSWorker.remote(
            worker_id=i,
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
        )
        for i in range(args.workers)
    ]
    ray.get([worker.health.remote() for worker in workers])
    state = AppState(workers=workers)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
