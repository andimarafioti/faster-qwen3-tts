#!/usr/bin/env python3
"""Small read-aloud web server for Faster Qwen3-TTS.

Usage:
    python examples/read_aloud_server.py --backend ggml --port 7861
    python examples/read_aloud_server.py --backend torch --model Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import io
import os
import json
import re
import shutil
import sys
import sqlite3
import threading
import time
import uuid
import wave
from pathlib import Path
from typing import Iterable
import urllib.request

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from faster_qwen3_tts import FasterQwen3TTS  # noqa: E402


BASE_DIR = Path(__file__).resolve().parent
INDEX_HTML = BASE_DIR / "read_aloud.html"
OUTPUT_DIR = Path(os.environ.get("READ_ALOUD_HISTORY_DIR", BASE_DIR.parent / "tts_history"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_LIMIT = max(1, int(os.environ.get("READ_ALOUD_HISTORY_LIMIT", "100")))
LEGACY_OUTPUT_DIR = Path(os.environ.get("READ_ALOUD_LEGACY_HISTORY_DIR", "/tmp/faster-qwen3-tts-read-aloud"))
BATCH_DB_PATH = Path(os.environ.get("READ_ALOUD_BATCH_DB", OUTPUT_DIR / "batch.sqlite3"))
NOTES_API_BASE = os.environ.get("NOTES_API_BASE", "http://localhost:9999").rstrip("/")

MODEL_LOCK = threading.Lock()
HISTORY_LOCK = threading.Lock()
INFERENCE_LOCK = threading.Lock()
INTERACTIVE_CONDITION = threading.Condition()
INTERACTIVE_WAITING = 0
INTERACTIVE_ACTIVE = 0
BATCH_STOP = threading.Event()
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


class BatchRequest(BaseModel):
    items: list[SpeakRequest] = Field(min_length=1, max_length=100)


def _post_note(text: str) -> None:
    def send() -> None:
        try:
            body = json.dumps({"channel": "app", "text": text[:500]}).encode()
            request = urllib.request.Request(
                f"{NOTES_API_BASE}/note", data=body,
                headers={"Content-Type": "application/json"}, method="POST",
            )
            urllib.request.urlopen(request, timeout=2).close()
        except Exception:
            pass
    threading.Thread(target=send, daemon=True).start()


@contextmanager
def _interactive_slot():
    global INTERACTIVE_WAITING, INTERACTIVE_ACTIVE
    with INTERACTIVE_CONDITION:
        INTERACTIVE_WAITING += 1
        INTERACTIVE_CONDITION.notify_all()
    INFERENCE_LOCK.acquire()
    with INTERACTIVE_CONDITION:
        INTERACTIVE_WAITING -= 1
        INTERACTIVE_ACTIVE += 1
    try:
        yield
    finally:
        with INTERACTIVE_CONDITION:
            INTERACTIVE_ACTIVE -= 1
            INTERACTIVE_CONDITION.notify_all()
        INFERENCE_LOCK.release()


def _wait_for_batch_slot() -> bool:
    while not BATCH_STOP.is_set():
        with INTERACTIVE_CONDITION:
            if INTERACTIVE_WAITING or INTERACTIVE_ACTIVE:
                INTERACTIVE_CONDITION.wait(timeout=1)
                continue
        if INFERENCE_LOCK.acquire(timeout=1):
            with INTERACTIVE_CONDITION:
                if INTERACTIVE_WAITING or INTERACTIVE_ACTIVE:
                    INFERENCE_LOCK.release()
                    continue
            return True
    return False


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



def _history_metadata_path(audio_id: str) -> Path:
    return OUTPUT_DIR / f"{audio_id}.json"


def _save_history_metadata(audio_id: str, req: SpeakRequest, result: dict, streamed: bool) -> dict:
    entry = {
        **result, "text": req.text, "instruction": req.instruction,
        "dialogue_mode": req.dialogue_mode, "streamed": streamed,
        "timestamp": int(time.time() * 1000),
    }
    with HISTORY_LOCK:
        _history_metadata_path(audio_id).write_text(
            json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        files = sorted(OUTPUT_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        for stale in files[HISTORY_LIMIT:]:
            stale.with_suffix(".wav").unlink(missing_ok=True)
            stale.unlink(missing_ok=True)
    return entry


def _read_history() -> list[dict]:
    entries = []
    with HISTORY_LOCK:
        paths = sorted(OUTPUT_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        for path in paths[:HISTORY_LIMIT]:
            try:
                entry = json.loads(path.read_text(encoding="utf-8"))
                if (OUTPUT_DIR / f"{entry['id']}.wav").is_file():
                    entries.append(entry)
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                continue
    return entries


def _clear_history() -> None:
    with HISTORY_LOCK:
        for pattern in ("*.json", "*.wav"):
            for path in OUTPUT_DIR.glob(pattern):
                path.unlink(missing_ok=True)


def _migrate_legacy_history() -> int:
    if not LEGACY_OUTPUT_DIR.is_dir() or LEGACY_OUTPUT_DIR == OUTPUT_DIR:
        return 0
    migrated = 0
    with HISTORY_LOCK:
        for source in LEGACY_OUTPUT_DIR.glob("*.wav"):
            target = OUTPUT_DIR / source.name
            metadata_path = target.with_suffix(".json")
            if target.exists() or metadata_path.exists():
                continue
            try:
                info = sf.info(source)
                shutil.copy2(source, target)
                audio_id = source.stem
                timestamp = int(source.stat().st_mtime * 1000)
                entry = {
                    "id": audio_id,
                    "url": f"/audio/{audio_id}.wav",
                    "sample_rate": int(info.samplerate),
                    "duration_s": float(info.duration),
                    "elapsed_s": 0.0,
                    "rtf": 0.0,
                    "text": "Prior generation",
                    "instruction": "",
                    "dialogue_mode": False,
                    "streamed": False,
                    "timestamp": timestamp,
                    "legacy": True,
                }
                metadata_path.write_text(json.dumps(entry, indent=2), encoding="utf-8")
                migrated += 1
            except (OSError, RuntimeError):
                target.unlink(missing_ok=True)
    return migrated


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
    result = {
        "id": audio_id,
        "url": f"/audio/{audio_id}.wav",
        "sample_rate": int(sample_rate or 24000),
        "duration_s": duration,
        "elapsed_s": elapsed,
        "rtf": duration / elapsed if elapsed > 0 else 0.0,
        "segments": len(segments),
    }
    _save_history_metadata(audio_id, req, result, streamed=False)
    return result


def _stream_chunks_unlocked(req: SpeakRequest, audio_id: str | None = None) -> Iterable[bytes]:
    if req.seed is not None:
        torch.manual_seed(req.seed)

    model = _load_model()
    audio_id = audio_id or uuid.uuid4().hex
    started = time.perf_counter()
    wav_writer = None
    sample_count = 0
    sample_rate = 24000
    first = True
    try:
        for audio_chunk, sample_rate, _timing in model.generate_voice_design_streaming(
            text=req.text, instruct=req.instruction, language=req.language,
            max_new_tokens=req.max_new_tokens, temperature=req.temperature,
            top_k=req.top_k, top_p=req.top_p, do_sample=not req.greedy,
            repetition_penalty=req.repetition_penalty, chunk_size=req.chunk_size,
        ):
            if first:
                wav_writer = wave.open(str(OUTPUT_DIR / f"{audio_id}.wav"), "wb")
                wav_writer.setnchannels(1)
                wav_writer.setsampwidth(2)
                wav_writer.setframerate(int(sample_rate))
                yield _wav_header(int(sample_rate))
                first = False
            pcm = _pcm16(audio_chunk)
            if pcm:
                wav_writer.writeframes(pcm)
                sample_count += len(pcm) // 2
                yield pcm
        if first:
            yield _wav_header(24000)
    finally:
        if wav_writer is not None:
            wav_writer.close()
        if sample_count:
            elapsed = time.perf_counter() - started
            duration = sample_count / float(sample_rate)
            result = {
                "id": audio_id, "url": f"/audio/{audio_id}.wav",
                "sample_rate": int(sample_rate), "duration_s": duration,
                "elapsed_s": elapsed, "rtf": duration / elapsed if elapsed > 0 else 0.0,
            }
            _save_history_metadata(audio_id, req, result, streamed=True)


def _stream_chunks(req: SpeakRequest, audio_id: str | None = None) -> Iterable[bytes]:
    with _interactive_slot():
        yield from _stream_chunks_unlocked(req, audio_id)


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
    result = {
        "id": audio_id,
        "url": f"/audio/{audio_id}.wav",
        "sample_rate": int(sample_rate),
        "duration_s": duration,
        "elapsed_s": elapsed,
        "rtf": duration / elapsed if elapsed > 0 else 0.0,
    }
    _save_history_metadata(audio_id, req, result, streamed=False)
    return result


def _batch_connect() -> sqlite3.Connection:
    connection = sqlite3.connect(BATCH_DB_PATH, timeout=30)
    connection.row_factory = sqlite3.Row
    return connection


def _init_batch_db() -> None:
    BATCH_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _batch_connect() as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("""CREATE TABLE IF NOT EXISTS batch_jobs (
            id TEXT PRIMARY KEY, batch_id TEXT NOT NULL, position INTEGER NOT NULL,
            status TEXT NOT NULL, request_json TEXT NOT NULL, result_json TEXT,
            error TEXT, created_at INTEGER NOT NULL, started_at INTEGER, finished_at INTEGER
        )""")
        connection.execute("UPDATE batch_jobs SET status='queued', started_at=NULL WHERE status='running'")


def _enqueue_batch(items: list[SpeakRequest]) -> dict:
    batch_id = uuid.uuid4().hex
    now = int(time.time() * 1000)
    rows = [(uuid.uuid4().hex, batch_id, index, "queued", item.json(), now)
            for index, item in enumerate(items, start=1)]
    with _batch_connect() as connection:
        connection.executemany(
            "INSERT INTO batch_jobs (id,batch_id,position,status,request_json,created_at) VALUES (?,?,?,?,?,?)",
            rows,
        )
    return {"batch_id": batch_id, "queued": len(rows)}


def _batch_rows() -> list[dict]:
    with _batch_connect() as connection:
        rows = connection.execute(
            "SELECT * FROM batch_jobs ORDER BY created_at DESC, position ASC LIMIT 500"
        ).fetchall()
    items = []
    for row in rows:
        item = dict(row)
        item["request"] = json.loads(item.pop("request_json"))
        raw_result = item.pop("result_json")
        item["result"] = json.loads(raw_result) if raw_result else None
        items.append(item)
    return items


def _cancel_batch_job(job_id: str) -> bool:
    with _batch_connect() as connection:
        cursor = connection.execute(
            "DELETE FROM batch_jobs WHERE id=? AND status='queued'", (job_id,)
        )
    return cursor.rowcount > 0


def _claim_batch_job() -> sqlite3.Row | None:
    with _batch_connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            "SELECT * FROM batch_jobs WHERE status='queued' ORDER BY created_at,position LIMIT 1"
        ).fetchone()
        if row is not None:
            connection.execute(
                "UPDATE batch_jobs SET status='running',started_at=? WHERE id=?",
                (int(time.time() * 1000), row["id"]),
            )
        return row


def _finish_batch_job(job_id: str, result: dict | None = None, error: str | None = None) -> None:
    with _batch_connect() as connection:
        connection.execute(
            "UPDATE batch_jobs SET status=?,result_json=?,error=?,finished_at=? WHERE id=?",
            ("failed" if error else "completed", json.dumps(result) if result else None,
             error, int(time.time() * 1000), job_id),
        )


def _batch_worker() -> None:
    while not BATCH_STOP.is_set():
        if not _wait_for_batch_slot():
            return
        job = None
        try:
            job = _claim_batch_job()
            if job is None:
                BATCH_STOP.wait(1)
                continue
            req = SpeakRequest.parse_raw(job["request_json"])
            _post_note(f"[read-aloud/batch] rendering -- batch {job['batch_id'][:8]} item {job['position']}")
            _finish_batch_job(job["id"], result=_render_file(req))
        except Exception as exc:
            if job is not None:
                _finish_batch_job(job["id"], error=str(exc))
        finally:
            INFERENCE_LOCK.release()


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
def speak(req: SpeakRequest):
    try:
        with _interactive_slot():
            return JSONResponse(_render_file(req))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/speak/stream")
def speak_stream(req: SpeakRequest):
    try:
        audio_id = uuid.uuid4().hex
        return StreamingResponse(
            _stream_chunks(req, audio_id), media_type="audio/wav", headers={"X-Audio-Id": audio_id}
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/speak/stream-url")
def speak_stream_url(req: SpeakRequest):
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
        return StreamingResponse(_stream_chunks(req, stream_id), media_type="audio/wav")
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


@app.get("/api/history")
def history():
    return {"items": _read_history()}


@app.delete("/api/history")
def clear_history():
    _clear_history()
    return {"status": "cleared"}


@app.post("/api/batch")
def enqueue_batch(batch: BatchRequest):
    result = _enqueue_batch(batch.items)
    _post_note(f"[read-aloud/batch] queued -- {result['queued']} quiet-time requests")
    return result


@app.get("/api/batch")
def batch_status():
    return {"items": _batch_rows()}


@app.delete("/api/batch/{job_id}")
def cancel_batch(job_id: str):
    if not _cancel_batch_job(job_id):
        raise HTTPException(status_code=409, detail="Only queued jobs can be removed")
    return {"status": "removed"}


@app.get("/api/request-template")
def request_template():
    return JSONResponse(json.loads(SpeakRequest(text="Hello.").json()))


def main() -> None:
    global ARGS
    ARGS = _parse_args()
    if not ARGS.no_preload:
        _post_note(f"[read-aloud/server] loading -- {ARGS.model} with {ARGS.backend}")
        print(f"Loading {ARGS.model} with {ARGS.backend} backend...")
        _load_model()
    migrated = _migrate_legacy_history()
    if migrated:
        _post_note(f"[read-aloud/library] migrated -- {migrated} prior WAV files added")
    _init_batch_db()
    threading.Thread(target=_batch_worker, name="batch-worker", daemon=True).start()
    _post_note(f"[read-aloud/server] ready -- interactive priority and quiet-time batch queue on :{ARGS.port}")
    print(f"Open http://{ARGS.host}:{ARGS.port}")
    uvicorn.run(app, host=ARGS.host, port=ARGS.port)


if __name__ == "__main__":
    main()
