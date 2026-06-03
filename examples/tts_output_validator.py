"""Asynchronous TTS output validation using audio checks and AGX ASR."""

from __future__ import annotations

import difflib
import io
import json
import math
import os
import queue
import re
import threading
import time
import uuid
import wave
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib import error, request

import numpy as np

try:
    from examples.tts_pronunciation_advisor import enqueue_from_validation
except ImportError:  # pragma: no cover - direct script execution fallback
    from tts_pronunciation_advisor import enqueue_from_validation


DEFAULT_ASR_BASE_URL = "http://agx.taild500c8.ts.net:8001"


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def validation_enabled() -> bool:
    return _env_bool("QWEN_TTS_VALIDATION_ENABLED", True)


def _asr_base_url() -> str:
    return os.getenv("QWEN_TTS_VALIDATION_ASR_BASE_URL", DEFAULT_ASR_BASE_URL).strip().rstrip("/")


def _timeout_s() -> float:
    return float(os.getenv("QWEN_TTS_VALIDATION_ASR_TIMEOUT_S", "60"))


def _max_records() -> int:
    return max(20, int(os.getenv("QWEN_TTS_VALIDATION_MAX_RECORDS", "200")))


def _max_queue_size() -> int:
    return max(1, int(os.getenv("QWEN_TTS_VALIDATION_QUEUE_SIZE", "100")))


def _max_workers() -> int:
    return max(1, int(os.getenv("QWEN_TTS_VALIDATION_MAX_WORKERS", "1")))


def _similarity_warning() -> float:
    return float(os.getenv("QWEN_TTS_VALIDATION_SIMILARITY_WARNING", "0.58"))


def _similarity_failure() -> float:
    return float(os.getenv("QWEN_TTS_VALIDATION_SIMILARITY_FAILURE", "0.35"))


def _sample_rate() -> float:
    return max(0.0, min(1.0, float(os.getenv("QWEN_TTS_VALIDATION_SAMPLE_RATE", "1.0"))))


def _config_dir() -> Path:
    return Path(os.getenv("QWEN_TTS_CONFIG_DIR", "~/.config/faster-qwen3-tts")).expanduser()


def _candidate_log_path() -> Path:
    return Path(
        os.getenv("QWEN_TTS_PRONUNCIATION_CANDIDATES", str(_config_dir() / "pronunciation_candidates.jsonl"))
    ).expanduser()


def normalize_compare_text(text: str) -> str:
    value = (text or "").lower()
    value = re.sub(r"[\s\W_]+", "", value, flags=re.UNICODE)
    return value


def text_similarity(expected: str, actual: str) -> float:
    left = normalize_compare_text(expected)
    right = normalize_compare_text(actual)
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return difflib.SequenceMatcher(a=left, b=right).ratio()


def _tail_similarity(expected: str, actual: str) -> float:
    left = normalize_compare_text(expected)
    right = normalize_compare_text(actual)
    if not left:
        return 1.0
    tail_len = max(6, min(40, len(left) // 3))
    return text_similarity(left[-tail_len:], right[-tail_len:])


def _lowercase_candidate_tokens(text: str) -> list[str]:
    tokens = re.findall(r"(?<![A-Za-z0-9_])([a-z]{2,8})(?![A-Za-z0-9_])", text or "")
    return sorted(set(token.lower() for token in tokens))


def _record_pronunciation_candidates(job: ValidationJob, record: dict[str, Any]) -> None:
    issues = set(record.get("issues") or [])
    if not ({"text_mismatch", "possible_truncation"} & issues):
        return
    expected_tokens = _lowercase_candidate_tokens(job.expected_text)
    if not expected_tokens:
        return
    asr_tokens = set(_lowercase_candidate_tokens(str(record.get("asr_text") or "")))
    rows = []
    for token in expected_tokens:
        rows.append(
            {
                "ts": int(time.time()),
                "validation_id": job.validation_id,
                "trace_id": job.trace_id,
                "endpoint": job.endpoint,
                "token": token,
                "present_in_asr": token in asr_tokens,
                "issues": sorted(issues),
                "similarity": record.get("similarity"),
                "expected_text": job.expected_text[:240],
                "asr_text": str(record.get("asr_text") or "")[:240],
            }
        )
    try:
        path = _candidate_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:
        return


def analyze_wav_bytes(wav_bytes: bytes) -> dict[str, Any]:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
        sample_rate = int(wav_file.getframerate())
        channels = int(wav_file.getnchannels())
        sample_width = int(wav_file.getsampwidth())
        frames = int(wav_file.getnframes())
        raw = wav_file.readframes(frames)

    if sample_width != 2:
        raise ValueError(f"Only 16-bit PCM WAV is supported, got sample_width={sample_width}")

    pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32)
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1)
    if pcm.size == 0 or sample_rate <= 0:
        return {
            "sample_rate_hz": sample_rate,
            "duration_s": 0.0,
            "rms": 0.0,
            "peak": 0.0,
            "voice_ratio": 0.0,
            "max_silence_s": 0.0,
        }

    audio = pcm / 32768.0
    abs_audio = np.abs(audio)
    rms = float(math.sqrt(float(np.mean(audio * audio))))
    peak = float(np.max(abs_audio))
    duration_s = float(len(audio) / sample_rate)

    frame_size = max(1, int(sample_rate * 0.05))
    frames_abs = [
        float(np.sqrt(float(np.mean(audio[idx : idx + frame_size] ** 2))))
        for idx in range(0, len(audio), frame_size)
    ]
    threshold = max(0.004, rms * 0.28)
    voiced = [value > threshold for value in frames_abs]
    voice_ratio = sum(1 for item in voiced if item) / max(1, len(voiced))

    max_silent_frames = 0
    current = 0
    for item in voiced:
        if item:
            max_silent_frames = max(max_silent_frames, current)
            current = 0
        else:
            current += 1
    max_silent_frames = max(max_silent_frames, current)
    max_silence_s = max_silent_frames * 0.05

    return {
        "sample_rate_hz": sample_rate,
        "duration_s": duration_s,
        "rms": rms,
        "peak": peak,
        "voice_ratio": voice_ratio,
        "max_silence_s": max_silence_s,
    }


def _audio_issues(audio: dict[str, Any], expected_audio_s: float | None = None) -> list[str]:
    issues: list[str] = []
    duration_s = float(audio.get("duration_s") or 0.0)
    rms = float(audio.get("rms") or 0.0)
    peak = float(audio.get("peak") or 0.0)
    voice_ratio = float(audio.get("voice_ratio") or 0.0)
    max_silence_s = float(audio.get("max_silence_s") or 0.0)

    if duration_s <= 0.05 or peak < 0.002 or rms < 0.001:
        issues.append("empty_audio")
    if expected_audio_s is not None and expected_audio_s >= 6.0 and duration_s < max(2.5, expected_audio_s * 0.55):
        issues.append("suspicious_short_duration")
    if duration_s >= 2.5 and voice_ratio < 0.12:
        issues.append("low_voice_ratio")
    if max_silence_s >= max(2.5, duration_s * 0.45):
        issues.append("long_silence")
    if expected_audio_s is not None and duration_s > max(12.0, expected_audio_s * 2.8):
        issues.append("suspicious_duration")
    return issues


def _multipart_body(field_name: str, filename: str, data: bytes, content_type: str) -> tuple[bytes, str]:
    boundary = f"----qwen-tts-validation-{uuid.uuid4().hex}"
    chunks = [
        f"--{boundary}\r\n".encode("utf-8"),
        (
            f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n'
            f"Content-Type: {content_type}\r\n\r\n"
        ).encode("utf-8"),
        data,
        b"\r\n",
        f"--{boundary}--\r\n".encode("utf-8"),
    ]
    return b"".join(chunks), boundary


def call_asr_transcribe(wav_bytes: bytes, filename: str = "tts-validation.wav") -> dict[str, Any]:
    body, boundary = _multipart_body("audio", filename, wav_bytes, "audio/wav")
    req = request.Request(
        f"{_asr_base_url()}/api/asr/transcribe",
        data=body,
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with request.urlopen(req, timeout=_timeout_s()) as response:
        payload = response.read()
    decoded = json.loads(payload.decode("utf-8"))
    if not isinstance(decoded, dict):
        raise RuntimeError("ASR returned non-object JSON")
    return decoded


def _verdict(issues: list[str], similarity: float | None) -> str:
    if "empty_audio" in issues or "text_mismatch" in issues:
        return "failed"
    if similarity is not None and similarity < _similarity_failure():
        return "failed"
    if issues:
        return "warning"
    return "passed"


@dataclass
class ValidationJob:
    validation_id: str
    expected_text: str
    wav_bytes: bytes
    trace_id: str = ""
    endpoint: str = ""
    speaker: str = ""
    language: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


class ValidationStore:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._records: dict[str, dict[str, Any]] = {}
        self._order: deque[str] = deque()

    def upsert(self, validation_id: str, record: dict[str, Any]) -> None:
        with self._lock:
            existing = self._records.get(validation_id, {})
            merged = {**existing, **record}
            self._records[validation_id] = merged
            if validation_id not in self._order:
                self._order.appendleft(validation_id)
            while len(self._order) > _max_records():
                old_id = self._order.pop()
                self._records.pop(old_id, None)

    def get(self, validation_id: str) -> dict[str, Any] | None:
        with self._lock:
            item = self._records.get(validation_id)
            return dict(item) if item else None

    def recent(self, limit: int) -> list[dict[str, Any]]:
        with self._lock:
            ids = list(self._order)[: max(1, min(200, int(limit)))]
            return [dict(self._records[item]) for item in ids if item in self._records]


_store = ValidationStore()
_queue: queue.Queue[ValidationJob] | None = None
_started = False
_started_lock = threading.Lock()


def _base_record(job: ValidationJob, status: str) -> dict[str, Any]:
    return {
        "success": True,
        "validation_id": job.validation_id,
        "status": status,
        "verdict": "pending" if status in {"queued", "running"} else None,
        "trace_id": job.trace_id,
        "endpoint": job.endpoint,
        "speaker": job.speaker,
        "language": job.language,
        "expected_text": job.expected_text,
        "metadata": job.metadata,
        "created_at": job.created_at,
        "updated_at": time.time(),
    }


def _validate_job(job: ValidationJob) -> dict[str, Any]:
    started = time.time()
    audio = analyze_wav_bytes(job.wav_bytes)
    expected_audio_s = job.metadata.get("estimated_audio_s") if isinstance(job.metadata, dict) else None
    issues = _audio_issues(audio, float(expected_audio_s) if expected_audio_s is not None else None)

    asr_payload: dict[str, Any] | None = None
    asr_text = ""
    similarity: float | None = None
    try:
        asr_payload = call_asr_transcribe(job.wav_bytes, filename=f"{job.validation_id}.wav")
        if asr_payload.get("success") is False:
            issues.append("asr_failed")
        asr_text = str(asr_payload.get("text") or asr_payload.get("raw_asr_text") or "").strip()
    except (error.URLError, TimeoutError, OSError, RuntimeError, ValueError) as exc:
        return {
            **_base_record(job, "done"),
            "verdict": "skipped",
            "passed": False,
            "issues": ["asr_unavailable"],
            "error": f"{type(exc).__name__}: {exc}",
            "audio": audio,
            "asr_text": "",
            "duration_s": time.time() - started,
            "updated_at": time.time(),
        }

    if job.expected_text.strip():
        similarity = text_similarity(job.expected_text, asr_text)
        tail_similarity = _tail_similarity(job.expected_text, asr_text)
        if similarity < _similarity_warning():
            issues.append("text_mismatch")
        elif tail_similarity < 0.45:
            issues.append("possible_truncation")

    verdict = _verdict(issues, similarity)
    return {
        **_base_record(job, "done"),
        "verdict": verdict,
        "passed": verdict == "passed",
        "issues": sorted(set(issues)),
        "audio": audio,
        "asr_text": asr_text,
        "asr_payload": asr_payload,
        "similarity": similarity,
        "duration_s": time.time() - started,
        "updated_at": time.time(),
    }


def _worker_loop() -> None:
    assert _queue is not None
    while True:
        job = _queue.get()
        _store.upsert(job.validation_id, {**_base_record(job, "running"), "started_at": time.time()})
        try:
            record = _validate_job(job)
        except Exception as exc:
            record = {
                **_base_record(job, "done"),
                "verdict": "failed",
                "passed": False,
                "issues": ["validator_error"],
                "error": f"{type(exc).__name__}: {exc}",
                "updated_at": time.time(),
            }
        _record_pronunciation_candidates(job, record)
        try:
            enqueue_from_validation(record)
        except Exception:
            pass
        _store.upsert(job.validation_id, record)
        print(
            "[TTS-VALIDATION] "
            f"id={job.validation_id} trace_id={job.trace_id or '-'} "
            f"endpoint={job.endpoint or '-'} verdict={record.get('verdict')} "
            f"issues={','.join(record.get('issues') or []) or '-'}",
            flush=True,
        )
        _queue.task_done()


def _ensure_started() -> None:
    global _queue, _started
    if _started:
        return
    with _started_lock:
        if _started:
            return
        _queue = queue.Queue(maxsize=_max_queue_size())
        for idx in range(_max_workers()):
            thread = threading.Thread(target=_worker_loop, name=f"tts-output-validator-{idx + 1}", daemon=True)
            thread.start()
        _started = True


def enqueue_validation(
    *,
    expected_text: str,
    wav_bytes: bytes,
    trace_id: str = "",
    endpoint: str = "",
    speaker: str = "",
    language: str = "",
    metadata: dict[str, Any] | None = None,
) -> str | None:
    if not validation_enabled():
        return None
    if _sample_rate() < 1.0:
        threshold = int(_sample_rate() * 10000)
        if uuid.uuid4().int % 10000 >= threshold:
            return None
    _ensure_started()
    assert _queue is not None
    validation_id = f"val-{uuid.uuid4().hex[:16]}"
    job = ValidationJob(
        validation_id=validation_id,
        expected_text=expected_text,
        wav_bytes=wav_bytes,
        trace_id=trace_id,
        endpoint=endpoint,
        speaker=speaker,
        language=language,
        metadata=metadata or {},
    )
    try:
        _queue.put_nowait(job)
    except queue.Full:
        _store.upsert(
            validation_id,
            {
                **_base_record(job, "done"),
                "verdict": "skipped",
                "passed": False,
                "issues": ["queue_full"],
                "updated_at": time.time(),
            },
        )
        return validation_id
    _store.upsert(validation_id, _base_record(job, "queued"))
    return validation_id


def get_validation_result(validation_id: str) -> dict[str, Any] | None:
    return _store.get(validation_id)


def recent_validation_results(limit: int = 20) -> list[dict[str, Any]]:
    return _store.recent(limit)


def validation_headers(validation_id: str | None) -> dict[str, str]:
    if not validation_id:
        return {}
    return {
        "X-TTS-Validation-Id": validation_id,
        "X-TTS-Validation-Status": "queued",
    }
