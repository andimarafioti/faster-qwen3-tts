#!/usr/bin/env python3
"""
Faster Qwen3-TTS Demo Server

Usage:
    python demo/server.py
    python demo/server.py --model Qwen/Qwen3-TTS-12Hz-1.7B-Base --port 7860
    python demo/server.py --no-preload  # skip startup model load
"""

import argparse
import asyncio
import base64
from collections import OrderedDict
import hashlib
import io
import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

# Allow running from any directory
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from faster_qwen3_tts import FasterQwen3TTS, apply_continuation_state_delta
except ImportError:
    print("Error: faster_qwen3_tts not found.")
    print("Install with:  pip install -e .  (from the repo root)")
    sys.exit(1)

from nano_parakeet import from_pretrained as _parakeet_from_pretrained


_ALL_MODELS = [
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
]

_active_models_env = os.environ.get("ACTIVE_MODELS", "")
if _active_models_env:
    _allowed = {m.strip() for m in _active_models_env.split(",") if m.strip()}
    AVAILABLE_MODELS = [m for m in _ALL_MODELS if m in _allowed]
else:
    AVAILABLE_MODELS = list(_ALL_MODELS)

BASE_DIR = Path(__file__).resolve().parent
# Assets that need to be downloaded at runtime go to a writable directory.
# /app is read-only in HF Spaces; fall back to /tmp.
_ASSET_DIR = Path(os.environ.get("ASSET_DIR", "/tmp/faster-qwen3-tts-assets"))
PRESET_TRANSCRIPTS = _ASSET_DIR / "samples" / "parity" / "icl_transcripts.txt"
PRESET_REFS = [
    ("ref_audio_3", _ASSET_DIR / "ref_audio_3.wav", "Clone 1"),
    ("ref_audio_2", _ASSET_DIR / "ref_audio_2.wav", "Clone 2"),
    ("ref_audio", _ASSET_DIR / "ref_audio.wav", "Clone 3"),
]

_GITHUB_RAW = "https://raw.githubusercontent.com/andimarafioti/faster-qwen3-tts/main"
_PRESET_REMOTE = {
    "ref_audio":   f"{_GITHUB_RAW}/ref_audio.wav",
    "ref_audio_2": f"{_GITHUB_RAW}/ref_audio_2.wav",
    "ref_audio_3": f"{_GITHUB_RAW}/ref_audio_3.wav",
}
_TRANSCRIPT_REMOTE = f"{_GITHUB_RAW}/samples/parity/icl_transcripts.txt"


def _fetch_preset_assets() -> None:
    """Download preset wav files and transcripts from GitHub if not present locally."""
    import urllib.request
    _ASSET_DIR.mkdir(parents=True, exist_ok=True)
    PRESET_TRANSCRIPTS.parent.mkdir(parents=True, exist_ok=True)
    if not PRESET_TRANSCRIPTS.exists():
        try:
            urllib.request.urlretrieve(_TRANSCRIPT_REMOTE, PRESET_TRANSCRIPTS)
        except Exception as e:
            print(f"Warning: could not fetch transcripts: {e}")
    for key, path, _ in PRESET_REFS:
        if not path.exists() and key in _PRESET_REMOTE:
            try:
                urllib.request.urlretrieve(_PRESET_REMOTE[key], path)
                print(f"Downloaded {path.name}")
            except Exception as e:
                print(f"Warning: could not fetch {key}: {e}")

_preset_refs: dict[str, dict] = {}


def _load_preset_transcripts() -> dict[str, str]:
    if not PRESET_TRANSCRIPTS.exists():
        return {}
    transcripts = {}
    for line in PRESET_TRANSCRIPTS.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key_part, text = line.split(":", 1)
        key = key_part.split("(")[0].strip()
        transcripts[key] = text.strip()
    return transcripts


def _load_preset_refs() -> None:
    transcripts = _load_preset_transcripts()
    for key, path, label in PRESET_REFS:
        if not path.exists():
            continue
        content = path.read_bytes()
        cached_path = _get_cached_ref_path(content)
        _preset_refs[key] = {
            "id": key,
            "label": label,
            "filename": path.name,
            "path": cached_path,
            "ref_text": transcripts.get(key, ""),
            "audio_b64": base64.b64encode(content).decode(),
        }


def _prime_preset_voice_cache(model: FasterQwen3TTS) -> None:
    if not _preset_refs:
        return
    for preset in _preset_refs.values():
        ref_path = preset["path"]
        ref_text = preset["ref_text"]
        for xvec_only in (True, False):
            try:
                model._prepare_generation(
                    text="Hello.",
                    ref_audio=ref_path,
                    ref_text=ref_text,
                    language="English",
                    xvec_only=xvec_only,
                    non_streaming_mode=True,
                )
            except Exception:
                continue

app = FastAPI(title="Faster Qwen3-TTS Demo")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_model_cache: OrderedDict[str, FasterQwen3TTS] = OrderedDict()
_model_cache_max: int = int(os.environ.get("MODEL_CACHE_SIZE", "2"))
_active_model_name: str | None = None
_loading = False
_ref_cache: dict[str, str] = {}
_ref_cache_lock = threading.Lock()
_parakeet = None
_generation_lock = asyncio.Lock()
_generation_waiters: int = 0  # requests waiting for or holding the generation lock
_continuation_sessions: OrderedDict[str, dict] = OrderedDict()
_continuation_sessions_max: int = int(os.environ.get("CONTINUATION_SESSION_CACHE_SIZE", "4"))
_continuation_sessions_lock = threading.Lock()

# Guard against inputs that would overflow the static KV cache (max_seq_len=2048).
# At ~3-4 chars/token for English the overhead of system/ref tokens leaves room
# for roughly 1000 chars before we approach the limit.
MAX_TEXT_CHARS = 1000
# ~10 MB covers 1 minute of 44.1 kHz stereo 16-bit WAV.
MAX_AUDIO_BYTES = 10 * 1024 * 1024
CONTINUATION_COMPARE_CONTEXT_SECONDS = float(
    os.environ.get("CONTINUATION_COMPARE_CONTEXT_SECONDS", "2.0")
)
_AUDIO_TOO_LARGE_MSG = (
    "Audio file too large ({size_mb:.1f} MB). "
    "Voice cloning works best with short clips under 1 minute — please upload a shorter recording."
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _to_wav_b64(audio: np.ndarray, sr: int) -> str:
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    if audio.ndim > 1:
        audio = audio.squeeze()
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return b64


def _concat_audio(audio_list) -> np.ndarray:
    if isinstance(audio_list, np.ndarray):
        return audio_list.astype(np.float32).squeeze()
    parts = [np.array(a, dtype=np.float32).squeeze() for a in audio_list if len(a) > 0]
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


def _trim_audio_tail(audio: np.ndarray, sr: int, seconds: float = CONTINUATION_COMPARE_CONTEXT_SECONDS) -> np.ndarray:
    audio = np.array(audio, dtype=np.float32).squeeze()
    if sr <= 0 or seconds <= 0 or audio.size == 0:
        return np.zeros(0, dtype=np.float32)
    keep = max(1, int(round(sr * seconds)))
    if audio.size <= keep:
        return audio
    return audio[-keep:]


def _append_audio_tail(
    current_tail: np.ndarray | None,
    chunk: np.ndarray,
    sr: int,
    seconds: float = CONTINUATION_COMPARE_CONTEXT_SECONDS,
) -> np.ndarray:
    chunk = np.array(chunk, dtype=np.float32).squeeze()
    if current_tail is None or current_tail.size == 0:
        return _trim_audio_tail(chunk, sr, seconds)
    if chunk.size == 0:
        return _trim_audio_tail(current_tail, sr, seconds)
    combined = np.concatenate([current_tail, chunk])
    return _trim_audio_tail(combined, sr, seconds)


def _prepend_context_audio(context_audio: np.ndarray, context_sr: int, audio: np.ndarray, sr: int) -> np.ndarray:
    audio = np.array(audio, dtype=np.float32).squeeze()
    context_audio = np.array(context_audio, dtype=np.float32).squeeze()
    if context_audio.size == 0:
        return audio
    if context_sr != sr:
        context_audio = torchaudio.functional.resample(
            torch.from_numpy(context_audio).unsqueeze(0),
            context_sr,
            sr,
        ).squeeze(0).cpu().numpy()
    return np.concatenate([context_audio.astype(np.float32), audio])

def _get_cached_ref_path(content: bytes) -> str:
    digest = hashlib.sha1(content).hexdigest()
    with _ref_cache_lock:
        cached = _ref_cache.get(digest)
        if cached and os.path.exists(cached):
            return cached
        tmp_dir = Path(tempfile.gettempdir())
        path = tmp_dir / f"faster_qwen3_tts_ref_{digest}.wav"
        if not path.exists():
            path.write_bytes(content)
        _ref_cache[digest] = str(path)
        return str(path)


def _default_non_streaming_mode_for_mode(mode: str) -> bool:
    return mode != "voice_clone"


def _clear_continuation_sessions() -> None:
    with _continuation_sessions_lock:
        _continuation_sessions.clear()


def _make_continuation_session_id() -> str:
    token = f"{time.time_ns()}-{os.getpid()}-{threading.get_ident()}".encode()
    return hashlib.sha1(token).hexdigest()[:16]


def _build_continuation_template(
    *,
    mode: str,
    language: str,
    non_streaming_mode: bool,
    ref_audio_path: str | None = None,
    ref_text: str = "",
    xvec_only: bool = True,
    speaker: str = "",
    instruct: str = "",
) -> dict:
    return {
        "mode": mode,
        "language": language,
        "non_streaming_mode": bool(non_streaming_mode),
        "ref_audio_path": ref_audio_path,
        "ref_text": ref_text,
        "xvec_only": bool(xvec_only),
        "speaker": speaker,
        "instruct": instruct,
    }


def _store_continuation_session(
    *,
    model_name: str,
    template: dict,
    state: dict,
    audio_tail: np.ndarray,
    sample_rate: int,
) -> str:
    session_id = _make_continuation_session_id()
    with _continuation_sessions_lock:
        _continuation_sessions[session_id] = {
            "id": session_id,
            "model_name": model_name,
            "template": template,
            "state": state,
            "audio_tail": np.array(audio_tail, dtype=np.float32).copy(),
            "sample_rate": int(sample_rate),
            "created_at": time.time(),
        }
        _continuation_sessions.move_to_end(session_id)
        while len(_continuation_sessions) > _continuation_sessions_max:
            _continuation_sessions.popitem(last=False)
    return session_id


def _get_continuation_session(session_id: str) -> dict | None:
    with _continuation_sessions_lock:
        session = _continuation_sessions.get(session_id)
        if session is None:
            return None
        _continuation_sessions.move_to_end(session_id)
        return session


def _run_demo_generation(
    model: FasterQwen3TTS,
    *,
    template: dict,
    text: str,
    temperature: float,
    top_k: int,
    repetition_penalty: float,
    continuation_state: dict | None = None,
    return_continuation_state: bool | str = False,
    max_new_tokens: int = 360,
):
    common = {
        "text": text,
        "language": template["language"],
        "non_streaming_mode": template["non_streaming_mode"],
        "temperature": temperature,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
        "max_new_tokens": max_new_tokens,
    }
    if continuation_state is not None:
        common["continuation_state"] = continuation_state
    if return_continuation_state:
        common["return_continuation_state"] = return_continuation_state
        common["continuation_state_device"] = "cpu"

    mode = template["mode"]
    if mode == "voice_clone":
        return model.generate_voice_clone(
            ref_audio=template["ref_audio_path"],
            ref_text=template["ref_text"],
            xvec_only=template["xvec_only"],
            **common,
        )
    if mode == "custom":
        return model.generate_custom_voice(
            speaker=template["speaker"],
            instruct=template["instruct"],
            **common,
        )
    return model.generate_voice_design(
        instruct=template["instruct"],
        **common,
    )


# ─── Routes ───────────────────────────────────────────────────────────────────

_fetch_preset_assets()
_load_preset_refs()

@app.get("/")
async def root():
    return FileResponse(Path(__file__).parent / "index.html")


@app.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    """Transcribe reference audio using nano-parakeet."""
    if _parakeet is None:
        raise HTTPException(status_code=503, detail="Transcription model not loaded")

    content = await audio.read()
    if len(content) > MAX_AUDIO_BYTES:
        raise HTTPException(
            status_code=400,
            detail=_AUDIO_TOO_LARGE_MSG.format(size_mb=len(content) / 1024 / 1024),
        )

    def run():
        wav, sr = sf.read(io.BytesIO(content), dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        wav_t = torch.from_numpy(wav)
        if sr != 16000:
            wav_t = torchaudio.functional.resample(wav_t.unsqueeze(0), sr, 16000).squeeze(0)
        return _parakeet.transcribe(wav_t.cuda())

    text = await asyncio.to_thread(run)
    return {"text": text}


@app.get("/status")
async def get_status():
    speakers = []
    model_type = None
    active = _model_cache.get(_active_model_name) if _active_model_name else None
    if active is not None:
        try:
            model_type = active.model.model.tts_model_type
            speakers = active.model.get_supported_speakers() or []
        except Exception:
            speakers = []
    return {
        "loaded": active is not None,
        "model": _active_model_name,
        "loading": _loading,
        "available_models": AVAILABLE_MODELS,
        "model_type": model_type,
        "speakers": speakers,
        "transcription_available": _parakeet is not None,
        "preset_refs": [
            {"id": p["id"], "label": p["label"], "ref_text": p["ref_text"]}
            for p in _preset_refs.values()
        ],
        "queue_depth": _generation_waiters,
        "cached_models": list(_model_cache.keys()),
    }


@app.get("/preset_ref/{preset_id}")
async def get_preset_ref(preset_id: str):
    preset = _preset_refs.get(preset_id)
    if not preset:
        raise HTTPException(status_code=404, detail="Preset not found")
    return {
        "id": preset["id"],
        "label": preset["label"],
        "filename": preset["filename"],
        "ref_text": preset["ref_text"],
        "audio_b64": preset["audio_b64"],
    }


@app.post("/load")
async def load_model(model_id: str = Form(...)):
    global _active_model_name, _loading

    # Already in cache — instant switch, no GPU work needed
    if model_id in _model_cache:
        _clear_continuation_sessions()
        _active_model_name = model_id
        _model_cache.move_to_end(model_id)
        return {"status": "already_loaded", "model": model_id}

    _loading = True

    def _do_load():
        global _active_model_name, _loading
        try:
            _clear_continuation_sessions()
            if len(_model_cache) >= _model_cache_max:
                evicted, _ = _model_cache.popitem(last=False)
                print(f"Model cache full — evicted: {evicted}")
            new_model = FasterQwen3TTS.from_pretrained(
                model_id,
                device="cuda",
                dtype=torch.bfloat16,
            )
            print("Capturing CUDA graphs…")
            new_model._warmup(prefill_len=100)
            _model_cache[model_id] = new_model
            _model_cache.move_to_end(model_id)
            _active_model_name = model_id
            _prime_preset_voice_cache(new_model)
            print("CUDA graphs captured — model ready.")
        finally:
            _loading = False

    # Hold the generation lock while loading to prevent OOM from concurrent inference
    async with _generation_lock:
        await asyncio.to_thread(_do_load)
    return {"status": "loaded", "model": model_id}


@app.post("/generate/stream")
async def generate_stream(
    text: str = Form(...),
    language: str = Form("English"),
    mode: str = Form("voice_clone"),
    ref_text: str = Form(""),
    speaker: str = Form(""),
    instruct: str = Form(""),
    xvec_only: bool = Form(True),
    chunk_size: int = Form(8),
    temperature: float = Form(0.9),
    top_k: int = Form(50),
    repetition_penalty: float = Form(1.05),
    non_streaming_mode: bool | None = Form(None),
    ref_preset: str = Form(""),
    ref_audio: UploadFile = File(None),
):
    if not _active_model_name or _active_model_name not in _model_cache:
        raise HTTPException(status_code=400, detail="Model not loaded. Click 'Load' first.")
    if len(text) > MAX_TEXT_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"Text too long ({len(text)} chars). Maximum is {MAX_TEXT_CHARS} characters.",
        )

    tmp_path = None
    tmp_is_cached = False

    if ref_preset and ref_preset in _preset_refs:
        preset = _preset_refs[ref_preset]
        tmp_path = preset["path"]
        tmp_is_cached = True
        if not ref_text:
            ref_text = preset["ref_text"]
    elif ref_audio and ref_audio.filename:
        content = await ref_audio.read()
        if len(content) > MAX_AUDIO_BYTES:
            raise HTTPException(
                status_code=400,
                detail=_AUDIO_TOO_LARGE_MSG.format(size_mb=len(content) / 1024 / 1024),
            )
        tmp_path = _get_cached_ref_path(content)
        tmp_is_cached = True

    if non_streaming_mode is None:
        non_streaming_mode = _default_non_streaming_mode_for_mode(mode)

    template = _build_continuation_template(
        mode=mode,
        language=language,
        non_streaming_mode=non_streaming_mode,
        ref_audio_path=tmp_path,
        ref_text=ref_text,
        xvec_only=xvec_only,
        speaker=speaker,
        instruct=instruct,
    )

    loop = asyncio.get_event_loop()
    queue: asyncio.Queue[str | None] = asyncio.Queue()

    def run_generation():
        try:
            # Resolve the model after the generation lock is held so we always
            # use the currently active model, not a stale reference captured
            # before a concurrent /load request changed the active model.
            model = _model_cache.get(_active_model_name)
            if model is None:
                raise RuntimeError("No model loaded. Please load a model first.")

            t0 = time.perf_counter()
            total_audio_s = 0.0
            voice_clone_ms = 0.0
            running_state = None
            running_tail = np.zeros(0, dtype=np.float32)
            running_sr = None

            if mode == "voice_clone":
                gen = model.generate_voice_clone_streaming(
                    text=text,
                    language=language,
                    ref_audio=tmp_path,
                    ref_text=ref_text,
                    xvec_only=xvec_only,
                    non_streaming_mode=non_streaming_mode,
                    chunk_size=chunk_size,
                    temperature=temperature,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    return_continuation_state="delta",
                    continuation_state_device="cpu",
                    max_new_tokens=360,  # cap at 30s (12 Hz codec)
                )
            elif mode == "custom":
                if not speaker:
                    raise ValueError("Speaker ID is required for custom voice")
                gen = model.generate_custom_voice_streaming(
                    text=text,
                    speaker=speaker,
                    language=language,
                    instruct=instruct,
                    non_streaming_mode=non_streaming_mode,
                    chunk_size=chunk_size,
                    temperature=temperature,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    return_continuation_state="delta",
                    continuation_state_device="cpu",
                    max_new_tokens=360,
                )
            else:
                gen = model.generate_voice_design_streaming(
                    text=text,
                    instruct=instruct,
                    language=language,
                    non_streaming_mode=non_streaming_mode,
                    chunk_size=chunk_size,
                    temperature=temperature,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    return_continuation_state="delta",
                    continuation_state_device="cpu",
                    max_new_tokens=360,
                )

            # Use timing data from the generator itself (measured after voice-clone
            # encoding, so TTFA and RTF reflect pure LLM generation latency).
            ttfa_ms = None
            total_gen_ms = 0.0

            # Prime generator to capture wall-clock time to first chunk
            first_audio = next(gen, None)
            if first_audio is not None:
                audio_chunk, sr, timing = first_audio
                delta = timing.get("continuation_state_delta")
                if delta is not None:
                    running_state = apply_continuation_state_delta(running_state, delta)
                wall_first_ms = (time.perf_counter() - t0) * 1000
                model_ms = timing.get("prefill_ms", 0) + timing.get("decode_ms", 0)
                voice_clone_ms = max(0.0, wall_first_ms - model_ms)
                total_gen_ms += timing.get('prefill_ms', 0) + timing.get('decode_ms', 0)
                if ttfa_ms is None:
                    ttfa_ms = total_gen_ms

                audio_chunk = _concat_audio(audio_chunk)
                running_sr = sr
                running_tail = _append_audio_tail(running_tail, audio_chunk, sr)
                dur = len(audio_chunk) / sr
                total_audio_s += dur
                rtf = total_audio_s / (total_gen_ms / 1000) if total_gen_ms > 0 else 0.0

                audio_b64 = _to_wav_b64(audio_chunk, sr)
                payload = {
                    "type": "chunk",
                    "audio_b64": audio_b64,
                    "sample_rate": sr,
                    "ttfa_ms": round(ttfa_ms),
                    "voice_clone_ms": round(voice_clone_ms),
                    "rtf": round(rtf, 3),
                    "total_audio_s": round(total_audio_s, 3),
                    "elapsed_ms": round(time.perf_counter() - t0, 3) * 1000,
                }
                loop.call_soon_threadsafe(queue.put_nowait, json.dumps(payload))

            for audio_chunk, sr, timing in gen:
                delta = timing.get("continuation_state_delta")
                if delta is not None:
                    running_state = apply_continuation_state_delta(running_state, delta)
                # prefill_ms is non-zero only on the first chunk
                total_gen_ms += timing.get('prefill_ms', 0) + timing.get('decode_ms', 0)
                if ttfa_ms is None:
                    ttfa_ms = total_gen_ms  # already in ms

                audio_chunk = _concat_audio(audio_chunk)
                running_sr = sr
                running_tail = _append_audio_tail(running_tail, audio_chunk, sr)
                dur = len(audio_chunk) / sr
                total_audio_s += dur
                rtf = total_audio_s / (total_gen_ms / 1000) if total_gen_ms > 0 else 0.0

                audio_b64 = _to_wav_b64(audio_chunk, sr)
                payload = {
                    "type": "chunk",
                    "audio_b64": audio_b64,
                    "sample_rate": sr,
                    "ttfa_ms": round(ttfa_ms),
                    "voice_clone_ms": round(voice_clone_ms),
                    "rtf": round(rtf, 3),
                    "total_audio_s": round(total_audio_s, 3),
                    "elapsed_ms": round(time.perf_counter() - t0, 3) * 1000,
                }
                loop.call_soon_threadsafe(queue.put_nowait, json.dumps(payload))

            rtf = total_audio_s / (total_gen_ms / 1000) if total_gen_ms > 0 else 0.0
            session_id = None
            if running_state is not None:
                session_id = _store_continuation_session(
                    model_name=_active_model_name,
                    template=template,
                    state=running_state,
                    audio_tail=running_tail,
                    sample_rate=running_sr or 24000,
                )
            done_payload = {
                "type": "done",
                "ttfa_ms": round(ttfa_ms) if ttfa_ms else 0,
                "voice_clone_ms": round(voice_clone_ms),
                "rtf": round(rtf, 3),
                "total_audio_s": round(total_audio_s, 3),
                "total_ms": round((time.perf_counter() - t0) * 1000),
                "continuation_session_id": session_id,
            }
            loop.call_soon_threadsafe(queue.put_nowait, json.dumps(done_payload))

        except Exception as e:
            import traceback
            err = {"type": "error", "message": str(e), "detail": traceback.format_exc()}
            loop.call_soon_threadsafe(queue.put_nowait, json.dumps(err))
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)
            if tmp_path and os.path.exists(tmp_path) and not tmp_is_cached:
                os.unlink(tmp_path)

    async def sse():
        global _generation_waiters
        lock_acquired = False
        _generation_waiters += 1
        people_ahead = _generation_waiters - 1 + (1 if _generation_lock.locked() else 0)
        try:
            if people_ahead > 0:
                yield f"data: {json.dumps({'type': 'queued', 'position': people_ahead})}\n\n"

            await _generation_lock.acquire()
            lock_acquired = True
            _generation_waiters -= 1

            thread = threading.Thread(target=run_generation, daemon=True)
            thread.start()

            while True:
                msg = await queue.get()
                if msg is None:
                    break
                yield f"data: {msg}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            if lock_acquired:
                _generation_lock.release()
            else:
                _generation_waiters -= 1

    return StreamingResponse(
        sse(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )




@app.post("/generate")
async def generate_non_streaming(
    text: str = Form(...),
    language: str = Form("English"),
    mode: str = Form("voice_clone"),
    ref_text: str = Form(""),
    speaker: str = Form(""),
    instruct: str = Form(""),
    xvec_only: bool = Form(True),
    temperature: float = Form(0.9),
    top_k: int = Form(50),
    repetition_penalty: float = Form(1.05),
    non_streaming_mode: bool | None = Form(None),
    ref_preset: str = Form(""),
    ref_audio: UploadFile = File(None),
):
    if not _active_model_name or _active_model_name not in _model_cache:
        raise HTTPException(status_code=400, detail="Model not loaded. Click 'Load' first.")
    if len(text) > MAX_TEXT_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"Text too long ({len(text)} chars). Maximum is {MAX_TEXT_CHARS} characters.",
        )

    tmp_path = None
    tmp_is_cached = False

    if ref_preset and ref_preset in _preset_refs:
        preset = _preset_refs[ref_preset]
        tmp_path = preset["path"]
        tmp_is_cached = True
        if not ref_text:
            ref_text = preset["ref_text"]
    elif ref_audio and ref_audio.filename:
        content = await ref_audio.read()
        if len(content) > MAX_AUDIO_BYTES:
            raise HTTPException(
                status_code=400,
                detail=_AUDIO_TOO_LARGE_MSG.format(size_mb=len(content) / 1024 / 1024),
            )
        tmp_path = _get_cached_ref_path(content)
        tmp_is_cached = True

    if non_streaming_mode is None:
        non_streaming_mode = _default_non_streaming_mode_for_mode(mode)

    template = _build_continuation_template(
        mode=mode,
        language=language,
        non_streaming_mode=non_streaming_mode,
        ref_audio_path=tmp_path,
        ref_text=ref_text,
        xvec_only=xvec_only,
        speaker=speaker,
        instruct=instruct,
    )

    def run():
        # Resolve the model after the generation lock is held.
        model = _model_cache.get(_active_model_name)
        if model is None:
            raise RuntimeError("No model loaded. Please load a model first.")
        t0 = time.perf_counter()
        if mode == "custom" and not speaker:
            raise ValueError("Speaker ID is required for custom voice")
        result = _run_demo_generation(
            model,
            template=template,
            text=text,
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            return_continuation_state="full",
            max_new_tokens=360,
        )
        audio_list, sr, info = result
        elapsed = time.perf_counter() - t0
        audio = _concat_audio(audio_list)
        dur = len(audio) / sr
        session_id = None
        state = info.get("continuation_state") if info else None
        if state is not None:
            session_id = _store_continuation_session(
                model_name=_active_model_name,
                template=template,
                state=state,
                audio_tail=_trim_audio_tail(audio, sr),
                sample_rate=sr,
            )
        return audio, sr, elapsed, dur, session_id

    global _generation_waiters
    _generation_waiters += 1
    lock_acquired = False
    try:
        await _generation_lock.acquire()
        lock_acquired = True
        _generation_waiters -= 1
        audio, sr, elapsed, dur, session_id = await asyncio.to_thread(run)
        rtf = dur / elapsed if elapsed > 0 else 0.0
        return JSONResponse({
            "audio_b64": _to_wav_b64(audio, sr),
            "sample_rate": sr,
            "continuation_session_id": session_id,
            "metrics": {
                "total_ms": round(elapsed * 1000),
                "audio_duration_s": round(dur, 3),
                "rtf": round(rtf, 3),
            },
        })
    finally:
        if lock_acquired:
            _generation_lock.release()
        else:
            _generation_waiters -= 1
        if tmp_path and os.path.exists(tmp_path) and not tmp_is_cached:
            os.unlink(tmp_path)


@app.post("/generate/compare_continuation")
async def generate_continuation_compare(
    session_id: str = Form(...),
    text: str = Form(...),
    temperature: float = Form(0.9),
    top_k: int = Form(50),
    repetition_penalty: float = Form(1.05),
):
    if not _active_model_name or _active_model_name not in _model_cache:
        raise HTTPException(status_code=400, detail="Model not loaded. Click 'Load' first.")
    if len(text) > MAX_TEXT_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"Text too long ({len(text)} chars). Maximum is {MAX_TEXT_CHARS} characters.",
        )

    def run():
        model = _model_cache.get(_active_model_name)
        if model is None:
            raise RuntimeError("No model loaded. Please load a model first.")

        session = _get_continuation_session(session_id)
        if session is None:
            raise RuntimeError("Saved continuation state not found. Generate the first sentence again.")
        if session["model_name"] != _active_model_name:
            raise RuntimeError("Saved continuation state belongs to a different model. Generate the first sentence again.")

        template = session["template"]
        mode = template["mode"]
        if mode == "custom" and not template["speaker"]:
            raise RuntimeError("Saved continuation state is missing the custom speaker.")
        context_audio = np.array(session.get("audio_tail", np.zeros(0, dtype=np.float32)), dtype=np.float32)
        context_sr = int(session.get("sample_rate", 24000))

        t0 = time.perf_counter()
        fresh_result = _run_demo_generation(
            model,
            template=template,
            text=text,
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            max_new_tokens=360,
        )
        fresh_elapsed = time.perf_counter() - t0
        fresh_audio_list, fresh_sr = fresh_result
        fresh_audio = _concat_audio(fresh_audio_list)
        fresh_compare_audio = _prepend_context_audio(context_audio, context_sr, fresh_audio, fresh_sr)
        fresh_dur = len(fresh_audio) / fresh_sr

        t1 = time.perf_counter()
        continued_result = _run_demo_generation(
            model,
            template=template,
            text=text,
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            continuation_state=session["state"],
            max_new_tokens=360,
        )
        cont_elapsed = time.perf_counter() - t1
        cont_audio_list, cont_sr = continued_result
        cont_audio = _concat_audio(cont_audio_list)
        cont_compare_audio = _prepend_context_audio(context_audio, context_sr, cont_audio, cont_sr)
        cont_dur = len(cont_audio) / cont_sr

        return {
            "continuation_session_id": session_id,
            "context_seconds": CONTINUATION_COMPARE_CONTEXT_SECONDS,
            "fresh": {
                "audio_b64": _to_wav_b64(fresh_compare_audio, fresh_sr),
                "sample_rate": fresh_sr,
                "metrics": {
                    "total_ms": round(fresh_elapsed * 1000),
                    "audio_duration_s": round(fresh_dur, 3),
                    "rtf": round(fresh_dur / fresh_elapsed, 3) if fresh_elapsed > 0 else 0.0,
                },
            },
            "continued": {
                "audio_b64": _to_wav_b64(cont_compare_audio, cont_sr),
                "sample_rate": cont_sr,
                "metrics": {
                    "total_ms": round(cont_elapsed * 1000),
                    "audio_duration_s": round(cont_dur, 3),
                    "rtf": round(cont_dur / cont_elapsed, 3) if cont_elapsed > 0 else 0.0,
                },
            },
        }

    global _generation_waiters
    _generation_waiters += 1
    lock_acquired = False
    try:
        await _generation_lock.acquire()
        lock_acquired = True
        _generation_waiters -= 1
        return JSONResponse(await asyncio.to_thread(run))
    finally:
        if lock_acquired:
            _generation_lock.release()
        else:
            _generation_waiters -= 1


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Faster Qwen3-TTS Demo Server")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        help="Model to preload at startup (default: 1.7B-Base)",
    )
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 7860)))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument(
        "--no-preload",
        action="store_true",
        help="Skip model loading at startup (load via UI instead)",
    )
    args = parser.parse_args()

    if not args.no_preload:
        global _active_model_name, _parakeet
        print(f"Loading model: {args.model}")
        _startup_model = FasterQwen3TTS.from_pretrained(
            args.model,
            device="cuda",
            dtype=torch.bfloat16,
        )
        print("Capturing CUDA graphs…")
        _startup_model._warmup(prefill_len=100)
        _model_cache[args.model] = _startup_model
        _active_model_name = args.model
        _prime_preset_voice_cache(_startup_model)
        print("TTS model ready.")

        print("Loading transcription model (nano-parakeet)…")
        _parakeet = _parakeet_from_pretrained(device="cuda")
        print("Transcription model ready.")

        print(f"Ready. Open http://localhost:{args.port}")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
