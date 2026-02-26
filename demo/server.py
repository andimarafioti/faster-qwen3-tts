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
    from faster_qwen3_tts import FasterQwen3TTS
except ImportError:
    print("Error: faster_qwen3_tts not found.")
    print("Install with:  pip install -e .  (from the repo root)")
    sys.exit(1)

from nano_parakeet import from_pretrained as _parakeet_from_pretrained


AVAILABLE_MODELS = [
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
]

BASE_DIR = Path(__file__).resolve().parent
PRESET_TRANSCRIPTS = BASE_DIR / "samples" / "parity" / "icl_transcripts.txt"
PRESET_REFS = [
    ("ref_audio_3", BASE_DIR / "ref_audio_3.wav", "Clone 1"),
    ("ref_audio_2", BASE_DIR / "ref_audio_2.wav", "Clone 2"),
    ("ref_audio", BASE_DIR / "ref_audio.wav", "Clone 3"),
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
        _active_model_name = model_id
        _model_cache.move_to_end(model_id)
        return {"status": "already_loaded", "model": model_id}

    _loading = True

    def _do_load():
        global _active_model_name, _loading
        try:
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
    ref_preset: str = Form(""),
    ref_audio: UploadFile = File(None),
):
    if not _active_model_name or _active_model_name not in _model_cache:
        raise HTTPException(status_code=400, detail="Model not loaded. Click 'Load' first.")

    model = _model_cache[_active_model_name]
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
        tmp_path = _get_cached_ref_path(content)
        tmp_is_cached = True

    loop = asyncio.get_event_loop()
    queue: asyncio.Queue[str | None] = asyncio.Queue()

    def run_generation():
        try:
            t0 = time.perf_counter()
            total_audio_s = 0.0
            voice_clone_ms = 0.0

            if mode == "voice_clone":
                gen = model.generate_voice_clone_streaming(
                    text=text,
                    language=language,
                    ref_audio=tmp_path,
                    ref_text=ref_text,
                    xvec_only=xvec_only,
                    chunk_size=chunk_size,
                    temperature=temperature,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
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
                    chunk_size=chunk_size,
                    temperature=temperature,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    max_new_tokens=360,
                )
            else:
                gen = model.generate_voice_design_streaming(
                    text=text,
                    instruct=instruct,
                    language=language,
                    chunk_size=chunk_size,
                    temperature=temperature,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
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
                wall_first_ms = (time.perf_counter() - t0) * 1000
                model_ms = timing.get("prefill_ms", 0) + timing.get("decode_ms", 0)
                voice_clone_ms = max(0.0, wall_first_ms - model_ms)
                total_gen_ms += timing.get('prefill_ms', 0) + timing.get('decode_ms', 0)
                if ttfa_ms is None:
                    ttfa_ms = total_gen_ms

                audio_chunk = _concat_audio(audio_chunk)
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
                # prefill_ms is non-zero only on the first chunk
                total_gen_ms += timing.get('prefill_ms', 0) + timing.get('decode_ms', 0)
                if ttfa_ms is None:
                    ttfa_ms = total_gen_ms  # already in ms

                audio_chunk = _concat_audio(audio_chunk)
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
            done_payload = {
                "type": "done",
                "ttfa_ms": round(ttfa_ms) if ttfa_ms else 0,
                "voice_clone_ms": round(voice_clone_ms),
                "rtf": round(rtf, 3),
                "total_audio_s": round(total_audio_s, 3),
                "total_ms": round((time.perf_counter() - t0) * 1000),
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
    ref_preset: str = Form(""),
    ref_audio: UploadFile = File(None),
):
    if not _active_model_name or _active_model_name not in _model_cache:
        raise HTTPException(status_code=400, detail="Model not loaded. Click 'Load' first.")

    model = _model_cache[_active_model_name]
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
        tmp_path = _get_cached_ref_path(content)
        tmp_is_cached = True

    def run():
        t0 = time.perf_counter()
        if mode == "voice_clone":
            audio_list, sr = model.generate_voice_clone(
                text=text,
                language=language,
                ref_audio=tmp_path,
                ref_text=ref_text,
                xvec_only=xvec_only,
                temperature=temperature,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                max_new_tokens=360,  # cap at 30s (12 Hz codec)
            )
        elif mode == "custom":
            if not speaker:
                raise ValueError("Speaker ID is required for custom voice")
            audio_list, sr = model.generate_custom_voice(
                text=text,
                speaker=speaker,
                language=language,
                instruct=instruct,
                temperature=temperature,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                max_new_tokens=360,
            )
        else:
            audio_list, sr = model.generate_voice_design(
                text=text,
                instruct=instruct,
                language=language,
                temperature=temperature,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                max_new_tokens=360,
            )
        elapsed = time.perf_counter() - t0
        audio = _concat_audio(audio_list)
        dur = len(audio) / sr
        return audio, sr, elapsed, dur

    global _generation_waiters
    _generation_waiters += 1
    lock_acquired = False
    try:
        await _generation_lock.acquire()
        lock_acquired = True
        _generation_waiters -= 1
        audio, sr, elapsed, dur = await asyncio.to_thread(run)
        rtf = dur / elapsed if elapsed > 0 else 0.0
        return JSONResponse({
            "audio_b64": _to_wav_b64(audio, sr),
            "sample_rate": sr,
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
