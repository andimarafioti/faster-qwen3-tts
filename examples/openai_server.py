#!/usr/bin/env python3
"""
OpenAI-compatible TTS API server for faster-qwen3-tts.

Exposes POST /v1/audio/speech compatible with OpenAI's TTS API, enabling
integration with OpenWebUI, llama-swap, and other OpenAI-compatible clients.

Usage:
    pip install "faster-qwen3-tts[demo]"

    # Single default voice:
    python examples/openai_server.py \\
        --ref-audio voice.wav --ref-text "Reference transcription" \\
        --language English

    # Multiple named voices from a JSON config:
    python examples/openai_server.py --voices voices.json

    # Custom model and port:
    python examples/openai_server.py \\
        --model Qwen/Qwen3-TTS-12Hz-0.6B-Base \\
        --ref-audio voice.wav --ref-text "transcript" \\
        --port 8000

Voices config (voices.json):
    {
        "alloy": {"ref_audio": "voice.wav", "ref_text": "...", "language": "English"},
        "echo":  {"ref_audio": "voice2.wav", "ref_text": "...", "language": "English"}
    }

API usage:
    curl -s http://localhost:8000/v1/audio/speech \\
        -H "Content-Type: application/json" \\
        -d '{"model": "tts-1", "input": "Hello!", "voice": "alloy", "response_format": "wav"}' \\
        --output speech.wav
"""
import argparse
import asyncio
import io
import json
import logging
import os
import queue
import re
import struct
import sys
import tempfile
import threading
from pathlib import Path
from typing import AsyncGenerator, Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

app = FastAPI(title="faster-qwen3-tts OpenAI-compatible API")

tts_model = None
voices: dict = {}
default_voice: Optional[str] = None
SAMPLE_RATE = 24000  # updated once the model loads
_model_lock = threading.Lock()  # prevent concurrent GPU inference

# Voice management state. Populated by main() when --voices is used.
# voices_json_path is the source-of-truth JSON file on disk that we
# atomically rewrite on upload / delete. voice_samples_dir is where
# uploaded WAVs live.
voices_json_path: Optional[Path] = None
voice_samples_dir: Optional[Path] = None
_voices_lock = threading.Lock()  # protect voices dict + voices.json file

# Voice names must be filesystem-safe and URL-safe. Matches a-z, 0-9,
# underscore, hyphen — no dots, slashes, or whitespace.
_VOICE_NAME_RE = re.compile(r"^[A-Za-z0-9_\-]+$")

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class SpeechRequest(BaseModel):
    model: str = "tts-1"
    input: str
    voice: str = "alloy"
    response_format: str = "wav"  # wav | pcm | mp3
    speed: float = 1.0           # accepted but not yet applied


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------


def _to_pcm16(pcm: np.ndarray) -> bytes:
    """Convert float32 numpy array to raw 16-bit little-endian PCM bytes."""
    return np.clip(pcm * 32768, -32768, 32767).astype(np.int16).tobytes()


def _wav_header(sample_rate: int, data_len: int = 0xFFFFFFFF) -> bytes:
    """Build a WAV header.  Use data_len=0xFFFFFFFF for streaming (unknown size)."""
    n_channels = 1
    bits = 16
    byte_rate = sample_rate * n_channels * bits // 8
    block_align = n_channels * bits // 8
    riff_size = 0xFFFFFFFF if data_len == 0xFFFFFFFF else 36 + data_len
    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", riff_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, n_channels, sample_rate,
                          byte_rate, block_align, bits))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_len))
    return buf.getvalue()


def _to_wav_bytes(pcm: np.ndarray, sample_rate: int) -> bytes:
    """Convert float32 numpy array to a complete WAV file in memory."""
    raw = _to_pcm16(pcm)
    return _wav_header(sample_rate, len(raw)) + raw


def _to_mp3_bytes(pcm: np.ndarray, sample_rate: int) -> bytes:
    """Convert float32 numpy array to MP3 bytes (requires pydub + ffmpeg)."""
    try:
        from pydub import AudioSegment
    except ImportError:
        raise HTTPException(
            status_code=400,
            detail="response_format='mp3' requires pydub: pip install pydub",
        )
    segment = AudioSegment(
        _to_pcm16(pcm),
        frame_rate=sample_rate,
        sample_width=2,
        channels=1,
    )
    buf = io.BytesIO()
    segment.export(buf, format="mp3")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Voice resolution
# ---------------------------------------------------------------------------


def resolve_voice(voice_name: str) -> dict:
    """Return voice config dict or fall back to default, else raise 400."""
    if voice_name in voices:
        return voices[voice_name]
    if default_voice and default_voice in voices:
        logger.warning(
            "Voice %r not configured; falling back to default voice %r",
            voice_name,
            default_voice,
        )
        return voices[default_voice]
    raise HTTPException(
        status_code=400,
        detail=(
            f"Voice {voice_name!r} is not configured. "
            f"Available voices: {list(voices.keys())}"
        ),
    )


def _validate_voice_name(name: str) -> None:
    """Raise 400 if the voice name is empty or unsafe for the filesystem."""
    if not name or not _VOICE_NAME_RE.match(name):
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid voice name. Use only letters, digits, underscore, "
                "and hyphen (no dots, slashes, or whitespace)."
            ),
        )


def _persist_voices_unlocked() -> None:
    """Rewrite voices.json atomically. Caller must hold _voices_lock.

    Note on bind mounts: atomic rename only works when voices.json lives
    inside a *directory* bind mount, not a file bind mount. The latter
    binds the kernel mountpoint to the file itself, and os.replace()
    across it returns EBUSY. Mount the parent directory instead.
    """
    if voices_json_path is None:
        # Server wasn't started with --voices; nothing to persist.
        return
    parent = voices_json_path.parent
    parent.mkdir(parents=True, exist_ok=True)
    # Write to a sibling temp file then atomically rename so a crash
    # mid-write can't corrupt voices.json.
    fd, tmp_path = tempfile.mkstemp(
        prefix=".voices-", suffix=".json.tmp", dir=str(parent)
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(voices, f, indent=2, ensure_ascii=False)
            f.write("\n")
        # mkstemp creates files with secure 0600 perms. When voices.json
        # is on a host bind mount the server owner differs from the host
        # user, so widen to 0644 so the host user (and any sidecar tools
        # like add_voice.py) can still read the file.
        os.chmod(tmp_path, 0o644)
        os.replace(tmp_path, str(voices_json_path))
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Streaming helper: run sync generator in a background thread
# ---------------------------------------------------------------------------


async def _stream_chunks(voice_cfg: dict, text: str) -> AsyncGenerator[bytes, None]:
    """
    Run generate_voice_clone_streaming in a background thread and yield
    raw PCM bytes for each chunk as they arrive.
    """
    q: queue.Queue = queue.Queue()
    _DONE = object()

    def producer():
        try:
            with _model_lock:
                for chunk, _sr, _timing in tts_model.generate_voice_clone_streaming(
                    text=text,
                    language=voice_cfg.get("language", "Auto"),
                    ref_audio=voice_cfg["ref_audio"],
                    ref_text=voice_cfg.get("ref_text", ""),
                    chunk_size=voice_cfg.get("chunk_size", 12),
                    non_streaming_mode=False,
                ):
                    q.put(chunk)
        except Exception as exc:
            q.put(exc)
        finally:
            q.put(_DONE)

    thread = threading.Thread(target=producer, daemon=True)
    thread.start()

    loop = asyncio.get_event_loop()
    while True:
        item = await loop.run_in_executor(None, q.get)
        if item is _DONE:
            break
        if isinstance(item, Exception):
            raise item
        yield _to_pcm16(item)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": tts_model is not None}


# ---------------------------------------------------------------------------
# Voice management API
#
# Mirrors the shape of vllm-omni's /v1/audio/voices endpoints so existing
# clients can drop in unchanged. Uploaded voices are written to
# voice_samples_dir (next to voices.json by default) and the voices.json
# file is rewritten atomically on every mutation.
#
# Requires --voices <file> (or QWEN_TTS_VOICES); when using --ref-audio
# in single-voice mode there is no persistent registry to mutate.
# ---------------------------------------------------------------------------


@app.get("/v1/audio/voices")
async def list_voices():
    with _voices_lock:
        names = sorted(voices.keys())
        details = [
            {
                "name": name,
                "ref_text": cfg.get("ref_text", ""),
                "language": cfg.get("language", "Auto"),
                "ref_audio": cfg.get("ref_audio", ""),
            }
            for name, cfg in voices.items()
        ]
    return {"voices": names, "uploaded_voices": details}


@app.post("/v1/audio/voices")
async def upload_voice(
    audio_sample: UploadFile = File(...),
    name: str = Form(...),
    ref_text: str = Form(""),
    language: str = Form("English"),
    consent: Optional[str] = Form(None),  # accepted for vllm-omni compat; unused
):
    if voices_json_path is None or voice_samples_dir is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Server was started without --voices, so runtime voice "
                "management is disabled. Restart with --voices <file> to "
                "enable upload/delete."
            ),
        )
    _validate_voice_name(name)
    if consent:
        logger.info("Voice upload %r received consent field: %r", name, consent)

    content = await audio_sample.read()
    if not content:
        raise HTTPException(status_code=400, detail="audio_sample is empty")

    dest = voice_samples_dir / f"{name}.wav"

    with _voices_lock:
        voice_samples_dir.mkdir(parents=True, exist_ok=True)
        existed = name in voices
        try:
            dest.write_bytes(content)
        except OSError as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to write reference audio to {dest}: {exc}",
            )
        voices[name] = {
            "ref_audio": str(dest),
            "ref_text": ref_text,
            "language": language,
        }
        try:
            _persist_voices_unlocked()
        except OSError as exc:
            # Roll back the in-memory change so state stays consistent.
            if existed:
                logger.exception("voices.json write failed; in-memory entry kept")
            else:
                voices.pop(name, None)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to persist voices.json: {exc}",
            )

        global default_voice
        if default_voice is None or default_voice not in voices:
            default_voice = name

    return {
        "name": name,
        "status": "replaced" if existed else "created",
        "ref_audio": str(dest),
        "file_size": len(content),
    }


@app.delete("/v1/audio/voices/{name}")
async def delete_voice(name: str):
    if voices_json_path is None:
        raise HTTPException(
            status_code=503,
            detail="Server was started without --voices; nothing to delete.",
        )
    _validate_voice_name(name)

    with _voices_lock:
        if name not in voices:
            raise HTTPException(
                status_code=404, detail=f"Voice {name!r} not found"
            )
        entry = voices.pop(name)
        try:
            _persist_voices_unlocked()
        except OSError as exc:
            voices[name] = entry  # roll back
            raise HTTPException(
                status_code=500,
                detail=f"Failed to persist voices.json: {exc}",
            )

        # Best-effort delete of the reference WAV. Only remove files that
        # live under our managed voice_samples_dir, so we never accidentally
        # unlink a WAV the user pointed at via an absolute path in voices.json.
        ref_audio = entry.get("ref_audio", "")
        if ref_audio and voice_samples_dir is not None:
            try:
                ref_path = Path(ref_audio).resolve()
                vsd = voice_samples_dir.resolve()
                if ref_path.is_relative_to(vsd) and ref_path.exists():
                    ref_path.unlink()
            except (OSError, ValueError) as exc:
                logger.warning("Failed to delete %s: %s", ref_audio, exc)

        global default_voice
        if default_voice == name:
            default_voice = next(iter(voices), None)

    return {"name": name, "status": "deleted"}


@app.post("/v1/audio/speech")
async def create_speech(req: SpeechRequest):
    if tts_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not req.input.strip():
        raise HTTPException(status_code=400, detail="'input' text is empty")

    voice_cfg = resolve_voice(req.voice)
    fmt = req.response_format.lower()

    _CONTENT_TYPES = {
        "wav": "audio/wav",
        "pcm": "audio/pcm",
        "mp3": "audio/mpeg",
    }
    if fmt not in _CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"response_format {fmt!r} not supported. Use: wav, pcm, mp3",
        )
    content_type = _CONTENT_TYPES[fmt]

    # --- MP3: generate all audio, then encode (non-streaming) ---
    if fmt == "mp3":
        loop = asyncio.get_event_loop()

        def _generate():
            with _model_lock:
                return tts_model.generate_voice_clone(
                    text=req.input,
                    language=voice_cfg.get("language", "Auto"),
                    ref_audio=voice_cfg["ref_audio"],
                    ref_text=voice_cfg.get("ref_text", ""),
                )

        audio_arrays, sr = await loop.run_in_executor(None, _generate)
        audio = audio_arrays[0] if audio_arrays else np.zeros(1, dtype=np.float32)
        return Response(content=_to_mp3_bytes(audio, sr), media_type=content_type)

    # --- WAV / PCM: stream chunks as they are generated ---
    async def audio_stream():
        if fmt == "wav":
            yield _wav_header(SAMPLE_RATE)  # stream with unknown data length
        async for raw_chunk in _stream_chunks(voice_cfg, req.input):
            yield raw_chunk

    return StreamingResponse(audio_stream(), media_type=content_type)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _parse_args():
    p = argparse.ArgumentParser(
        description="OpenAI-compatible TTS server for faster-qwen3-tts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--model",
        default=os.environ.get("QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"),
        help="HuggingFace model ID or local path (default: Qwen/Qwen3-TTS-12Hz-1.7B-Base)",
    )
    p.add_argument(
        "--voices",
        default=os.environ.get("QWEN_TTS_VOICES"),
        metavar="FILE",
        help="JSON file mapping voice names to {ref_audio, ref_text, language}",
    )
    p.add_argument(
        "--ref-audio",
        default=os.environ.get("QWEN_TTS_REF_AUDIO"),
        metavar="FILE",
        help="Reference audio file when --voices is not used",
    )
    p.add_argument(
        "--ref-text",
        default=os.environ.get("QWEN_TTS_REF_TEXT", ""),
        help="Transcript of --ref-audio",
    )
    p.add_argument(
        "--language",
        default=os.environ.get("QWEN_TTS_LANGUAGE", "Auto"),
        help="Target language (English, French, Auto, …) when --voices is not used",
    )
    p.add_argument(
        "--voice-samples-dir",
        default=os.environ.get("QWEN_TTS_VOICE_SAMPLES_DIR"),
        metavar="DIR",
        help=(
            "Directory where uploaded reference WAVs are written via "
            "POST /v1/audio/voices. Defaults to <dir of voices.json>/voice_samples."
        ),
    )
    p.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    p.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    p.add_argument("--device", default="cuda", help="Torch device (default: cuda)")
    return p.parse_args()


def main():
    global tts_model, voices, default_voice, SAMPLE_RATE
    global voices_json_path, voice_samples_dir

    args = _parse_args()

    # Build voice registry
    if args.voices:
        with open(args.voices) as f:
            voices = json.load(f)
        default_voice = next(iter(voices)) if voices else None
        voices_json_path = Path(args.voices).resolve()
        # Uploaded WAVs live next to voices.json by default.
        if args.voice_samples_dir:
            voice_samples_dir = Path(args.voice_samples_dir).resolve()
        else:
            voice_samples_dir = voices_json_path.parent / "voice_samples"
        voice_samples_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Loaded %d voice(s) from %s (samples dir: %s)",
            len(voices),
            args.voices,
            voice_samples_dir,
        )
    elif args.ref_audio:
        voices = {
            "default": {
                "ref_audio": args.ref_audio,
                "ref_text": args.ref_text,
                "language": args.language,
            }
        }
        default_voice = "default"
        logger.info("Using single voice from --ref-audio: %s", args.ref_audio)
    else:
        print(
            "ERROR: provide --ref-audio <file> or --voices <config.json>",
            file=sys.stderr,
        )
        sys.exit(1)

    from faster_qwen3_tts import FasterQwen3TTS

    logger.info("Loading model %s on %s …", args.model, args.device)
    tts_model = FasterQwen3TTS.from_pretrained(
        args.model,
        device=args.device,
        dtype=torch.bfloat16,
    )
    SAMPLE_RATE = tts_model.sample_rate
    logger.info("Model ready. Sample rate: %d Hz", SAMPLE_RATE)
    logger.info("Server listening on http://%s:%d", args.host, args.port)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
