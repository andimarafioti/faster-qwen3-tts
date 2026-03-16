"""
Faster Qwen3 TTS Streaming API Server

Provides both HTTP and WebSocket endpoints for text-to-speech generation
with streaming audio output support using CUDA graphs for 6-10x speedup.
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel
from contextlib import asynccontextmanager
import torch
import numpy as np
import os
import logging
import time
import io
import struct
import json
from typing import Optional, AsyncIterator
from config import DEVICE, DTYPE, MODEL_NAME, ATTN_IMPLEMENTATION, VOICES_DIR, CHUNK_SIZE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Global model instance
model = None
# Voice metadata: maps voice name to (ref_audio_path, ref_text)
# FasterQwen3TTS caches the actual embeddings internally
voices = {}

# Enable TensorFloat32 for better performance on Ampere+ GPUs
torch.set_float32_matmul_precision('high')


def get_dtype(dtype_str: str):
    """Convert dtype string to torch dtype."""
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return dtype_map.get(dtype_str, torch.bfloat16)


def discover_voices(voices_dir: str) -> dict:
    """
    Discover all voice samples in voices_dir.

    Args:
        voices_dir: Directory containing voice samples

    Returns:
        dict: Mapping of voice name to (ref_audio_path, ref_text)
    """
    voice_map = {}

    if not os.path.exists(voices_dir):
        logging.warning(f"Voices directory not found: {voices_dir}")
        return voice_map

    # Find all .wav files
    wav_files = [f for f in os.listdir(voices_dir) if f.endswith('.wav')]

    if not wav_files:
        logging.warning(f"No .wav files found in {voices_dir}")
        return voice_map

    logging.info("=" * 80)
    logging.info(f"Discovering voices in {voices_dir}")
    logging.info(f"Found {len(wav_files)} voice sample(s)")

    for wav_file in wav_files:
        voice_name = os.path.splitext(wav_file)[0]
        wav_path = os.path.join(voices_dir, wav_file)

        # Look for matching .txt file
        txt_file = f"{voice_name}.txt"
        txt_path = os.path.join(voices_dir, txt_file)

        ref_text = None
        if os.path.exists(txt_path):
            try:
                with open(txt_path, 'r', encoding='utf-8') as f:
                    ref_text = f.read().strip()
                logging.info(f"  [{voice_name}] Found transcript: {txt_path}")
            except Exception as e:
                logging.warning(f"  [{voice_name}] Failed to read transcript: {e}")
        else:
            logging.warning(f"  [{voice_name}] No matching .txt file found")

        if ref_text:
            voice_map[voice_name] = (wav_path, ref_text)
            logging.info(f"  [{voice_name}] ✓ Loaded")
        else:
            logging.warning(f"  [{voice_name}] ✗ Skipping (no transcript)")

    logging.info("=" * 80)
    logging.info(f"Successfully loaded {len(voice_map)} voice(s): {list(voice_map.keys())}")
    logging.info("=" * 80)

    return voice_map


def create_wav_header(sample_rate: int, num_channels: int = 1, bits_per_sample: int = 16):
    """Create WAV file header for streaming."""
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = 0  # Placeholder for streaming

    header = io.BytesIO()
    header.write(b'RIFF')
    header.write(struct.pack('<I', data_size + 36))
    header.write(b'WAVE')
    header.write(b'fmt ')
    header.write(struct.pack('<I', 16))
    header.write(struct.pack('<H', 1))
    header.write(struct.pack('<H', num_channels))
    header.write(struct.pack('<I', sample_rate))
    header.write(struct.pack('<I', byte_rate))
    header.write(struct.pack('<H', block_align))
    header.write(struct.pack('<H', bits_per_sample))
    header.write(b'data')
    header.write(struct.pack('<I', data_size))

    return header.getvalue()


LANGUAGE_CODE_MAP = {
    "en": "English",
    "fr": "French",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "de": "German",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "ar": "Arabic",
}


class TTSRequest(BaseModel):
    text: str
    language: str = "English"
    voice: Optional[str] = None
    voice_id: Optional[str] = None  # Chatterbox compatibility alias
    uid: Optional[str] = None
    denoise: Optional[bool] = False

    def model_post_init(self, __context):
        # Map voice_id -> voice for chatterbox compatibility
        if self.voice is None and self.voice_id is not None:
            self.voice = self.voice_id
        # Map language codes (e.g. "en" -> "English")
        if self.language in LANGUAGE_CODE_MAP:
            self.language = LANGUAGE_CODE_MAP[self.language]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    global model, voices

    logging.info("=" * 80)
    logging.info("Loading Faster Qwen3 TTS Model...")
    logging.info(f"Model: {MODEL_NAME}")
    logging.info(f"Device: {DEVICE}")
    logging.info(f"DType: {DTYPE}")
    logging.info(f"Attention: {ATTN_IMPLEMENTATION}")
    logging.info("=" * 80)

    from faster_qwen3_tts import FasterQwen3TTS

    start = time.time()
    model = FasterQwen3TTS.from_pretrained(
        MODEL_NAME,
        device=DEVICE,
        dtype=get_dtype(DTYPE),
        attn_implementation=ATTN_IMPLEMENTATION,
    )
    logging.info(f"Model loaded in {time.time() - start:.2f}s")

    # Discover voices
    voices = discover_voices(VOICES_DIR)

    if not voices:
        logging.warning("=" * 80)
        logging.warning("WARNING: No voices loaded!")
        logging.warning("Voice cloning is REQUIRED for generation to work.")
        logging.warning(f"Please ensure voice .wav and .txt files exist in {VOICES_DIR}")
        logging.warning("=" * 80)

    logging.info("=" * 80)
    logging.info("Faster Qwen3 TTS API Server Ready!")
    logging.info(f"Available voices: {list(voices.keys())}")
    logging.info(f"Streaming chunk size: {CHUNK_SIZE}")
    logging.info("=" * 80)

    yield

    # Cleanup
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(
    title="Faster Qwen3 TTS API",
    description="Text-to-Speech API with real-time streaming support using CUDA graphs",
    version="0.2.4",
    lifespan=lifespan
)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "device": DEVICE,
        "dtype": DTYPE,
        "streaming_enabled": True,
        "available_voices": list(voices.keys()),
        "voice_count": len(voices),
    }


@app.get("/voices")
async def list_voices(uid: Optional[str] = None):
    """List available voices."""
    # List all currently loaded voices
    system_voices = [k for k in voices.keys() if "/" not in k]

    user_voices = []
    if uid:
        user_prefix = f"{uid}/"
        user_voices = [
            {"id": k.replace(user_prefix, ""), "duration": 0.0}
            for k in voices.keys() if k.startswith(user_prefix)
        ]
        user_voices.sort(key=lambda x: x["id"])

    return {
        "system_voices": sorted(system_voices),
        "user_voices": user_voices,
    }


@app.post("/voices/upload")
async def upload_voice(
    voice_name: str = Form(...),
    uid: Optional[str] = Form(None),
    wav_file: UploadFile = File(...),
    txt_file: UploadFile = File(...),
):
    """Upload a new voice for cloning."""
    global voices

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Sanitize voice_name
    voice_name = voice_name.strip().replace("/", "_").replace("\\", "_")

    if not voice_name:
        raise HTTPException(status_code=400, detail="Voice name cannot be empty")

    # Resolve target directory
    voice_dir = os.path.join(VOICES_DIR, uid) if uid else VOICES_DIR
    os.makedirs(voice_dir, exist_ok=True)

    cache_key = f"{uid}/{voice_name}" if uid else voice_name

    try:
        start_time = time.time()

        if not wav_file.filename.endswith('.wav'):
            raise HTTPException(status_code=400, detail="Audio file must be a .wav file")

        # Save and validate WAV file
        import tempfile
        import torchaudio

        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        os.close(tmp_fd)

        try:
            content = await wav_file.read()
            with open(tmp_path, "wb") as f:
                f.write(content)

            wav, sr = torchaudio.load(tmp_path)
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)

            duration = wav.shape[1] / sr
            if duration < 3.0:
                raise HTTPException(
                    status_code=400,
                    detail=f"Audio too short ({duration:.1f}s). Minimum 3s required."
                )

            wav_path = os.path.join(voice_dir, f"{voice_name}.wav")
            torchaudio.save(wav_path, wav, sr)
            logging.info(f"Saved voice audio to {wav_path}")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        # Save transcript
        txt_path = os.path.join(voice_dir, f"{voice_name}.txt")
        txt_content = await txt_file.read()
        ref_text = txt_content.decode('utf-8').strip()

        if not ref_text:
            raise HTTPException(status_code=400, detail="Transcript cannot be empty")

        with open(txt_path, 'wb') as f:
            f.write(txt_content)
        logging.info(f"Saved transcript to {txt_path}")

        # Add to voices dictionary
        voices[cache_key] = (wav_path, ref_text)

        load_time = time.time() - start_time

        return {
            "status": "success",
            "voice_name": voice_name,
            "load_time_seconds": round(load_time, 2),
            "duration": round(duration, 1),
            "message": f"Voice '{voice_name}' uploaded successfully",
        }

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Failed to upload voice '{voice_name}': {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload voice: {str(e)}")


@app.post("/tts/stream")
async def tts_stream_http(request: TTSRequest):
    """HTTP streaming endpoint that returns audio as a streaming WAV file."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Get voice
    voice_name = request.voice
    if not voice_name:
        if not voices:
            raise HTTPException(status_code=503, detail="No voices available")
        voice_name = list(voices.keys())[0]

    # Check uid-specific voice first
    cache_key = f"{request.uid}/{voice_name}" if request.uid else voice_name
    if cache_key not in voices and voice_name not in voices:
        raise HTTPException(status_code=404, detail=f"Voice '{voice_name}' not found")

    ref_audio, ref_text = voices.get(cache_key, voices.get(voice_name))

    logging.info(f"HTTP Stream: text='{request.text[:50]}...', language={request.language}, voice={voice_name}")

    async def generate_audio() -> AsyncIterator[bytes]:
        """Generate audio chunks and yield as bytes."""
        try:
            sample_rate = 24000
            header_sent = False

            for chunk, sr, timing in model.generate_voice_clone_streaming(
                text=request.text,
                language=request.language,
                ref_audio=ref_audio,
                ref_text=ref_text,
                chunk_size=CHUNK_SIZE,
                xvec_only=False,
            ):
                sample_rate = sr

                if not header_sent:
                    yield create_wav_header(sample_rate)
                    header_sent = True

                pcm_data = (chunk * 32767).astype(np.int16)
                yield pcm_data.tobytes()

            logging.info(f"HTTP streaming complete")

        except Exception as e:
            logging.error(f"Error in HTTP streaming: {e}")
            raise

    return StreamingResponse(
        generate_audio(),
        media_type="audio/wav",
        headers={
            "Content-Disposition": f"attachment; filename=tts_output.wav",
            "X-Sample-Rate": "24000",
        }
    )


@app.websocket("/tts/ws")
async def tts_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time TTS streaming."""
    await websocket.accept()
    logging.info("WebSocket connection established")

    try:
        while True:
            data = await websocket.receive_text()
            request_data = json.loads(data)

            text = request_data.get("text", "")
            language = request_data.get("language", "English")
            voice_name = request_data.get("voice", None)

            if not text:
                await websocket.send_json({
                    "type": "error",
                    "message": "No text provided"
                })
                continue

            # Get voice
            if not voice_name:
                if not voices:
                    await websocket.send_json({
                        "type": "error",
                        "message": "No voices available"
                    })
                    continue
                voice_name = list(voices.keys())[0]

            if voice_name not in voices:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Voice '{voice_name}' not found"
                })
                continue

            ref_audio, ref_text = voices[voice_name]

            logging.info(f"WebSocket: text='{text[:50]}...', language={language}, voice={voice_name}")

            # Send start message
            await websocket.send_json({
                "type": "start",
                "sample_rate": 24000,
                "chunk_size": CHUNK_SIZE,
                "voice": voice_name,
            })

            # Generate and stream audio
            try:
                chunk_count = 0
                start_time = time.time()

                for chunk, sample_rate, timing in model.generate_voice_clone_streaming(
                    text=text,
                    language=language,
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                    chunk_size=CHUNK_SIZE,
                    xvec_only=False,
                ):
                    chunk_count += 1
                    pcm_data = (chunk * 32767).astype(np.int16)

                    await websocket.send_json({
                        "type": "audio",
                        "data": pcm_data.tolist(),
                        "sample_rate": sample_rate,
                        "chunk_index": chunk_count,
                    })

                total_time = time.time() - start_time

                await websocket.send_json({
                    "type": "end",
                    "total_chunks": chunk_count,
                    "total_time_seconds": total_time,
                })

                logging.info(f"WebSocket streaming complete: {chunk_count} chunks in {total_time:.2f}s")

            except Exception as e:
                logging.error(f"Error during streaming: {e}")
                await websocket.send_json({
                    "type": "error",
                    "message": str(e)
                })

    except WebSocketDisconnect:
        logging.info("WebSocket connection closed")
    except Exception as e:
        logging.error(f"WebSocket error: {e}")
        try:
            await websocket.close()
        except:
            pass


@app.post("/tts")
async def tts_generate(request: TTSRequest):
    """Non-streaming TTS endpoint."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text required")

    if not request.voice:
        raise HTTPException(status_code=400, detail="Voice required")

    # Check uid-specific voice first
    cache_key = f"{request.uid}/{request.voice}" if request.uid else request.voice
    if cache_key not in voices and request.voice not in voices:
        raise HTTPException(status_code=404, detail=f"Voice '{request.voice}' not found")

    ref_audio, ref_text = voices.get(cache_key, voices.get(request.voice))

    try:
        start_time = time.time()

        wavs, sample_rate = model.generate_voice_clone(
            text=request.text,
            language=request.language,
            ref_audio=ref_audio,
            ref_text=ref_text,
            xvec_only=False,
        )

        generation_time = round(time.time() - start_time, 2)

        wav_np = wavs[0].astype(np.float32)
        audio_duration = round(len(wav_np) / sample_rate, 2)
        pcm_bytes = wav_np.tobytes()

        logging.info(f"Generated {audio_duration}s audio in {generation_time}s")

        headers = {
            "X-Audio-Duration": str(audio_duration),
            "X-Generation-Time": str(generation_time),
            "X-Sample-Rate": str(sample_rate),
            "X-Channels": "1",
            "X-Format": "float32",
        }
        return Response(content=pcm_bytes, media_type="application/octet-stream", headers=headers)

    except Exception as e:
        logging.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")
    finally:
        torch.cuda.empty_cache()
