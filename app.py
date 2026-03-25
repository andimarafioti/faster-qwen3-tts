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
import json
from typing import Optional, AsyncIterator
from config import DEVICE, DTYPE, MODEL_NAME, ATTN_IMPLEMENTATION, VOICES_DIR, CHUNK_SIZE, VOICE_CACHE_BUCKET, VOICE_CACHE_PREFIX

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Global model instance
model = None
model_ready = False
# Voice metadata: maps voice name to (ref_audio_path, ref_text, vcp)
# vcp is a pre-loaded voice_clone_prompt dict (or None if not yet extracted)
voices = {}
# Cached GCS bucket client (initialized once on first use)
_gcs_bucket = None

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
        dict: Mapping of voice name to (ref_audio_path, ref_text, vcp)
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

        pt_path = os.path.join(voices_dir, f"{voice_name}.pt")
        vcp = load_voice_clone_prompt(pt_path, ref_text) if os.path.exists(pt_path) else None

        if ref_text or vcp:
            voice_map[voice_name] = (wav_path, ref_text, vcp)
            logging.info(f"  [{voice_name}] ✓ Loaded")
        else:
            logging.warning(f"  [{voice_name}] ✗ Skipping (no transcript and no .pt)")

    logging.info("=" * 80)
    logging.info(f"Successfully loaded {len(voice_map)} voice(s): {list(voice_map.keys())}")
    logging.info("=" * 80)

    return voice_map


def fetch_voice_from_gcs(voice_name: str, uid: Optional[str] = None) -> bool:
    """Download a voice from GCS and load it into the voices dict.

    Returns True if the voice was successfully fetched and loaded.
    """
    global voices
    if not VOICE_CACHE_BUCKET or not VOICE_CACHE_PREFIX:
        return False

    bucket = get_gcs_bucket()
    if bucket is None:
        return False

    try:
        gcs_prefix = f"{VOICE_CACHE_PREFIX}/{uid}" if uid else VOICE_CACHE_PREFIX

        voice_dir = os.path.join(VOICES_DIR, uid) if uid else VOICES_DIR
        os.makedirs(voice_dir, exist_ok=True)

        wav_path = os.path.join(voice_dir, f"{voice_name}.wav")
        txt_path = os.path.join(voice_dir, f"{voice_name}.txt")

        # If already on disk (e.g. from init_voices), skip GCS download for wav/txt
        if os.path.exists(wav_path):
            ref_text = None
            if os.path.exists(txt_path):
                with open(txt_path, 'r', encoding='utf-8') as f:
                    ref_text = f.read().strip()
        else:
            wav_blob = bucket.blob(f"{gcs_prefix}/{voice_name}.wav")
            if not wav_blob.exists():
                return False
            wav_blob.download_to_filename(wav_path)

            ref_text = None
            txt_blob = bucket.blob(f"{gcs_prefix}/{voice_name}.txt")
            if txt_blob.exists():
                txt_blob.download_to_filename(txt_path)
                with open(txt_path, 'r', encoding='utf-8') as f:
                    ref_text = f.read().strip()

        pt_path = os.path.join(voice_dir, f"{voice_name}.pt")
        pt_blob = bucket.blob(f"{gcs_prefix}/{voice_name}.pt")
        if os.path.exists(pt_path):
            logging.info(f"Using cached .pt embedding for voice '{voice_name}' from disk")
        elif pt_blob.exists():
            pt_blob.download_to_filename(pt_path)
            logging.info(f"Downloaded .pt embedding for voice '{voice_name}' from GCS")
        elif model is not None:
            if extract_speaker_embedding(wav_path, ref_text, pt_path):
                logging.info(f"Extracted .pt embedding for voice '{voice_name}'")
                try:
                    pt_blob.upload_from_filename(pt_path)
                    logging.info(f"Uploaded .pt embedding for voice '{voice_name}' to GCS")
                except Exception as e:
                    logging.warning(f"GCS upload of .pt failed for '{voice_name}': {e}")
            else:
                pt_path = None

        cache_key = f"{uid}/{voice_name}" if uid else voice_name
        vcp = load_voice_clone_prompt(pt_path, ref_text) if pt_path and os.path.exists(pt_path) else None
        if vcp is None and model is not None and os.path.exists(wav_path):
            if extract_speaker_embedding(wav_path, ref_text, pt_path):
                vcp = load_voice_clone_prompt(pt_path, ref_text)
                try:
                    pt_blob.upload_from_filename(pt_path)
                except Exception as e:
                    logging.warning(f"GCS upload of .pt failed for '{voice_name}': {e}")
        voices[cache_key] = (wav_path, ref_text, vcp)
        logging.info(f"Fetched voice '{cache_key}' from GCS")
        return True

    except Exception as e:
        logging.warning(f"Failed to fetch voice '{voice_name}' from GCS: {e}")
        return False

def get_gcs_bucket():
    """Return cached GCS bucket if configured, else None."""
    global _gcs_bucket
    if not VOICE_CACHE_BUCKET or not VOICE_CACHE_PREFIX:
        return None
    if _gcs_bucket is not None:
        return _gcs_bucket
    try:
        from google.cloud import storage as gcs
        _gcs_bucket = gcs.Client().bucket(VOICE_CACHE_BUCKET)
        return _gcs_bucket
    except Exception as e:
        logging.warning(f"GCS client init failed: {e}")
        return None


def extract_speaker_embedding(wav_path: str, ref_text: str, pt_out: str) -> bool:
    """Extract speaker embedding from wav and save to pt_out. Returns True on success."""
    try:
        prompt_items = model.model.create_voice_clone_prompt(
            ref_audio=wav_path,
            ref_text=ref_text or "",
            x_vector_only_mode=not bool(ref_text),
        )
        torch.save({
            'ref_spk_embedding': prompt_items[0].ref_spk_embedding.cpu(),
            'ref_code': prompt_items[0].ref_code,
        }, pt_out)
        return True
    except Exception as e:
        logging.warning(f"Failed to extract embedding: {e}")
        return False


def resolve_voice(voice_name: str, uid: Optional[str] = None):
    """Resolve a voice name to (ref_audio, ref_text, vcp). Raises HTTPException on failure."""
    cache_key = f"{uid}/{voice_name}" if uid else voice_name
    if cache_key not in voices and voice_name not in voices:
        if not fetch_voice_from_gcs(voice_name, uid=uid):
            raise HTTPException(status_code=404, detail=f"Voice '{voice_name}' not found")
    ref_audio, ref_text, vcp = voices.get(cache_key, voices.get(voice_name))
    if not ref_audio and not ref_text and not vcp:
        raise HTTPException(status_code=500, detail=f"Voice '{voice_name}' has no usable data")
    if ref_audio and not ref_text and not vcp:
        logging.warning(f"Voice '{voice_name}' has no transcript or embedding, quality may be reduced")
    return ref_audio, ref_text, vcp


def load_voice_clone_prompt(pt_path: str, ref_text: Optional[str] = None) -> Optional[dict]:
    data = torch.load(pt_path, weights_only=True)
    if not isinstance(data, dict):
        # old format (embedding tensor only) — needs re-extraction to get ref_code
        return None
    spk_emb = data['ref_spk_embedding'].to(DEVICE)
    ref_code = data.get('ref_code')
    return dict(
        ref_code=[ref_code],
        ref_spk_embedding=[spk_emb],
        x_vector_only_mode=[ref_code is None],
    )

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
    request_id: Optional[str] = None

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
    global model, voices, model_ready

    logging.info("=" * 80)
    logging.info("Loading Faster Qwen3 TTS Model...")
    logging.info(f"Model: {MODEL_NAME}")
    logging.info(f"Device: {DEVICE}")
    logging.info(f"DType: {DTYPE}")
    logging.info(f"Attention: {ATTN_IMPLEMENTATION}")
    logging.info("=" * 80)

    # Download voices from GCS if configured
    from init_voices import init_voices
    init_voices()

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

    # Auto-extract .pt speaker embeddings for voices that don't have one yet
    gcs_bucket = get_gcs_bucket()

    for voice_name, (wav_path, ref_text, vcp) in list(voices.items()):
        if vcp is None and wav_path and os.path.exists(wav_path):
            try:
                logging.info(f"Extracting speaker embedding for '{voice_name}'...")
                pt_out = os.path.join(os.path.dirname(wav_path), f"{voice_name}.pt")
                if not extract_speaker_embedding(wav_path, ref_text, pt_out):
                    continue
                new_vcp = load_voice_clone_prompt(pt_out, ref_text)
                voices[voice_name] = (wav_path, ref_text, new_vcp)
                logging.info(f"  Saved embedding to {pt_out}")
                if gcs_bucket:
                    try:
                        gcs_bucket.blob(f"{VOICE_CACHE_PREFIX}/{voice_name}.pt").upload_from_filename(pt_out)
                        logging.info(f"  Uploaded {voice_name}.pt to GCS")
                    except Exception as e:
                        logging.warning(f"  GCS upload failed for {voice_name}.pt: {e}")
            except Exception as e:
                logging.warning(f"  Failed to extract embedding for '{voice_name}': {e}")

    if not voices:
        logging.warning("=" * 80)
        logging.warning("WARNING: No voices loaded!")
        logging.warning("Voice cloning is REQUIRED for generation to work.")
        logging.warning(f"Please ensure voice .wav and .txt files exist in {VOICES_DIR}")
        logging.warning("=" * 80)

    # Warmup: run one inference to trigger CUDA graph capture/JIT compilation
    # so the first real request isn't slow.
    if voices:
        warmup_voice = list(voices.keys())[0]
        ref_audio, ref_text, vcp = voices[warmup_voice]
        logging.info(f"Warming up model with voice '{warmup_voice}'...")
        warmup_start = time.time()
        try:
            for _ in model.generate_voice_clone_streaming(
                text="Hello.",
                language="English",
                ref_audio=ref_audio if not vcp else None,
                ref_text=ref_text,
                voice_clone_prompt=vcp,
                chunk_size=CHUNK_SIZE,
                xvec_only=False,
            ):
                pass
            logging.info(f"Warmup complete in {time.time() - warmup_start:.2f}s")
            model_ready = True
        except Exception as e:
            logging.warning(f"Warmup failed (non-fatal): {e}")
            model_ready = True

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
    if not model_ready:
        raise HTTPException(status_code=503, detail="Model not ready")
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
    """List available voices. Checks GCS for user voices if configured."""
    # List all currently loaded system voices
    system_voices = [k for k in voices.keys() if "/" not in k]

    user_voices = []
    if uid:
        # Check local memory first
        user_prefix = f"{uid}/"
        local_voices = {
            k.replace(user_prefix, "")
            for k in voices.keys() if k.startswith(user_prefix)
        }

        # Check GCS for additional user voices
        bucket = get_gcs_bucket()
        if bucket:
            try:
                gcs_prefix = f"{VOICE_CACHE_PREFIX}/{uid}/"
                blobs = bucket.list_blobs(prefix=gcs_prefix)
                for blob in blobs:
                    if blob.name.endswith('.wav'):
                        name = os.path.splitext(os.path.basename(blob.name))[0]
                        local_voices.add(name)
            except Exception as e:
                logging.warning(f"Failed to list GCS voices for uid={uid}: {e}")

        user_voices = [{"id": name, "loaded": f"{uid}/{name}" in voices, "has_embedding": voices.get(f"{uid}/{name}", (None, None, None))[2] is not None} for name in sorted(local_voices)]

    return {
        "system_voices": [{"id": k, "has_embedding": voices.get(k, (None, None, None))[2] is not None} for k in sorted(system_voices)],
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

        # Auto-extract .pt speaker embedding
        pt_path = os.path.join(voice_dir, f"{voice_name}.pt")
        if extract_speaker_embedding(wav_path, ref_text, pt_path):
            logging.info(f"Extracted speaker embedding for '{voice_name}'")
        else:
            pt_path = None

        # Add to voices dictionary
        vcp = load_voice_clone_prompt(pt_path, ref_text) if pt_path else None
        voices[cache_key] = (wav_path, ref_text, vcp)

        # Upload to GCS if configured
        bucket = get_gcs_bucket()
        if bucket:
            try:
                gcs_prefix = f"{VOICE_CACHE_PREFIX}/{uid}" if uid else VOICE_CACHE_PREFIX
                bucket.blob(f"{gcs_prefix}/{voice_name}.wav").upload_from_filename(wav_path)
                bucket.blob(f"{gcs_prefix}/{voice_name}.txt").upload_from_filename(txt_path)
                if pt_path and os.path.exists(pt_path):
                    bucket.blob(f"{gcs_prefix}/{voice_name}.pt").upload_from_filename(pt_path)
                    logging.info(f"Uploaded .pt embedding for voice '{voice_name}' to GCS")
                logging.info(f"Uploaded voice '{voice_name}' to gs://{VOICE_CACHE_BUCKET}/{gcs_prefix}/")
            except Exception as e:
                logging.warning(f"GCS upload failed for voice '{voice_name}': {e}")

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
    """HTTP streaming endpoint that returns audio as raw PCM float32 bytes."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Get voice
    voice_name = request.voice
    if not voice_name:
        if not voices:
            raise HTTPException(status_code=503, detail="No voices available")
        voice_name = list(voices.keys())[0]

    ref_audio, ref_text, vcp = resolve_voice(voice_name, uid=request.uid)

    logging.info(f"HTTP Stream: text='{request.text[:50]}...', language={request.language}, voice={voice_name}")

    stream_start_time = time.time()
    ttfa_ms = None  # Time to first audio chunk

    async def generate_audio() -> AsyncIterator[bytes]:
        """Generate audio chunks and yield as raw PCM float32 bytes."""
        try:
            first_chunk = True
            for chunk, sr, timing in model.generate_voice_clone_streaming(
                text=request.text,
                language=request.language,
                ref_audio=ref_audio if not vcp else None,
                ref_text=ref_text,
                voice_clone_prompt=vcp,
                chunk_size=CHUNK_SIZE,
                xvec_only=False,
            ):

                pcm = np.nan_to_num(chunk.astype(np.float32), nan=0.0, posinf=1.0, neginf=-1.0)
                pcm = np.clip(pcm, -1.0, 1.0)
                if first_chunk:
                    nonlocal ttfa_ms
                    ttfa_ms = (time.time() - stream_start_time) * 1000
                    logging.info(f"Time to first audio chunk: {ttfa_ms:.2f} ms")
                    first_chunk = False
                yield pcm.tobytes()

            logging.info(f"HTTP streaming complete")

        except Exception as e:
            logging.error(f"Error in HTTP streaming: {e}")
            raise

    return StreamingResponse(
        generate_audio(),
        media_type="application/octet-stream",
        headers={
            "X-Sample-Rate": "24000",
            "X-Channels": "1",
            "X-Format": "float32",
            **({"X-Request-ID": request.request_id} if request.request_id else {}),
            **({"X-TTFA-MS": str(ttfa_ms)} if ttfa_ms else {}),
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
            try:
                request = TTSRequest(**json.loads(data))
            except Exception as e:
                await websocket.send_json({"type": "error", "message": str(e)})
                continue

            text = request.text
            language = request.language
            voice_name = request.voice
            request_id = request.request_id

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

            uid = request.uid
            try:
                ref_audio, ref_text, vcp = resolve_voice(voice_name, uid=uid)
            except HTTPException as e:
                await websocket.send_json({"type": "error", "message": e.detail})
                continue

            logging.info(f"WebSocket: text='{text[:50]}...', language={language}, voice={voice_name}")

            # Send start message
            await websocket.send_json({
                "type": "start",
                "sample_rate": 24000,
                "chunk_size": CHUNK_SIZE,
                "voice": voice_name,
                "format": "float32",
                "request_id": request_id,
            })

            # Generate and stream audio
            try:
                chunk_count = 0
                start_time = time.time()

                for chunk, sample_rate, timing in model.generate_voice_clone_streaming(
                    text=text,
                    language=language,
                    ref_audio=ref_audio if not vcp else None,
                    ref_text=ref_text,
                    voice_clone_prompt=vcp,
                    chunk_size=CHUNK_SIZE,
                    xvec_only=False,
                ):
                    chunk_count += 1
                    pcm_data = np.nan_to_num(chunk.astype(np.float32), nan=0.0, posinf=1.0, neginf=-1.0)
                    pcm_data = np.clip(pcm_data, -1.0, 1.0)

                    await websocket.send_json({
                        "type": "audio",
                        "data": pcm_data.tolist(),
                        "sample_rate": sample_rate,
                        "chunk_index": chunk_count,
                    })

                total_time = time.time() - start_time

                silence = np.zeros(int(24000 * 0.3), dtype=np.float32)
                chunk_count += 1
                await websocket.send_json({
                    "type": "audio",
                    "data": silence.tolist(),
                    "sample_rate": 24000,
                    "chunk_index": chunk_count,
                })

                await websocket.send_json({
                    "type": "end",
                    "total_chunks": chunk_count,
                    "total_time_seconds": total_time,
                    "request_id": request_id,
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

    ref_audio, ref_text, vcp = resolve_voice(request.voice, uid=request.uid)

    logging.info(f"HTTP Generate: text='{request.text[:50]}...', language={request.language}, voice={request.voice}")

    try:
        start_time = time.time()

        wavs, sample_rate = model.generate_voice_clone(
            text=request.text,
            language=request.language,
            ref_audio=ref_audio if not vcp else None,
            ref_text=ref_text,
            voice_clone_prompt=vcp,
            xvec_only=False,
        )

        generation_time = round(time.time() - start_time, 2)

        wav_np = wavs[0].astype(np.float32)
        audio_duration = round(len(wav_np) / sample_rate, 2)
        pcm_bytes = wav_np.tobytes()

        rtf = round(audio_duration / generation_time, 3) if generation_time > 0 else 0.0
        logging.info(f"Generated {audio_duration}s audio in {generation_time}s (RTF={rtf})")

        headers = {
            "X-Audio-Duration": str(audio_duration),
            "X-Generation-Time": str(generation_time),
            "X-RTF": str(rtf),
            "X-Sample-Rate": str(sample_rate),
            "X-Channels": "1",
            "X-Format": "float32",
            **({"X-Request-ID": request.request_id} if request.request_id else {}),
        }
        return Response(content=pcm_bytes, media_type="application/octet-stream", headers=headers)

    except Exception as e:
        logging.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")
    finally:
        torch.cuda.empty_cache()
