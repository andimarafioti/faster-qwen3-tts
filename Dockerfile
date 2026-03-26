# ============================================================
# Dockerfile - Faster Qwen3 TTS with CUDA 12.6
# ============================================================

FROM nvidia/cuda:12.6.0-cudnn-devel-ubuntu24.04

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies and clean up in one layer to reduce image size
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3-pip python3.12-venv python3.12-dev \
    sox libsox-fmt-all \
    ffmpeg \
    git \
    wget \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/* /tmp/* /var/tmp/*

WORKDIR /app

# Remove EXTERNALLY-MANAGED to allow pip installs (safe in Docker containers)
RUN rm -f /usr/lib/python3.*/EXTERNALLY-MANAGED

# Install PyTorch 2.10.0 with CUDA 12.8 support first (matches GCP CUDA driver)
# This ensures we get the CUDA-enabled build instead of CPU-only from default PyPI
RUN pip install --no-cache-dir \
    torch==2.10.0 \
    torchaudio==2.10.0 \
    --index-url https://download.pytorch.org/whl/cu128

# Install API server dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install faster-qwen3-tts 0.2.5 from local source (has voice_clone_prompt support)
COPY pyproject.toml .
COPY faster_qwen3_tts/ /app/faster_qwen3_tts/
RUN pip install --no-cache-dir -e .

# Copy application code
COPY config.yaml .
COPY config.py .
COPY init_voices.py .
COPY otel_setup.py .
COPY app.py .

# Copy voice samples to the image
COPY voices/ /app/voices/

# Expose port for API
EXPOSE 8000

# Run the API server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

# ============================================================
# BUILD & RUN COMMANDS
# ============================================================
#
# Build:
#   docker build -t faster-qwen3-tts-api .
#
# Run:
#   docker run --gpus all -p 8000:8000 \
#     -v $(pwd)/voices:/app/voices \
#     faster-qwen3-tts-api
#
# Test:
#   curl http://localhost:8000/health
#
#   curl -X POST http://localhost:8000/tts/stream \
#     -H "Content-Type: application/json" \
#     -d '{"text": "Hello world", "language": "English", "voice": "my_voice"}' \
#     --output output.wav
#
# ============================================================
