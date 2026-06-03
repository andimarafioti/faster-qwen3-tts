# ============================================================
# Dockerfile - Faster Qwen3 TTS API (runtime image)
# ============================================================
# Inherits from Dockerfile.base which has all dependencies and
# the model weights pre-baked in. Build the base first.
#
# Build base (once):
#   docker build -f Dockerfile.base -t faster-qwen3-tts-base:latest .
#
# Build API image:
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
# ============================================================

FROM faster-qwen3-tts-base:1.0.1

# Install the faster_qwen3_tts package
COPY pyproject.toml .
COPY faster_qwen3_tts/ /app/faster_qwen3_tts/
RUN pip install --no-cache-dir -e .

# Copy application code
COPY config.yaml .
COPY config.py .
COPY init_voices.py .
COPY otel_setup.py .
COPY tts_registry.py .
COPY app.py .

# Copy voice samples to the image
COPY voices/ /app/voices/

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]