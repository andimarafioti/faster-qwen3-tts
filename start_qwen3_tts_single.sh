#!/bin/bash
# Start a single-GPU non-Ray Qwen3-TTS service for an Edge backup host.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

SERVICE_NAME="${QWEN_TTS_SERVICE_NAME:-qwen3-tts-single}"
PYTHON_BIN="${QWEN_TTS_PYTHON:-$ROOT_DIR/.venv/bin/python}"
GPU_LIST="${CUDA_VISIBLE_DEVICES:-0}"
MODEL_PATH="${QWEN_TTS_MODEL:-/home/admin/models/Qwen3-TTS-12Hz-1.7B-CustomVoice}"
HOST="${QWEN_TTS_HOST:-0.0.0.0}"
PORT="${QWEN_TTS_PORT:-8091}"
DEVICE="${QWEN_TTS_DEVICE:-cuda:0}"
ATTN="${QWEN_TTS_ATTN:-sdpa}"
DTYPE="${QWEN_TTS_DTYPE:-float16}"
LANGUAGE="${QWEN_TTS_LANGUAGE:-Chinese}"
SPEAKER="${QWEN_TTS_SPEAKER:-Serena}"
CHUNK_SIZE="${QWEN_TTS_CHUNK_SIZE:-8}"
MAX_NEW_TOKENS="${QWEN_TTS_MAX_NEW_TOKENS:-512}"

if [ ! -x "$PYTHON_BIN" ]; then
    echo "ERROR: Python not found or not executable: $PYTHON_BIN" >&2
    exit 1
fi

if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: model directory not found: $MODEL_PATH" >&2
    exit 1
fi

echo "=== Starting Faster Qwen3-TTS single-GPU service ==="
echo "service: $SERVICE_NAME"
echo "model:   $MODEL_PATH"
echo "gpu:     $GPU_LIST"
echo "api:     http://127.0.0.1:$PORT"

if systemctl --user list-units --type=service --all --no-legend | grep -q "^  *${SERVICE_NAME}\\.service"; then
    echo "Stopping existing user service: ${SERVICE_NAME}.service"
    systemctl --user stop "${SERVICE_NAME}.service" || true
    systemctl --user reset-failed "${SERVICE_NAME}.service" || true
fi

systemd-run --user \
    --unit="$SERVICE_NAME" \
    --property="Restart=on-failure" \
    --property="RestartSec=5" \
    --working-directory="$ROOT_DIR" \
    /usr/bin/env CUDA_VISIBLE_DEVICES="$GPU_LIST" \
    "$PYTHON_BIN" "$ROOT_DIR/examples/single_gpu_custom_voice_server.py" \
    --model "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --device "$DEVICE" \
    --attn "$ATTN" \
    --dtype "$DTYPE" \
    --language "$LANGUAGE" \
    --speaker "$SPEAKER" \
    --chunk-size "$CHUNK_SIZE" \
    --max-new-tokens "$MAX_NEW_TOKENS"

echo "Waiting for service health..."
for _ in $(seq 1 60); do
    if curl -fsS "http://127.0.0.1:$PORT/health" >/tmp/qwen3_tts_single_health.json 2>/dev/null; then
        cat /tmp/qwen3_tts_single_health.json
        echo
        echo "Ready."
        exit 0
    fi
    sleep 2
done

echo "ERROR: service did not become healthy. Recent logs:" >&2
journalctl --user -u "${SERVICE_NAME}.service" -n 80 --no-pager >&2 || true
exit 1
