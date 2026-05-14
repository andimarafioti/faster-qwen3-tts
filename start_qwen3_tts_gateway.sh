#!/bin/bash
# Start the stable external Qwen3-TTS failover gateway on the public API port.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

SERVICE_NAME="${QWEN_TTS_GATEWAY_SERVICE_NAME:-qwen3-tts-gateway}"
PYTHON_BIN="${QWEN_TTS_GATEWAY_PYTHON:-/home/ivan/.venvs/qwen3-tts-ray/bin/python}"
HOST="${QWEN_TTS_GATEWAY_HOST:-0.0.0.0}"
PORT="${QWEN_TTS_GATEWAY_PORT:-8091}"
PRIMARY_URL="${QWEN_TTS_PRIMARY_URL:-http://127.0.0.1:8092}"
BACKUP_URLS="${QWEN_TTS_BACKUP_URLS:-http://192.168.31.72:8091,http://192.168.31.74:8091,http://edge.taild500c8.ts.net:8091}"
PRIMARY_TIMEOUT_S="${QWEN_TTS_PRIMARY_TIMEOUT_S:-20}"
BACKUP_TIMEOUT_S="${QWEN_TTS_BACKUP_TIMEOUT_S:-120}"
HEALTH_TIMEOUT_S="${QWEN_TTS_HEALTH_TIMEOUT_S:-2}"
CIRCUIT_BREAK_S="${QWEN_TTS_CIRCUIT_BREAK_S:-60}"

if [ ! -x "$PYTHON_BIN" ]; then
    echo "ERROR: Python not found or not executable: $PYTHON_BIN" >&2
    exit 1
fi

echo "=== Starting Qwen3-TTS failover gateway ==="
echo "service:  $SERVICE_NAME"
echo "listen:   http://127.0.0.1:$PORT"
echo "primary:  $PRIMARY_URL"
echo "backups:  $BACKUP_URLS"

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
    "$PYTHON_BIN" "$ROOT_DIR/examples/tts_failover_gateway.py" \
    --host "$HOST" \
    --port "$PORT" \
    --primary-url "$PRIMARY_URL" \
    --backup-urls "$BACKUP_URLS" \
    --primary-timeout-s "$PRIMARY_TIMEOUT_S" \
    --backup-timeout-s "$BACKUP_TIMEOUT_S" \
    --health-timeout-s "$HEALTH_TIMEOUT_S" \
    --circuit-break-s "$CIRCUIT_BREAK_S"

echo "Waiting for gateway health..."
for _ in $(seq 1 30); do
    if curl -fsS "http://127.0.0.1:$PORT/health" >/tmp/qwen3_tts_gateway_health.json 2>/dev/null; then
        cat /tmp/qwen3_tts_gateway_health.json
        echo
        echo "Ready."
        exit 0
    fi
    sleep 1
done

echo "ERROR: gateway did not become healthy. Recent logs:" >&2
journalctl --user -u "${SERVICE_NAME}.service" -n 80 --no-pager >&2 || true
exit 1
