#!/bin/bash
# Install and enable a persistent user service for the Qwen3-TTS failover gateway.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
USER_SYSTEMD_DIR="${HOME}/.config/systemd/user"
SERVICE_NAME="${QWEN_TTS_GATEWAY_SERVICE_NAME:-qwen3-tts-gateway}"
SERVICE_FILE="${USER_SYSTEMD_DIR}/${SERVICE_NAME}.service"

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

mkdir -p "$USER_SYSTEMD_DIR"

cat >"$SERVICE_FILE" <<EOF
[Unit]
Description=Qwen3-TTS failover gateway
After=network-online.target

[Service]
Type=simple
WorkingDirectory=${ROOT_DIR}
ExecStart=${PYTHON_BIN} ${ROOT_DIR}/examples/tts_failover_gateway.py --host ${HOST} --port ${PORT} --primary-url ${PRIMARY_URL} --backup-urls ${BACKUP_URLS} --primary-timeout-s ${PRIMARY_TIMEOUT_S} --backup-timeout-s ${BACKUP_TIMEOUT_S} --health-timeout-s ${HEALTH_TIMEOUT_S} --circuit-break-s ${CIRCUIT_BREAK_S}
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
systemctl --user enable --now "${SERVICE_NAME}.service"

echo "Installed and started: ${SERVICE_NAME}.service"
systemctl --user status "${SERVICE_NAME}.service" --no-pager -l | sed -n '1,80p'
