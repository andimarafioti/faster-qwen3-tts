#!/usr/bin/env bash
set -euo pipefail

# Install a user systemd timer that periodically learns lowercase technical
# pronunciation terms from TTS validation candidate logs.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_NAME="${QWEN_TTS_LEARNER_SERVICE_NAME:-qwen3-tts-pronunciation-learner}"
PYTHON_BIN="${QWEN_TTS_LEARNER_PYTHON:-/home/ivan/.venvs/qwen3-tts-ray/bin/python}"
ON_CALENDAR="${QWEN_TTS_LEARNER_ON_CALENDAR:-hourly}"
CONFIG_DIR="${QWEN_TTS_CONFIG_DIR:-$HOME/.config/faster-qwen3-tts}"
SYSTEMD_DIR="$HOME/.config/systemd/user"

mkdir -p "$SYSTEMD_DIR" "$CONFIG_DIR"

cat > "$SYSTEMD_DIR/$SERVICE_NAME.service" <<EOF
[Unit]
Description=Qwen3-TTS pronunciation lexicon learner

[Service]
Type=oneshot
WorkingDirectory=$ROOT_DIR
Environment=QWEN_TTS_CONFIG_DIR=$CONFIG_DIR
ExecStart=$PYTHON_BIN $ROOT_DIR/examples/tts_pronunciation_learner.py
EOF

cat > "$SYSTEMD_DIR/$SERVICE_NAME.timer" <<EOF
[Unit]
Description=Run Qwen3-TTS pronunciation lexicon learner periodically

[Timer]
OnCalendar=$ON_CALENDAR
Persistent=true

[Install]
WantedBy=timers.target
EOF

systemctl --user daemon-reload
systemctl --user enable --now "$SERVICE_NAME.timer"
systemctl --user list-timers "$SERVICE_NAME.timer" --no-pager
