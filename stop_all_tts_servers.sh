#!/usr/bin/env bash
# Stop every faster-qwen3-tts server: the systemd-managed stack plus any
# stray processes started manually (e.g. via launch_tts_server.sh).
set -uo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

SERVER_PATTERNS=(
    "examples/openai_server.py"
    "examples/chatterbox_turbo_server.py"
    "examples/read_aloud_server.py"
    "examples/tts_compare_server.py"
    "demo/server.py"
)
KNOWN_PORTS=(7850 7860 7861 7862 8000)

# 1. Stop the systemd-user stack, if present.
if command -v systemctl >/dev/null 2>&1 && systemctl --user list-units --all --no-legend 2>/dev/null | grep -q "tts-stack.target"; then
    echo "Stopping tts-stack.target (systemd --user)..."
    systemctl --user stop tts-stack.target
else
    echo "tts-stack.target not found, skipping systemd stop."
fi

# 2. Kill any remaining server processes by script path, in case they were
#    launched outside systemd (e.g. via launch_tts_server.sh).
found_any=0
for pattern in "${SERVER_PATTERNS[@]}"; do
    pids="$(pgrep -f "${PROJECT_DIR}/${pattern}" 2>/dev/null || true)"
    if [[ -n "${pids}" ]]; then
        found_any=1
        echo "Stopping ${pattern} (pid: ${pids//$'\n'/, })"
        pkill -TERM -f "${PROJECT_DIR}/${pattern}"
    fi
done

if [[ "${found_any}" -eq 1 ]]; then
    sleep 2
    for pattern in "${SERVER_PATTERNS[@]}"; do
        pkill -KILL -f "${PROJECT_DIR}/${pattern}" 2>/dev/null || true
    done
fi

# 3. Fallback: anything still bound to the known TTS ports.
for port in "${KNOWN_PORTS[@]}"; do
    pids="$(lsof -t -i TCP:"${port}" -sTCP:LISTEN 2>/dev/null || true)"
    if [[ -n "${pids}" ]]; then
        echo "Port ${port} still in use, killing pid(s): ${pids//$'\n'/, }"
        kill -TERM ${pids} 2>/dev/null || true
        sleep 1
        kill -KILL ${pids} 2>/dev/null || true
    fi
done

echo "Done."
