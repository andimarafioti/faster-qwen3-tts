#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${TTS_PYTHON:-${PROJECT_DIR}/.venv/bin/python}"
SERVER_SCRIPT="${PROJECT_DIR}/examples/openai_server.py"

MODEL="${QWEN_TTS_MODEL:-Qwen/Qwen3-TTS-12Hz-1.7B-Base}"
HOST="${QWEN_TTS_HOST:-0.0.0.0}"
PORT="${QWEN_TTS_PORT:-8000}"
DEVICE="${QWEN_TTS_DEVICE:-cuda}"
REF_AUDIO="${QWEN_TTS_REF_AUDIO:-${PROJECT_DIR}/ref_audio.wav}"
REF_TEXT="${QWEN_TTS_REF_TEXT:-I'm confused why some people have super short timelines, yet at the same time are bullish on scaling up reinforcement learning atop LLMs. If we're actually close to a human-like learner, then this whole approach of training on verifiable outcomes is doomed.}"
LANGUAGE="${QWEN_TTS_LANGUAGE:-English}"
VOICES_FILE="${QWEN_TTS_VOICES:-}"
NOTES_API_BASE="${NOTES_API_BASE:-http://localhost:9999}"
NOTES_CHANNEL="${NOTES_CHANNEL:-app}"

post_note() {
    local message="$1"
    curl -fsS --max-time 2 \
        -H 'Content-Type: application/json' \
        -d "$(printf '{\"channel\":\"%s\",\"text\":\"%s\"}' \
            "${NOTES_CHANNEL}" "${message}")" \
        "${NOTES_API_BASE}/note" >/dev/null 2>&1 || true
}

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Error: Python environment not found at ${PYTHON_BIN}" >&2
    echo "Run ./setup.sh first, or set TTS_PYTHON=/path/to/python." >&2
    exit 1
fi

if [[ ! -f "${SERVER_SCRIPT}" ]]; then
    echo "Error: server not found at ${SERVER_SCRIPT}" >&2
    exit 1
fi

server_args=(
    "${SERVER_SCRIPT}"
    --model "${MODEL}"
    --host "${HOST}"
    --port "${PORT}"
    --device "${DEVICE}"
)

if [[ -n "${VOICES_FILE}" ]]; then
    if [[ ! -f "${VOICES_FILE}" ]]; then
        echo "Error: voices configuration not found: ${VOICES_FILE}" >&2
        exit 1
    fi
    server_args+=(--voices "${VOICES_FILE}")
else
    if [[ ! -f "${REF_AUDIO}" ]]; then
        echo "Error: reference audio not found: ${REF_AUDIO}" >&2
        exit 1
    fi
    server_args+=(
        --ref-audio "${REF_AUDIO}"
        --ref-text "${REF_TEXT}"
        --language "${LANGUAGE}"
    )
fi

# Additional arguments override matching argparse options above.
server_args+=("$@")

echo "Starting Qwen3-TTS API at http://${HOST}:${PORT}"
echo "Model: ${MODEL}"
post_note "[faster-qwen3-tts/server] loading -- ${MODEL} on ${DEVICE}; API target :${PORT}"

child_pid=""
stop_server() {
    if [[ -n "${child_pid}" ]] && kill -0 "${child_pid}" 2>/dev/null; then
        kill -TERM "${child_pid}" 2>/dev/null || true
        wait "${child_pid}" 2>/dev/null || true
    fi
}
trap stop_server INT TERM

"${PYTHON_BIN}" "${server_args[@]}" &
child_pid=$!

set +e
wait "${child_pid}"
exit_code=$?
set -e
child_pid=""

if [[ ${exit_code} -eq 0 || ${exit_code} -eq 143 || ${exit_code} -eq 130 ]]; then
    post_note "[faster-qwen3-tts/server] stopped -- API on :${PORT} exited cleanly"
else
    post_note "[faster-qwen3-tts/server] failed -- server exited with status ${exit_code}"
fi

exit "${exit_code}"
