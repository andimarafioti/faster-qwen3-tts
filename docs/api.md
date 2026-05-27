# Qwen3-TTS HTTP API

This document is the client-facing contract for projects and AI agents that
need to call the shared TTS service. The key boundary is simple: the TTS service
synthesizes WAV audio, while remote clients play audio on their own machine.

Default local deployment:

```text
Base URL: http://127.0.0.1:8091
Audio format: 24 kHz 16-bit PCM WAV
Default speaker: Serena
Default language: Auto
```

For another machine on the LAN or Tailscale, replace the host and keep the same
port, for example `http://tts-host:8091`.

## Recommended Client Flow

Use this flow for assistant replies, chat messages, and other text that can be
long or contain Markdown/app UI noise:

```text
1. POST /api/tts/plan with the full text.
2. Read response.text and response.chunks.
3. POST /api/tts/speak for each chunk in order.
4. Play each returned WAV locally.
5. Prefetch later chunks while playing earlier chunks.
```

Recommended defaults:

- Call `/api/tts/plan` even for mixed Chinese/English text. It normalizes text,
  removes unreadable app noise, and returns chunk-level token budgets.
- For short text, if `total_chunks` is `1`, call `/api/tts/speak` once with
  `chunks[0].text` and `chunks[0].max_new_tokens`.
- Use synthesis concurrency up to the reported worker count from `/api/status`.
  The current main host usually runs 3 workers, so concurrency `2` or `3` is
  reasonable. If status is unavailable, use concurrency `1`.
- Play chunks strictly in chunk index order, even when synthesis responses
  finish out of order.
- Use about 10-15 seconds timeout for planning and up to 180 seconds for large
  synthesis chunks.

## Health

### `GET /health`

Checks the failover gateway and backend reachability.

```bash
curl http://127.0.0.1:8091/health
```

The response includes gateway status, primary availability, router state, and backend rows.

### `GET /api/status`

Returns worker status from the active TTS backend when routed through the gateway. Some GUI proxies may expose the same backend status as `/api/ttshealth`.

This endpoint is the capability-discovery entry point. Existing top-level fields
are stable; newer clients should also read `capabilities`, `client_defaults`,
`planning`, and `audio` instead of hard-coding speaker names or concurrency.

Useful fields:

```json
{
  "success": true,
  "status": "ready",
  "tts_enabled": true,
  "tts_model_loaded": true,
  "api_version": "tts-http-v1",
  "default_speaker": "serena",
  "speakers": ["aiden", "serena"],
  "workers_ready": 3,
  "capabilities": {
    "endpoints": ["/api/tts/plan", "/api/tts/speak", "/v1/audio/speech"],
    "audio_formats": ["wav"],
    "normalizer": "wetext",
    "supports_plan": true,
    "supports_max_new_tokens": true,
    "supports_trace_id": true,
    "tts_validation_enabled": true
  },
  "client_defaults": {
    "speaker": "serena",
    "language": "Auto",
    "max_chars_per_chunk": 90,
    "plan_timeout_s": 15,
    "speak_timeout_s": 180,
    "recommended_speak_concurrency": 3,
    "recommended_prefetch_chunks": 3
  },
  "planning": {
    "default_max_chars_per_chunk": 90,
    "max_chars_per_chunk_limit": 240,
    "min_chars_per_chunk": 28,
    "max_segments": 120
  },
  "audio": {
    "format": "wav",
    "content_type": "audio/wav",
    "sample_rate_hz": 24000
  },
  "validation": {
    "enabled": true,
    "headers": ["X-TTS-Validation-Id", "X-TTS-Validation-Status"],
    "result_endpoint": "/api/tts/validation/{validation_id}",
    "recent_endpoint": "/api/tts/validation/recent?limit=20"
  },
  "workers": [
    {
      "worker_id": 0,
      "status": "ready",
      "gpu_ids": ["0"],
      "default_speaker": "serena",
      "speakers": ["aiden", "serena"]
    }
  ]
}
```

## Planning Long Text

### `POST /api/tts/plan`

Splits text into playable chunks before synthesis. Remote clients should call
this first for long replies so playback can begin after the first chunk is
synthesized.

The planner also normalizes text for TTS. This is important for Markdown, app
event text, mixed Chinese/English, repeated separators, links, and image labels
such as `[Image #1]`.

Request:

```json
{
  "text": "Long text to speak",
  "trace_id": "client-optional-id",
  "max_chars_per_chunk": 80,
  "lang_hint": "Auto"
}
```

Fields:

- `text`: Required. Full text to speak.
- `trace_id`: Optional client request id. Echoed in logs and responses.
- `max_chars_per_chunk`: Optional. Default is server-side. `60-90` is a good
  range for chat playback.
- `lang_hint`: Optional. `Auto`, `Chinese`, or another model-supported language
  hint.

Response:

```json
{
  "success": true,
  "trace_id": "client-optional-id",
  "text": "normalized text",
  "chunks": [
    {
      "index": 0,
      "text": "first chunk",
      "est_chars": 11,
      "estimated_audio_s": 2.0,
      "max_new_tokens": 64
    }
  ],
  "total_chunks": 1,
  "truncated": false,
  "sanitized": false,
  "normalizer": "wetext"
}
```

`text` is the normalized text that was planned. `normalizer` is `wetext` when
the standard text normalizer handled the request, or `basic` when the service
fell back to minimal cleanup.

Response fields:

- `chunks[].text`: Send this exact text to `/api/tts/speak`.
- `chunks[].max_new_tokens`: Send this value to `/api/tts/speak` to avoid
  overlong trailing audio for short chunks.
- `sanitized`: `true` means the planner changed the input text before TTS.
- `truncated`: `true` means the planner hit the segment limit. The caller should
  decide whether to request a shorter message or speak the returned chunks only.

Example:

```bash
curl -sS http://127.0.0.1:8091/api/tts/plan \
  -H 'content-type: application/json' \
  -d '{
    "text": "[Image #1] 好像发了一个 hello。 --- 请继续。",
    "trace_id": "example-plan-1",
    "max_chars_per_chunk": 80,
    "lang_hint": "Chinese"
  }'
```

## Synthesis

### `POST /api/tts/speak`

Synthesizes one text chunk and returns `audio/wav`. This is the recommended endpoint for remote clients that need sound to come from the client machine.

Request:

```json
{
  "text": "Text chunk to speak",
  "speaker": "Serena",
  "speed": 1.0,
  "language": "Auto",
  "instruction": null,
  "max_new_tokens": 128,
  "trace_id": "client-optional-id"
}
```

Fields:

- `text`: Required. Prefer one `chunks[].text` returned by `/api/tts/plan`.
- `speaker`: Optional. Defaults to server speaker. Known CustomVoice speakers
  include `Serena`, `Aiden`, `Dylan`, `Eric`, `Ono_Anna`, `Ryan`, `Sohee`,
  `Uncle_Fu`, and `Vivian`; check `/api/status` for the live list.
- `language`: Optional. `Auto` is usually the safest choice for mixed text.
- `instruction`: Optional style instruction. Keep empty unless the model/server
  is known to honor it reliably.
- `max_new_tokens`: Optional but recommended when using `/api/tts/plan`.
- `trace_id`: Optional client request id. Returned in `X-TTS-Trace-Id` when
  present and useful for correlating plan, speak, and client playback logs.

Example:

```bash
curl -o chunk.wav \
  -H 'content-type: application/json' \
  -d '{"text":"你好，客户端播放。","language":"Auto"}' \
  http://127.0.0.1:8091/api/tts/speak
```

Clients should play the returned WAV locally, for example with `paplay`, `aplay`, `ffplay`, `afplay`, Web Audio, or a native audio API.

Useful diagnostic response headers include `X-TTS-Normalizer`,
`X-TTS-Hit-Token-Cap`, `X-TTS-Suspicious-Duration`, `X-TTS-Audio-Seconds`,
`X-TTS-Elapsed-Seconds`, and `X-TTS-RTF`.

Common headers:

```text
Content-Type: audio/wav
X-TTS-Worker-Id: 1
X-TTS-Gpu-Ids: 1
X-TTS-Speaker: serena
X-TTS-Normalizer: wetext
X-TTS-Hit-Token-Cap: false
X-TTS-Suspicious-Duration: false
X-TTS-Audio-Seconds: 3.04
X-TTS-Elapsed-Seconds: 1.55
X-TTS-RTF: 1.96
X-TTS-Trace-Id: client-optional-id
X-TTS-Validation-Id: val-1a2b3c4d
X-TTS-Validation-Status: queued
X-TTS-Backend: primary
X-TTS-Failover: false
```

If `X-TTS-Hit-Token-Cap` or `X-TTS-Suspicious-Duration` is `true`, log the
chunk text, header values, and trace id. The caller may retry with a shorter
chunk or a lower `max_new_tokens`.

## Output Validation

TTS output validation is enabled by default. It never blocks the audio response:
the service returns WAV immediately, then a background worker sends the generated
WAV to the AGX ASR service and checks for obvious problems such as empty audio,
long silence, suspicious duration, severe text mismatch, or possible truncation.

Configuration:

```text
QWEN_TTS_VALIDATION_ENABLED=1
QWEN_TTS_VALIDATION_ASR_BASE_URL=http://agx.taild500c8.ts.net:8001
QWEN_TTS_VALIDATION_MAX_WORKERS=1
QWEN_TTS_VALIDATION_QUEUE_SIZE=100
QWEN_TTS_VALIDATION_MAX_RECORDS=200
```

Set `QWEN_TTS_VALIDATION_ENABLED=0` to disable validation.

Fast check after synthesis:

```bash
curl -D /tmp/tts.headers -o /tmp/tts.wav \
  -H 'content-type: application/json' \
  -d '{"text":"这是一段验证测试。","speaker":"Serena","language":"Chinese","trace_id":"demo-1"}' \
  http://127.0.0.1:8091/api/tts/speak

VALIDATION_ID="$(tr -d '\r' < /tmp/tts.headers | awk -F': ' 'tolower($1)=="x-tts-validation-id"{print $2}')"
curl "http://127.0.0.1:8091/api/tts/validation/${VALIDATION_ID}?wait_ms=5000"
```

Recent validation results:

```bash
curl "http://127.0.0.1:8091/api/tts/validation/recent?limit=20"
```

Result shape:

```json
{
  "success": true,
  "validation_id": "val-1a2b3c4d",
  "status": "done",
  "verdict": "passed",
  "passed": true,
  "issues": [],
  "expected_text": "这是一段验证测试。",
  "asr_text": "这是一段验证测试。",
  "similarity": 0.96,
  "audio": {
    "duration_s": 2.72,
    "rms": 0.04,
    "voice_ratio": 0.88,
    "max_silence_s": 0.3
  }
}
```

`verdict` may be `passed`, `warning`, `failed`, or `skipped`. `skipped` usually
means the ASR service was unavailable or the validation queue was full; it does
not mean the original TTS request failed.

## Minimal Agent Implementation

Pseudo-code for an AI agent:

```python
import requests

BASE_URL = "http://127.0.0.1:8091"


def tts_plan(text: str) -> dict:
    response = requests.post(
        f"{BASE_URL}/api/tts/plan",
        json={
            "text": text,
            "trace_id": "agent-message-123",
            "max_chars_per_chunk": 80,
            "lang_hint": "Auto",
        },
        timeout=15,
    )
    response.raise_for_status()
    return response.json()


def tts_speak(chunk: dict) -> bytes:
    response = requests.post(
        f"{BASE_URL}/api/tts/speak",
        json={
            "text": chunk["text"],
            "speaker": "Serena",
            "language": "Auto",
            "max_new_tokens": chunk.get("max_new_tokens"),
            "trace_id": "agent-message-123",
        },
        timeout=180,
    )
    response.raise_for_status()
    return response.content
```

Playback is intentionally not part of the API. The caller should write or stream
the returned bytes to the local audio stack.

## Server-side Playback

### `POST /api/tts/play`

Synthesizes and plays audio on the TTS server host. Use this only when the speakers are attached to the TTS server. Remote desktop clients such as MiniCPM Desk Pet should not use this endpoint for normal replies.

Request:

```json
{
  "text": "Text to play on the server",
  "speaker": "aiden",
  "language": "Auto",
  "speed": 1.0,
  "mode": "replace",
  "max_chars_per_chunk": 45,
  "prefetch_window": 3,
  "retry_count": 1
}
```

This endpoint may not be available through every gateway or GUI proxy. Prefer
`/api/tts/speak` for reusable integrations.

### `POST /api/tts/play/stop`

Stops the current server-side playback job if one is active.

## Error Handling

The API uses JSON errors for non-audio failures:

```json
{
  "success": false,
  "error": "No text provided"
}
```

Typical caller behavior:

- `400`: fix the request payload.
- `503`: TTS is disabled or backend is unavailable; retry later or fall back to
  text-only output.
- `5xx`: log `trace_id`, chunk text length, response body, and retry a smaller
  chunk once if appropriate.

## Compatibility Notes

- `/api/tts/speak_json` returns JSON metadata and can include base64 audio, but `/api/tts/speak` is preferred for remote playback because it avoids base64 overhead.
- `/v1/audio/speech` is OpenAI-style WAV synthesis and remains available for compatible clients.
- Long client timeouts are expected. Use about 15 seconds for planning and up to 180 seconds for synthesis of a large chunk.
