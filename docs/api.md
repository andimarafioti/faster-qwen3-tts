# Qwen3-TTS HTTP API

This document describes the HTTP surfaces used by remote clients such as MiniCPM Desk Pet. The key boundary is simple: the TTS service synthesizes audio, while remote clients play audio on their own machine.

## Health

### `GET /health`

Checks the failover gateway and backend reachability.

```bash
curl http://127.0.0.1:8091/health
```

The response includes gateway status, primary availability, router state, and backend rows.

### `GET /api/status`

Returns worker status from the active TTS backend when routed through the gateway. Some GUI proxies may expose the same backend status as `/api/ttshealth`.

## Planning Long Text

### `POST /api/tts/plan`

Splits text into playable chunks before synthesis. Remote clients should call this first for long replies so playback can begin after the first chunk is synthesized.

Request:

```json
{
  "text": "Long text to speak",
  "trace_id": "client-optional-id",
  "max_chars_per_chunk": 80,
  "lang_hint": "Auto"
}
```

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

## Synthesis

### `POST /api/tts/speak`

Synthesizes one text chunk and returns `audio/wav`. This is the recommended endpoint for remote clients that need sound to come from the client machine.

Request:

```json
{
  "text": "Text chunk to speak",
  "speaker": "aiden",
  "speed": 1.0,
  "language": "Auto",
  "instruction": null,
  "max_new_tokens": 128
}
```

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

### Recommended Remote Client Flow

```text
model reply text
-> POST /api/tts/plan
-> POST /api/tts/speak for chunk 0 and prefetch chunk 1/2
-> play chunk 0 locally as soon as it returns
-> continue speaking chunks in order while prefetching later chunks
```

For dual-worker deployments, a client-side synthesis concurrency of 2 and a prefetch window of 3 is a good default. If worker status is unavailable, use concurrency 1.

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

### `POST /api/tts/play/stop`

Stops the current server-side playback job if one is active.

## Compatibility Notes

- `/api/tts/speak_json` returns JSON metadata and can include base64 audio, but `/api/tts/speak` is preferred for remote playback because it avoids base64 overhead.
- `/v1/audio/speech` is OpenAI-style WAV synthesis and remains available for compatible clients.
- Long client timeouts are expected. Use about 15 seconds for planning and up to 180 seconds for synthesis of a large chunk.
