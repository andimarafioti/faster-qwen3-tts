---
title: faster-qwen3-tts
author: andito
emoji: 🎙
tags: [text-to-speech, streaming, cuda-graphs]
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
hf_oauth: true
preload_from_hub:
  - nvidia/parakeet-tdt-0.6b-v3
  - Qwen/Qwen3-TTS-12Hz-0.6B-Base
  - Qwen/Qwen3-TTS-12Hz-1.7B-Base
  - Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice
  - Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
  - Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign
---

# Faster Qwen3-TTS Demo

This Space hosts the demo UI for **faster-qwen3-tts** with streaming audio, TTFA/RTF metrics, voice clone, custom voices, and voice design.

## Run locally (no Docker)

```bash
pip install "faster-qwen3-tts[demo]"
python server.py --model Qwen/Qwen3-TTS-12Hz-0.6B-Base
# open http://localhost:7860
```

## Run with Docker

```bash
docker build -t faster-qwen3-tts-demo .
docker run --gpus all -p 7860:7860 faster-qwen3-tts-demo
```

## Auth and quotas

Local launches default to an open demo with no login requirement. The Space
Docker image sets `DEMO_WEB_ONLY=1` and `DEMO_REQUIRE_LOGIN=1`, and the Space
enables Hugging Face OAuth via `hf_oauth: true`. In that mode, users must sign
in before the app UI or API routes are usable. Generation requests are tracked
by an HMAC of the user's stable Hugging Face OAuth ID: non-PRO users get
`DEMO_DAILY_FREE_REQUESTS` generations per UTC day, while PRO users are not
limited. The raw OAuth ID is not stored. For admin visibility, the database
also keeps a `usage_users` table mapping each HMAC user key to the latest
observed Hugging Face username and PRO status.

Usage is stored in a SQLite database inside an attached Hugging Face Storage
Bucket. Create a bucket and attach it to the Space as a read-write volume at
`/data`; the app will automatically store usage at
`/data/faster-qwen3-tts-usage.sqlite3`. If the bucket is not mounted, usage
falls back to temporary storage and resets when the Space restarts.

When `DEMO_WEB_ONLY=1`, expensive POST routes require a short-lived token that
is injected into the served web UI and sent back by the browser. This blocks
casual direct API calls, but OAuth is the real identity boundary.

Useful environment variables:

```bash
DEMO_WEB_ONLY=0                  # default: disable the browser-token gate
DEMO_REQUIRE_LOGIN=0             # default locally; set to 1 to require OAuth
DEMO_WEB_TOKEN_TTL_SECONDS=7200  # token lifetime
DEMO_WEB_GATE_SECRET=...         # optional stable signing secret
DEMO_USAGE_HASH_SECRET=...       # stable Space Secret for pseudonymous quota IDs
DEMO_DAILY_FREE_REQUESTS=10      # non-PRO daily generation limit
USAGE_BUCKET_MOUNT_PATH=/data    # attached HF Storage Bucket mount path
USAGE_DB_FILENAME=faster-qwen3-tts-usage.sqlite3
USAGE_DB_PATH=/data/usage.sqlite3 # optional full path override
```

Set `DEMO_USAGE_HASH_SECRET` as a Space Secret before using the persistent
bucket. If it is not set, the app falls back to `DEMO_WEB_GATE_SECRET`; if
neither is set, quota IDs are still private but rotate on restart, resetting
per-user counts.

To inspect usage from the bucket DB:

```sql
SELECT u.username, u.is_pro, d.day, d.count
FROM usage_daily d
LEFT JOIN usage_users u USING (user_key)
ORDER BY d.day DESC, d.count DESC;
```

Bucket setup:

```bash
hf buckets create HuggingFaceM4/faster-qwen3-tts-usage
```

For the hosted `HuggingFaceM4` Space, attach the private
`HuggingFaceM4/faster-qwen3-tts-usage` bucket in the Space settings under
Storage Buckets with mount path `/data` and read-write access. The
`andito/faster-qwen3-tts-usage` bucket is only needed for a personal
`andito`-owned Space or fork.

You can also attach the org bucket with `huggingface_hub`:

```python
from huggingface_hub import HfApi, Volume

api = HfApi()
api.set_space_volumes(
    "HuggingFaceM4/faster-qwen3-tts-demo",
    volumes=[
        Volume(
            type="bucket",
            source="HuggingFaceM4/faster-qwen3-tts-usage",
            mount_path="/data",
        ),
    ],
)
api.restart_space("HuggingFaceM4/faster-qwen3-tts-demo", factory_reboot=True)
```
