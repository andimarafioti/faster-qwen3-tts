# Hugging Face Inference Endpoint

This repo now includes a dedicated deployment path for Hugging Face Inference Endpoints:

- [`Dockerfile.hf`](../Dockerfile.hf): custom GPU container that serves [`examples/openai_server.py`](../examples/openai_server.py) on port `80`
- [`examples/create_hf_endpoint.py`](../examples/create_hf_endpoint.py): helper to create an endpoint with sensible replica bounds and container env wiring

## 1. Build and publish the image

Build the container from the repo root and push it to a registry visible to Hugging Face, for example GHCR:

```bash
docker build -f Dockerfile.hf -t ghcr.io/<org>/faster-qwen3-tts:hf .
docker push ghcr.io/<org>/faster-qwen3-tts:hf
```

## 2. Prepare the model repository

The endpoint mounts the selected model repository at `/repository`.

Recommended layout:

```text
/repository
  voices.json
  ref_audio.wav
  optional_other_voice.wav
  model files...
```

Example `voices.json`:

```json
{
  "alloy": {
    "ref_audio": "ref_audio.wav",
    "ref_text": "Reference transcript here.",
    "language": "English",
    "chunk_size": 12
  }
}
```

Relative `ref_audio` paths are resolved relative to `voices.json`, and then against `/repository`.

## 3. Create the endpoint

Use the helper script:

```bash
python3 examples/create_hf_endpoint.py \
  --name faster-qwen3-tts-prod \
  --repository your-org/your-qwen3-tts-repo \
  --image ghcr.io/<org>/faster-qwen3-tts:hf \
  --instance-type nvidia-a10g \
  --instance-size x1 \
  --min-replica 1 \
  --max-replica 6 \
  --voices voices.json \
  --wait
```

Defaults are intentionally simple:

- minimum replicas: `1`
- maximum replicas: `6`

The current public `huggingface_hub` Python API exposes replica bounds (`min_replica`, `max_replica`) but not the lower-level autoscaling metric/threshold knobs shown in the HF UI. If you want `pendingRequests`-based threshold tuning, create the endpoint with the helper first and then adjust that policy in the Hugging Face UI.

## Queueing behavior on one replica

[`examples/openai_server.py`](../examples/openai_server.py) now uses a bounded FIFO queue:

- one request performs inference at a time
- later requests wait in order
- requests are rejected with `429` if the queue is full
- `/health` exposes `queue_depth`, `waiting_requests`, and `max_pending_requests`

Environment variables:

- `QWEN_TTS_MODEL`: defaults to `/repository` when it exists
- `QWEN_TTS_VOICES`: path to `voices.json`
- `QWEN_TTS_REF_AUDIO`, `QWEN_TTS_REF_TEXT`, `QWEN_TTS_LANGUAGE`: single-voice mode
- `QWEN_TTS_MAX_PENDING`: total active + queued requests per replica
- `QWEN_TTS_CHUNK_SIZE`: default streaming chunk size

Tested with `huggingface_hub>=0.36.2`.

## Operational notes

- Keep `min_replica=1` if you want predictable latency. Scale-to-zero is possible, but cold starts will dominate TTFA.
- Keep one Uvicorn worker per replica. The queue is process-local by design.
- Streaming requests hold the replica until the stream finishes, so autoscaling based on `pendingRequests` is the right metric for this workload.
