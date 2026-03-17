"""Configuration loader and constants for Faster Qwen3 TTS API."""

import os
import yaml


def load_config() -> dict:
    """Load configuration from YAML file."""
    config_paths = [
        os.path.join("/app/config", "config.yaml"),
        os.path.join(os.path.dirname(__file__), "config.yaml"),
        "config.yaml",
    ]
    for path in config_paths:
        if os.path.exists(path):
            with open(path) as f:
                return yaml.safe_load(f)

    # Default configuration
    return {
        "model": {
            "name": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            "device": "cuda:0",
            "dtype": "bfloat16",
            "attn_implementation": "sdpa",
        },
        "voices": {
            "dir": "/app/voices",
        },
        "streaming": {
            "chunk_size": 8,
        },
    }


CONFIG = load_config()

MODEL_CONFIG = CONFIG.get("model", {})
VOICES_CONFIG = CONFIG.get("voices", {})
STREAMING_CONFIG = CONFIG.get("streaming", {})

MODEL_NAME = MODEL_CONFIG.get("name", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
DEVICE = MODEL_CONFIG.get("device", "cuda:0")
DTYPE = MODEL_CONFIG.get("dtype", "bfloat16")
ATTN_IMPLEMENTATION = MODEL_CONFIG.get("attn_implementation", "sdpa")

VOICES_DIR = VOICES_CONFIG.get("dir", "/app/voices")

CHUNK_SIZE = STREAMING_CONFIG.get("chunk_size", 8)

GCS_CONFIG = VOICES_CONFIG.get("gcs", {})
VOICE_CACHE_BUCKET = os.environ.get("VOICE_CACHE_BUCKET", GCS_CONFIG.get("bucket", ""))
VOICE_CACHE_PREFIX = os.environ.get("VOICE_CACHE_PREFIX", GCS_CONFIG.get("prefix", ""))
