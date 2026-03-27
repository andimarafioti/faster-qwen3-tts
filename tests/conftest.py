import sys
import os
import importlib.util
import tempfile
from contextlib import asynccontextmanager
from unittest.mock import MagicMock

import pytest

# ── Mock heavy modules before importing app ──────────────────────────
# app.py top-level imports these, which require CUDA or aren't available.
# faster_qwen3_tts is NOT mocked here (the real package loads fine in CI)
# (numpy/torch/transformers available), and the lifespan that calls
# FasterQwen3TTS.from_pretrained() is replaced by mock_lifespan in tests.
for mod in [
    "init_voices",
    "torchaudio",
]:
    sys.modules.setdefault(mod, MagicMock())

# GCS mock: link google.cloud.storage so `from google.cloud import storage`
# resolves to the same object as sys.modules["google.cloud.storage"]
_mock_gcs_storage = MagicMock()
mock_google_cloud = MagicMock()
mock_google_cloud.storage = _mock_gcs_storage
sys.modules.setdefault("google", MagicMock())
sys.modules["google.cloud"] = mock_google_cloud
sys.modules["google.cloud.storage"] = _mock_gcs_storage

# Set VOICES_DIR to a temp dir before importing app (it does mkdir at module level)
_test_voices_dir = tempfile.mkdtemp(prefix="tts_test_voices_")
os.environ["VOICES_DIR"] = _test_voices_dir
os.environ.pop("VOICE_CACHE_BUCKET", None)
os.environ.pop("VOICE_CACHE_PREFIX", None)

import torch
sys.modules['torchaudio'].load.return_value = (torch.zeros(1, 96000), 24000)
sys.modules["torchaudio"].__spec__ = importlib.util.spec_from_loader("torchaudio", loader=None)
import app as app_module
app_module.VOICES_DIR = _test_voices_dir
from helpers import make_wav_bytes


# ── Fixtures ────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _reset_voices():
    """Resets voices and model_ready between tests."""
    app_module.voices = {}
    app_module.model_ready = False
    yield
    app_module.voices = {}
    app_module.model_ready = False


@pytest.fixture(autouse=True)
def _reset_gcs_mock():
    """Reset GCS mock state between tests."""
    yield
    _mock_gcs_storage.Client.reset_mock(side_effect=True, return_value=True)


@pytest.fixture
def client():
    from starlette.testclient import TestClient

    @asynccontextmanager
    async def mock_lifespan(app):
        app_module.model = MagicMock()
        app_module.voices = {"test_voice": ("/fake/voice.wav", "Hello world.", {"fake": "vcp"})}
        app_module.model_ready = True
        yield

    original = app_module.app.router.lifespan_context
    app_module.app.router.lifespan_context = mock_lifespan
    with TestClient(app_module.app) as c:
        yield c
    app_module.app.router.lifespan_context = original


@pytest.fixture
def gcs_env(monkeypatch):
    monkeypatch.setenv("VOICE_CACHE_BUCKET", "test-bucket")
    monkeypatch.setenv("VOICE_CACHE_PREFIX", "voices")
    monkeypatch.setattr(app_module, "VOICE_CACHE_BUCKET", "test-bucket")
    monkeypatch.setattr(app_module, "VOICE_CACHE_PREFIX", "voices")
    monkeypatch.setattr(app_module, "_gcs_bucket", None)

# ── WAV and Text Fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def wav_4s():
    """4-second WAV file bytes (above 3s minimum)."""
    return make_wav_bytes(duration_seconds=4.0)


@pytest.fixture
def wav_1s():
    """1-second WAV file bytes (below 3s minimum)."""
    return make_wav_bytes(duration_seconds=1.0)


@pytest.fixture
def txt_file():
    return b"Hello this is a reference transcript."
