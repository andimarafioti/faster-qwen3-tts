import sys
from unittest.mock import MagicMock, patch
import torch

_mock_storage = sys.modules["google.cloud.storage"]


# ── Validation ──────────────────────────────────────────────────────────

def test_missing_all_fields(client):
    r = client.post("/voices/upload")
    assert r.status_code == 422


def test_missing_wav_file(client, txt_file):
    r = client.post(
        "/voices/upload",
        data={"voice_name": "test", "uid": "user1"},
        files={"txt_file": ("test.txt", txt_file, "text/plain")},
    )
    assert r.status_code == 422

def test_missing_txt_file(client, wav_4s):
    r = client.post(
        "/voices/upload",
        data={"voice_name": "test", "uid": "user1"},
        files={"wav_file": ("test.wav", wav_4s, "audio/wav")},
    )
    assert r.status_code == 422

def test_missing_voice_name(client, wav_4s, txt_file):
    r = client.post(
        "/voices/upload",
        data={"uid": "user1"},
        files={
            "wav_file": ("test.wav", wav_4s, "audio/wav"),
            "txt_file": ("test.txt", txt_file, "text/plain")
        },
    )
    assert r.status_code == 422


def test_empty_voice_name(client, wav_4s, txt_file):
    r = client.post(
        "/voices/upload",
        data={"voice_name": " ", "uid": "user1"},
        files={
            "wav_file": ("test.wav", wav_4s, "audio/wav"),
            "txt_file": ("test.txt", txt_file, "text/plain")
        },
    )
    assert r.status_code == 400
    assert "voice name cannot be empty" in r.json()["detail"].lower()


def test_audio_too_short(client, wav_1s, txt_file):
    with patch("torchaudio.load", return_value=(torch.zeros(1, 12000), 24000)):
        r = client.post(
            "/voices/upload",
            data={"voice_name": "myvoice", "uid": "user1"},
            files={
                "wav_file": ("short.wav", wav_1s, "audio/wav"),
                "txt_file": ("test.txt", txt_file, "text/plain"),
            },
        )
    assert r.status_code == 400
    assert "too short" in r.json()["detail"].lower()


def test_invalid_audio_format(client, txt_file):
    r = client.post(
        "/voices/upload",
        data={"voice_name": "myvoice", "uid": "user1"},
        files={
            "wav_file": ("myvoice.mp3", b"fake audio", "audio/mpeg"),
            "txt_file": ("myvoice.txt", txt_file, "text/plain")
        },
    )
    assert r.status_code == 400
    assert "audio file must be a .wav file" in r.json()["detail"].lower()

# ── Success (no GCS) ───────────────────────────────────────────────────

def test_success(client, wav_4s, txt_file):
    r = client.post(
        "/voices/upload",
        data={"voice_name": "myvoice", "uid": "user1"},
        files={
            "wav_file": ("rec.wav", wav_4s, "audio/wav"),
            "txt_file": ("rec.txt", txt_file, "text/plain")
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["voice_name"] == "myvoice"
    assert data["status"] == "success"
    assert data["duration"] > 0


def test_embedding_extraction_failure(client, wav_4s, txt_file):
    with patch("app.extract_speaker_embedding", return_value=False):
        r = client.post(
            "/voices/upload",
            data={"voice_name": "myvoice", "uid": "user1"},
            files={
                "wav_file": ("rec.wav", wav_4s, "audio/wav"),
                "txt_file": ("rec.txt", txt_file, "text/plain"),
            },
        )
    assert r.status_code == 200
    data = r.json()
    assert data["voice_name"] == "myvoice"


def test_voice_name_slash_stripped(client, wav_4s, txt_file):
    r = client.post(
        "/voices/upload",
        data={"voice_name": "my/voice", "uid": "user1"},
        files={
            "wav_file": ("rec.wav", wav_4s, "audio/wav"),
            "txt_file": ("rec.txt", txt_file, "text/plain")
        },
    )
    assert r.status_code == 200
    assert r.json()["voice_name"] == "my_voice"


def test_upload_with_uid(client, wav_4s, txt_file):
    r = client.post(
        "/voices/upload",
        data={"voice_name": "myvoice", "uid": "user1"},
        files={
            "wav_file": ("rec.wav", wav_4s, "audio/wav"),
            "txt_file": ("rec.txt", txt_file, "text/plain")
        },
    )
    assert r.status_code == 200
    assert r.json()["voice_name"] == "myvoice"


# ── With GCS (mock) ────────────────────────────────────────────────────

def test_gcs_upload_success(client, gcs_env, wav_4s, txt_file):
    mock_blob = MagicMock()
    mock_bucket = MagicMock()
    mock_bucket.blob.return_value = mock_blob
    mock_client = MagicMock()
    mock_client.bucket.return_value = mock_bucket
    _mock_storage.Client.return_value = mock_client

    with patch("app.extract_speaker_embedding", return_value=True), \
         patch("app.load_voice_clone_prompt", return_value={"fake": "vcp"}):
        r = client.post(
            "/voices/upload",
            data={"voice_name": "gcsvoice", "uid": "user1"},
            files={
                "wav_file": ("rec.wav", wav_4s, "audio/wav"),
                "txt_file": ("rec.txt", txt_file, "text/plain"),
            },
        )

    assert r.status_code == 200
    assert r.json()["voice_name"] == "gcsvoice"
    assert mock_blob.upload_from_filename.call_count == 2


def test_gcs_upload_failure(client, gcs_env, wav_4s, txt_file):
    _mock_storage.Client.side_effect = Exception("network error")

    with patch("app.extract_speaker_embedding", return_value=True), \
         patch("app.load_voice_clone_prompt", return_value={"fake": "vcp"}):
        r = client.post(
            "/voices/upload",
            data={"voice_name": "failvoice", "uid": "user1"},
            files={
                "wav_file": ("rec.wav", wav_4s, "audio/wav"),
                "txt_file": ("rec.txt", txt_file, "text/plain")
            },
        )

    assert r.status_code == 200
    assert r.json()["voice_name"] == "failvoice"
