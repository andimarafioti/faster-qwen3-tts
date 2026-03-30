import sys
from unittest.mock import MagicMock

_mock_storage = sys.modules["google.cloud.storage"]


# ── No GCS ──────────────────────────────────────────────────────────────

def test_no_uid(client):
    r = client.get("/voices")
    assert r.status_code == 200
    assert r.json()['user_voices'] == []


def test_empty_uid(client):
    r = client.get("/voices", params={"uid": ""})
    assert r.status_code == 200
    assert r.json()['user_voices'] == []


def test_uid_not_found(client):
    r = client.get("/voices", params={"uid": "nonexistent"})
    assert r.status_code == 200
    assert r.json()['user_voices'] == []

def test_uid_with_voices(client):
    import app as app_module
    uid = "user1"
    app_module.voices[f"{uid}/hello"] = ("/fake/hello.wav", "Hello.", {"fake": "vcp"})
    app_module.voices[f"{uid}/bye"] = ("/fake/bye.wav", "Bye.", None)

    r = client.get("/voices", params={"uid": uid})
    assert r.status_code == 200
    user_voices = r.json()["user_voices"]
    assert len(user_voices) == 2
    ids = [v["id"] for v in user_voices]
    assert "hello" in ids
    assert "bye" in ids
    hello = next(v for v in user_voices if v["id"] == "hello")
    bye = next(v for v in user_voices if v["id"] == "bye")
    assert hello["loaded"] is True
    assert hello["has_embedding"] is True
    assert bye["loaded"] is True
    assert bye["has_embedding"] is False


def test_system_voices_listed(client):
    r = client.get("/voices")
    system_voices = r.json()["system_voices"]
    assert any(v["id"] == "test_voice" for v in system_voices)


def test_has_embedding_false_when_vcp_none(client):
    import app as app_module
    app_module.voices["no_embed_voice"] = ("/fake/voice.wav", "Hello.", None)

    r = client.get("/voices")
    system_voices = r.json()["system_voices"]
    voice = next(v for v in system_voices if v["id"] == "no_embed_voice")
    assert voice["has_embedding"] is False


# ── With GCS (mock) ────────────────────────────────────────────────────

def test_gcs_returns_voices(client, gcs_env):
    uid = "gcs-user"

    mock_blob = MagicMock()
    mock_blob.name = f"voices/{uid}/hello.wav"
    mock_blob.size = 1000

    mock_bucket = MagicMock()
    mock_bucket.list_blobs.return_value = [mock_blob]
    mock_client = MagicMock()
    mock_client.bucket.return_value = mock_bucket
    _mock_storage.Client.return_value = mock_client

    r = client.get("/voices", params={"uid": uid})

    assert r.status_code == 200
    user_voices = r.json()["user_voices"]
    assert len(user_voices) == 1
    assert user_voices[0]["id"] == "hello"


def test_gcs_connection_failure(client, gcs_env):
    _mock_storage.Client.side_effect = Exception("auth failed")

    r = client.get("/voices", params={"uid": "any"})

    assert r.status_code == 200
    assert r.json()["user_voices"] == []
