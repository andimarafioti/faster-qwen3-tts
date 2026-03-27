import numpy as np


# ── /tts ────────────────────────────────────────────────────────────────

def test_tts_empty_text(client):
    r = client.post("/tts", json={"text": "", "voice": "test_voice"})
    assert r.status_code == 400


def test_tts_missing_voice(client):
    r = client.post("/tts", json={"text": "Hello world."})
    assert r.status_code == 400


def test_tts_unknown_voice(client):
    r = client.post("/tts", json={"text": "Hello world.", "voice": "unknown_voice"})
    assert r.status_code == 404


def test_tts_valid_request(client):
    import app as app_module
    app_module.model.generate_voice_clone.return_value = (
        [np.zeros((24000 * 4,), dtype=np.float32)],
        24000,
    )
    r = client.post("/tts", json={"text": "Hello world.", "voice": "test_voice"})
    assert r.status_code == 200
    assert r.headers["X-Sample-Rate"] == "24000"
    assert len(r.content) > 0


def test_tts_voice_id_alias(client):
    import app as app_module
    app_module.model.generate_voice_clone.return_value = (
        [np.zeros((24000 * 4,), dtype=np.float32)],
        24000,
    )
    r = client.post("/tts", json={"text": "Hello world.", "voice_id": "test_voice"})
    assert r.status_code == 200


# ── /tts/stream ──────────────────────────────────────────────────────────

def test_tts_stream_no_voices(client):
    import app as app_module
    app_module.voices = {}
    r = client.post("/tts/stream", json={"text": "Hello world."})
    assert r.status_code == 503

def test_tts_stream_unknown_voice(client):
    r = client.post("/tts/stream", json={"text": "Hello world.", "voice": "unknown"})
    assert r.status_code == 404

def test_tts_stream_valid_request(client):
    import app as app_module
    app_module.model.generate_voice_clone_streaming.return_value = iter([
        (np.zeros(256, dtype=np.float32), 24000, {}),
    ])
    r = client.post("/tts/stream", json={"text": "Hello world.", "voice": "test_voice"})
    assert r.status_code == 200
    assert r.headers["X-Sample-Rate"] == "24000"
    assert len(r.content) > 0

# ── /tts/ws ──────────────────────────────────────────────────────────────

def test_tts_ws_invalid_json(client):
    with client.websocket_connect("/tts/ws") as ws:
        ws.send_text("not a json")
        data = ws.receive_json()
        assert data["type"] == "error"


def test_tts_ws_empty_text(client):
    with client.websocket_connect("/tts/ws") as ws:
        ws.send_json({"text": "", "voice": "test_voice"})
        data = ws.receive_json()
        assert data["type"] == "error"
        assert data["message"] == "No text provided"


def test_tts_ws_unknown_voice(client):
    with client.websocket_connect("/tts/ws") as ws:
        ws.send_json({"text": "Hello world.", "voice": "unknown"})
        data = ws.receive_json()
        assert data["type"] == "error"


def test_tts_ws_no_voices(client):
    import app as app_module
    app_module.voices = {}
    with client.websocket_connect("/tts/ws") as ws:
        ws.send_json({"text": "Hello world."})
        data = ws.receive_json()
        assert data["type"] == "error"
        assert data["message"] == "No voices available"


def test_tts_ws_valid_request_end_message(client):
    import app as app_module
    app_module.model.generate_voice_clone_streaming.return_value = iter([
        (np.zeros(256, dtype=np.float32), 24000, {}),
    ])
    with client.websocket_connect("/tts/ws") as ws:
        ws.send_json({"text": "Hello world.", "voice": "test_voice"})
        start = ws.receive_json()
        assert start["type"] == "start"
        assert start["sample_rate"] == 24000
        audio = ws.receive_json()
        assert audio["type"] == "audio"
        end = ws.receive_json()
        assert end["type"] == "end"
        assert "total_chunks" in end
