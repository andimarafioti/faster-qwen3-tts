def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"

def test_health_not_ready(client):
    # Simulate model not ready by clearing voices and setting model_ready to False
    import app as app_module
    app_module.voices = {}
    app_module.model_ready = False
    r = client.get("/health")
    assert r.status_code == 503
    data = r.json()
    assert data["detail"] == "Model not ready"

def test_health_returns_voice_count(client):
    import app as app_module
    app_module.voices["voice1"] = ("/fake/voice1.wav", "Hello.", {"fake": "vcp"})
    app_module.voices["voice2"] = ("/fake/voice2.wav", "Hi.", {"fake": "vcp"})
    app_module.model_ready = True
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["voice_count"] == 3

def test_list_voices_no_uid(client):
    r = client.get("/voices")
    assert r.status_code == 200
    data = r.json()
    assert "user_voices" in data
    assert data["user_voices"] == []

def test_list_voices_unknown_uid(client):
    r = client.get("/voices", params={"uid": "nonexistent"})
    assert r.status_code == 200
    assert r.json()["user_voices"] == []

def test_clone_missing_fields(client):
    r = client.post("/voices/upload")
    assert r.status_code == 422

def test_tts_missing_fields(client):
    r = client.post("/tts", json={})
    assert r.status_code == 422

def test_tts_stream_missing_fields(client):
    r = client.post("/tts/stream", json={})
    assert r.status_code == 422

def test_tts_ws_missing_fields(client):
    with client.websocket_connect("/tts/ws") as ws:
        ws.send_text("{}")
        data = ws.receive_json()
        assert data["type"] == "error"