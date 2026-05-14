import argparse
import asyncio
from types import SimpleNamespace

import examples.tts_failover_gateway as gateway
from examples.tts_failover_gateway import (
    Backend,
    GatewayState,
    _build_config,
    _should_failover,
    _split_urls,
    _target_url,
)


def test_split_urls_trims_and_drops_empty_values():
    assert _split_urls(" http://a:1/ , ,http://b:2 ") == ["http://a:1", "http://b:2"]


def test_build_config_uses_ordered_backup_names():
    args = argparse.Namespace(
        primary_url="http://127.0.0.1:8092/",
        backup_urls="http://edge-a:8091,http://edge-b:8091",
        primary_timeout_s=20.0,
        backup_timeout_s=120.0,
        health_timeout_s=2.0,
        circuit_break_s=60.0,
    )

    config = _build_config(args)

    assert config.primary == Backend("primary", "http://127.0.0.1:8092")
    assert config.backups == [
        Backend("backup-1", "http://edge-a:8091"),
        Backend("backup-2", "http://edge-b:8091"),
    ]


def test_target_url_preserves_path_and_query():
    backend = Backend("primary", "http://127.0.0.1:8092")

    assert _target_url(backend, "/api/tts/speak", "a=1") == "http://127.0.0.1:8092/api/tts/speak?a=1"
    assert _target_url(backend, "health", "") == "http://127.0.0.1:8092/health"


def test_should_failover_only_on_server_errors():
    class Response:
        def __init__(self, status_code):
            self.status_code = status_code

    assert _should_failover(Response(200)) is False
    assert _should_failover(Response(400)) is False
    assert _should_failover(Response(500)) is True
    assert _should_failover(Response(503)) is True


def test_gateway_retries_backup_when_primary_returns_5xx(monkeypatch):
    class Response:
        def __init__(self, status_code, content):
            self.status_code = status_code
            self.content = content
            self.headers = {"content-type": "audio/wav"}

    args = argparse.Namespace(
        primary_url="http://primary:8092",
        backup_urls="http://edge:8091",
        primary_timeout_s=20.0,
        backup_timeout_s=120.0,
        health_timeout_s=2.0,
        circuit_break_s=60.0,
    )
    calls = []

    def fake_request(method, url, headers, data, timeout):
        calls.append(url)
        if url.startswith("http://primary:8092/"):
            return Response(503, b"busy")
        return Response(200, b"wav")

    monkeypatch.setattr(gateway.requests, "request", fake_request)
    monkeypatch.setattr(gateway, "state", GatewayState(_build_config(args)))

    class Request:
        method = "POST"
        headers = {"content-type": "application/json"}
        url = SimpleNamespace(query="")

        async def body(self):
            return '{"text":"你好。"}'.encode()

    response = asyncio.run(gateway._proxy(Request(), "/api/tts/speak"))

    assert response.status_code == 200
    assert response.body == b"wav"
    assert response.headers["X-TTS-Backend"] == "backup-1"
    assert response.headers["X-TTS-Failover"] == "true"
    assert calls == [
        "http://primary:8092/api/tts/speak",
        "http://edge:8091/api/tts/speak",
    ]
