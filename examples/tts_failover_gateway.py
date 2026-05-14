#!/usr/bin/env python3
"""Transparent failover gateway for Qwen3-TTS HTTP APIs."""

from __future__ import annotations

import argparse
import asyncio
import os
import time
from dataclasses import dataclass
from typing import Any

import requests
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response


HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
}


@dataclass(frozen=True)
class Backend:
    name: str
    url: str


@dataclass
class GatewayConfig:
    primary: Backend
    backups: list[Backend]
    primary_timeout_s: float
    backup_timeout_s: float
    health_timeout_s: float
    circuit_break_s: float


class GatewayState:
    def __init__(self, config: GatewayConfig) -> None:
        self.config = config
        self.primary_dead_until = 0.0
        self.last_failover_reason = ""

    def primary_available(self) -> bool:
        return time.monotonic() >= self.primary_dead_until

    def mark_primary_failed(self, reason: str) -> None:
        self.primary_dead_until = time.monotonic() + self.config.circuit_break_s
        self.last_failover_reason = reason


app = FastAPI(title="Qwen3-TTS failover gateway")
state: GatewayState | None = None


def _split_urls(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip().rstrip("/") for item in value.split(",") if item.strip()]


def _build_config(args: argparse.Namespace) -> GatewayConfig:
    backup_urls = _split_urls(args.backup_urls)
    return GatewayConfig(
        primary=Backend("primary", args.primary_url.rstrip("/")),
        backups=[Backend(f"backup-{idx + 1}", url) for idx, url in enumerate(backup_urls)],
        primary_timeout_s=args.primary_timeout_s,
        backup_timeout_s=args.backup_timeout_s,
        health_timeout_s=args.health_timeout_s,
        circuit_break_s=args.circuit_break_s,
    )


def _request_headers(request: Request) -> dict[str, str]:
    headers: dict[str, str] = {}
    for key, value in request.headers.items():
        if key.lower() not in HOP_BY_HOP_HEADERS:
            headers[key] = value
    return headers


def _response_headers(response: requests.Response, backend: Backend, failover: bool) -> dict[str, str]:
    headers: dict[str, str] = {}
    for key, value in response.headers.items():
        lower = key.lower()
        if lower not in HOP_BY_HOP_HEADERS:
            headers[key] = value
    headers["X-TTS-Backend"] = backend.name
    headers["X-TTS-Backend-Url"] = backend.url
    headers["X-TTS-Failover"] = str(failover).lower()
    return headers


def _target_url(backend: Backend, path: str, query: str) -> str:
    url = f"{backend.url}/{path.lstrip('/')}"
    if query:
        url = f"{url}?{query}"
    return url


def _send(
    backend: Backend,
    request: Request,
    path: str,
    body: bytes,
    timeout_s: float,
) -> requests.Response:
    return requests.request(
        method=request.method,
        url=_target_url(backend, path, request.url.query),
        headers=_request_headers(request),
        data=body,
        timeout=(min(3.0, timeout_s), timeout_s),
    )


def _should_failover(response: requests.Response) -> bool:
    return response.status_code >= 500


async def _try_backend(
    backend: Backend,
    request: Request,
    path: str,
    body: bytes,
    timeout_s: float,
) -> tuple[requests.Response | None, str | None]:
    try:
        response = await asyncio.to_thread(_send, backend, request, path, body, timeout_s)
    except requests.RequestException as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if _should_failover(response):
        return response, f"HTTP {response.status_code}"
    return response, None


async def _proxy(request: Request, path: str) -> Response:
    if state is None:
        return JSONResponse({"status": "error", "error": "Gateway not initialized"}, status_code=503)

    body = await request.body()
    config = state.config
    tried: list[dict[str, Any]] = []

    if state.primary_available():
        primary_response, reason = await _try_backend(
            config.primary,
            request,
            path,
            body,
            config.primary_timeout_s,
        )
        tried.append({"backend": config.primary.name, "url": config.primary.url, "reason": reason})
        if primary_response is not None and reason is None:
            return Response(
                content=primary_response.content,
                status_code=primary_response.status_code,
                headers=_response_headers(primary_response, config.primary, False),
                media_type=primary_response.headers.get("content-type"),
            )
        state.mark_primary_failed(reason or "primary failed")

    for backend in config.backups:
        backup_response, reason = await _try_backend(
            backend,
            request,
            path,
            body,
            config.backup_timeout_s,
        )
        tried.append({"backend": backend.name, "url": backend.url, "reason": reason})
        if backup_response is not None and reason is None:
            return Response(
                content=backup_response.content,
                status_code=backup_response.status_code,
                headers=_response_headers(backup_response, backend, True),
                media_type=backup_response.headers.get("content-type"),
            )

    return JSONResponse(
        {
            "status": "error",
            "error": "No TTS backend available",
            "tried": tried,
            "primary_dead_until_epoch_s": time.time() + max(0.0, state.primary_dead_until - time.monotonic()),
        },
        status_code=503,
    )


def _health_check(backend: Backend, timeout_s: float) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        response = requests.get(f"{backend.url}/health", timeout=(min(2.0, timeout_s), timeout_s))
        elapsed_s = time.perf_counter() - started
        ok = response.status_code < 500
        return {
            "name": backend.name,
            "url": backend.url,
            "ok": ok,
            "status_code": response.status_code,
            "elapsed_s": elapsed_s,
        }
    except requests.RequestException as exc:
        return {
            "name": backend.name,
            "url": backend.url,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


@app.get("/health")
async def health() -> JSONResponse:
    if state is None:
        return JSONResponse({"status": "error", "error": "Gateway not initialized"}, status_code=503)

    config = state.config
    backends = [config.primary] + config.backups
    rows = await asyncio.gather(
        *[asyncio.to_thread(_health_check, backend, config.health_timeout_s) for backend in backends]
    )
    any_ok = any(row.get("ok") for row in rows)
    return JSONResponse(
        {
            "status": "ok" if any_ok else "error",
            "gateway": "qwen3-tts-failover",
            "primary_available": state.primary_available(),
            "last_failover_reason": state.last_failover_reason,
            "backends": rows,
        },
        status_code=200 if any_ok else 503,
    )


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
async def proxy_all(request: Request, path: str) -> Response:
    return await _proxy(request, path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.getenv("QWEN_TTS_GATEWAY_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("QWEN_TTS_GATEWAY_PORT", "8091")))
    parser.add_argument("--primary-url", default=os.getenv("QWEN_TTS_PRIMARY_URL", "http://127.0.0.1:8092"))
    parser.add_argument(
        "--backup-urls",
        default=os.getenv(
            "QWEN_TTS_BACKUP_URLS",
            "http://192.168.31.72:8091,http://192.168.31.74:8091,http://edge.taild500c8.ts.net:8091",
        ),
    )
    parser.add_argument("--primary-timeout-s", type=float, default=float(os.getenv("QWEN_TTS_PRIMARY_TIMEOUT_S", "20")))
    parser.add_argument("--backup-timeout-s", type=float, default=float(os.getenv("QWEN_TTS_BACKUP_TIMEOUT_S", "120")))
    parser.add_argument("--health-timeout-s", type=float, default=float(os.getenv("QWEN_TTS_HEALTH_TIMEOUT_S", "2")))
    parser.add_argument("--circuit-break-s", type=float, default=float(os.getenv("QWEN_TTS_CIRCUIT_BREAK_S", "60")))
    return parser.parse_args()


def main() -> None:
    global state
    args = _parse_args()
    config = _build_config(args)
    if not config.backups:
        raise SystemExit("At least one backup URL is required")
    state = GatewayState(config)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
