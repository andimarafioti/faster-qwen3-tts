#!/usr/bin/env python3
"""Transparent failover gateway for Qwen3-TTS HTTP APIs."""

from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

try:
    from examples.tts_output_validator import (
        enqueue_validation,
        get_validation_result,
        recent_validation_results,
        validation_enabled,
    )
except ModuleNotFoundError:
    from tts_output_validator import (
        enqueue_validation,
        get_validation_result,
        recent_validation_results,
        validation_enabled,
    )

LLAMA_LOCAL_SCRIPTS = Path("/home/ivan/github/llama.cpp/scripts/local")
if str(LLAMA_LOCAL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(LLAMA_LOCAL_SCRIPTS))

from model_resource_manager import (  # noqa: E402
    ResourceBusy,
    TTS_MODEL,
    acquire_model,
    mark_error,
    mark_loaded,
    mark_unloaded,
    problem_detail,
    read_state,
    should_idle_unload,
)


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
        self.primary_lock = asyncio.Lock()
        self.playback_lock = asyncio.Lock()
        self.playback_job_id = ""
        self.playback_task: asyncio.Task[None] | None = None
        self.playback_stop_event: asyncio.Event | None = None
        self.idle_timeout_s = int(os.getenv("QWEN_TTS_IDLE_TIMEOUT", "900"))
        self.worker_service = os.getenv("QWEN_TTS_WORKER_SERVICE", "qwen3-tts-ray.service")
        self.startup_timeout_s = int(os.getenv("QWEN_TTS_STARTUP_TIMEOUT", "300"))
        self.idle_task: asyncio.Task[None] | None = None

    def primary_available(self) -> bool:
        return time.monotonic() >= self.primary_dead_until

    def mark_primary_failed(self, reason: str) -> None:
        self.primary_dead_until = time.monotonic() + self.config.circuit_break_s
        self.last_failover_reason = reason


app = FastAPI(title="Qwen3-TTS failover gateway")
state: GatewayState | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    if state is not None:
        state.idle_task = asyncio.create_task(_idle_watcher())
    try:
        yield
    finally:
        if state is not None and state.idle_task is not None:
            state.idle_task.cancel()
            await asyncio.gather(state.idle_task, return_exceptions=True)


app.router.lifespan_context = lifespan


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


def _is_local_tts_request(method: str, path: str) -> bool:
    if method.upper() != "POST":
        return False
    normalized = "/" + path.strip("/")
    return normalized in {
        "/api/tts/load",
        "/api/tts/plan",
        "/api/tts/speak",
        "/api/tts/speak_json",
        "/api/tts/playback_first_json",
        "/api/tts/playback_first_wav",
        "/v1/audio/speech",
    }


def _json_error_from_response(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text.strip() or f"HTTP {response.status_code}"
    if isinstance(payload, dict):
        detail = payload.get("detail") or payload.get("error") or payload.get("message")
        if detail:
            return str(detail)
    return f"HTTP {response.status_code}"


def _post_json_to_backend(
    backend: Backend,
    path: str,
    payload: dict[str, Any],
    timeout_s: float,
) -> requests.Response:
    return requests.post(
        _target_url(backend, path, ""),
        json=payload,
        timeout=(min(3.0, timeout_s), timeout_s),
    )


async def _try_tts_backend_json(
    backend: Backend,
    path: str,
    payload: dict[str, Any],
    timeout_s: float,
) -> tuple[requests.Response | None, str | None]:
    try:
        response = await asyncio.to_thread(_post_json_to_backend, backend, path, payload, timeout_s)
    except requests.RequestException as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if response.status_code >= 500:
        return response, f"HTTP {response.status_code}"
    if response.status_code >= 400:
        raise RuntimeError(_json_error_from_response(response))
    return response, None


async def _call_tts_backend(
    path: str,
    payload: dict[str, Any],
    *,
    expect_json: bool,
    timeout_s: float,
) -> Any:
    if state is None:
        raise RuntimeError("Gateway not initialized")

    config = state.config
    tried: list[dict[str, Any]] = []
    primary_lease = None
    primary_allowed = state.primary_available()

    if primary_allowed:
        try:
            primary_lease = await acquire_model(TTS_MODEL, _unload_other_model)
            await _ensure_primary_loaded()
        except ResourceBusy as exc:
            body_detail, _headers = problem_detail(requested_model=TTS_MODEL, exc=exc, instance=path)
            reason = body_detail["detail"]
            tried.append({"backend": config.primary.name, "url": config.primary.url, "reason": reason})
            state.mark_primary_failed(reason)
            primary_allowed = False
        except Exception as exc:
            if primary_lease is not None:
                await primary_lease.release()
                primary_lease = None
            reason = f"{type(exc).__name__}: {exc}"
            tried.append({"backend": config.primary.name, "url": config.primary.url, "reason": reason})
            state.mark_primary_failed(reason)
            primary_allowed = False

    if primary_allowed:
        try:
            response, reason = await _try_tts_backend_json(
                config.primary,
                path,
                payload,
                min(timeout_s, config.primary_timeout_s),
            )
            tried.append({"backend": config.primary.name, "url": config.primary.url, "reason": reason})
            if response is not None and reason is None:
                return response.json() if expect_json else response.content
            state.mark_primary_failed(reason or "primary failed")
        finally:
            if primary_lease is not None:
                await primary_lease.release()

    for backend in config.backups:
        response, reason = await _try_tts_backend_json(backend, path, payload, config.backup_timeout_s)
        tried.append({"backend": backend.name, "url": backend.url, "reason": reason})
        if response is not None and reason is None:
            return response.json() if expect_json else response.content

    raise RuntimeError(f"No TTS backend available: {tried}")


def _normalize_playback_text(text: Any) -> str:
    return str(text or "").strip()


def _normalize_float(value: Any, default: float, low: float, high: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(low, min(high, number))


def _plan_fallback_chunks(text: str, max_chars: int) -> list[dict[str, Any]]:
    parts = [part.strip() for part in text.replace("\r", "\n").split("\n") if part.strip()]
    if not parts:
        parts = [text]
    chunks: list[dict[str, Any]] = []
    for part in parts:
        start = 0
        while start < len(part):
            chunk = part[start : start + max_chars].strip()
            if chunk:
                chunks.append({"index": len(chunks), "text": chunk, "est_chars": len(chunk)})
            start += max_chars
    return chunks


async def _plan_playback_chunks(text: str, trace_id: str, max_chars: int) -> tuple[list[dict[str, Any]], bool]:
    try:
        result = await _call_tts_backend(
            "/api/tts/plan",
            {"text": text, "trace_id": trace_id, "max_chars_per_chunk": max_chars},
            expect_json=True,
            timeout_s=10.0,
        )
        chunks = result.get("chunks") if isinstance(result, dict) else None
        if isinstance(chunks, list):
            normalized = [
                {"index": idx, "text": str(item.get("text") or "").strip(), "est_chars": len(str(item.get("text") or "").strip())}
                for idx, item in enumerate(chunks)
                if isinstance(item, dict) and str(item.get("text") or "").strip()
            ]
            if normalized:
                return normalized, bool(result.get("truncated"))
    except Exception as exc:
        print(f"[TTS-GW] play plan fallback trace_id={trace_id} error={exc}", flush=True)
    return _plan_fallback_chunks(text, max_chars), False


async def _synthesize_playback_chunk(
    *,
    chunk_text: str,
    chunk_index: int,
    trace_id: str,
    speaker: str | None,
    language: str,
    speed: float,
    instruction: str | None,
    retry_count: int,
) -> bytes:
    payload: dict[str, Any] = {
        "text": chunk_text,
        "speed": speed,
        "language": language,
        "trace_id": f"{trace_id}-c{chunk_index + 1}",
    }
    if speaker:
        payload["speaker"] = speaker
    if instruction:
        payload["instruction"] = instruction

    attempt = 0
    while True:
        try:
            return await _call_tts_backend(
                "/api/tts/speak",
                payload,
                expect_json=False,
                timeout_s=120.0,
            )
        except Exception:
            if attempt >= retry_count:
                raise
            attempt += 1
            await asyncio.sleep(0.2)


async def _play_wav_bytes(wav_bytes: bytes, *, job_id: str, chunk_index: int) -> None:
    if not wav_bytes:
        raise RuntimeError("empty TTS audio")
    fd, wav_path = tempfile.mkstemp(prefix=f"qwen-tts-{job_id}-{chunk_index + 1}-", suffix=".wav")
    try:
        with os.fdopen(fd, "wb") as file:
            file.write(wav_bytes)
        player_commands = [
            ["paplay", wav_path],
            ["aplay", wav_path],
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", wav_path],
        ]
        last_error = ""
        for command in player_commands:
            proc = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                _stdout, stderr = await proc.communicate()
            except asyncio.CancelledError:
                if proc.returncode is None:
                    proc.terminate()
                    try:
                        await asyncio.wait_for(proc.wait(), timeout=2)
                    except asyncio.TimeoutError:
                        proc.kill()
                        await proc.wait()
                raise
            if proc.returncode == 0:
                return
            last_error = (stderr or b"").decode("utf-8", "ignore").strip()
        raise RuntimeError(last_error or "no audio player available")
    finally:
        try:
            os.unlink(wav_path)
        except OSError:
            pass


async def _run_playback_job(
    *,
    job_id: str,
    trace_id: str,
    chunks: list[dict[str, Any]],
    speaker: str | None,
    language: str,
    speed: float,
    instruction: str | None,
    prefetch_window: int,
    retry_count: int,
    stop_event: asyncio.Event,
) -> None:
    print(f"[TTS-GW] play start job_id={job_id} trace_id={trace_id} chunks={len(chunks)}", flush=True)
    in_flight: dict[int, asyncio.Task[bytes]] = {}

    def ensure_chunk(index: int) -> None:
        if index < 0 or index >= len(chunks) or index in in_flight or stop_event.is_set():
            return
        chunk_text = str(chunks[index].get("text") or "").strip()
        in_flight[index] = asyncio.create_task(
            _synthesize_playback_chunk(
                chunk_text=chunk_text,
                chunk_index=index,
                trace_id=trace_id,
                speaker=speaker,
                language=language,
                speed=speed,
                instruction=instruction,
                retry_count=retry_count,
            )
        )

    try:
        for index in range(min(len(chunks), prefetch_window)):
            ensure_chunk(index)

        for index in range(len(chunks)):
            if stop_event.is_set():
                break
            for lookahead in range(index, index + prefetch_window):
                ensure_chunk(lookahead)
            task = in_flight.pop(index, None)
            if task is None:
                ensure_chunk(index)
                task = in_flight.pop(index)
            wav_bytes = await task
            validation_id = enqueue_validation(
                expected_text=str(chunks[index].get("text") or ""),
                wav_bytes=wav_bytes,
                trace_id=f"{trace_id}-c{index + 1}",
                endpoint="/api/tts/play",
                speaker=speaker or "",
                language=language,
                metadata={
                    "job_id": job_id,
                    "chunk_index": index,
                    "total_chunks": len(chunks),
                },
            )
            if validation_id:
                chunks[index]["validation_id"] = validation_id
                print(
                    f"[TTS-GW] play validation queued job_id={job_id} "
                    f"chunk={index + 1} validation_id={validation_id}",
                    flush=True,
                )
            ensure_chunk(index + prefetch_window)
            if stop_event.is_set():
                break
            await _play_wav_bytes(wav_bytes, job_id=job_id, chunk_index=index)
        print(f"[TTS-GW] play finished job_id={job_id} trace_id={trace_id}", flush=True)
    except asyncio.CancelledError:
        print(f"[TTS-GW] play cancelled job_id={job_id} trace_id={trace_id}", flush=True)
        raise
    except Exception as exc:
        print(f"[TTS-GW] play failed job_id={job_id} trace_id={trace_id} error={exc}", flush=True)
    finally:
        for task in in_flight.values():
            task.cancel()
        if in_flight:
            await asyncio.gather(*in_flight.values(), return_exceptions=True)


def _clear_playback_if_current(job_id: str) -> None:
    if state is None or state.playback_job_id != job_id:
        return
    state.playback_job_id = ""
    state.playback_task = None
    state.playback_stop_event = None


async def _run_command(args: list[str], timeout_s: int = 60) -> subprocess.CompletedProcess[str]:
    return await asyncio.to_thread(
        subprocess.run,
        args,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_s,
        check=False,
    )


async def _stop_service(service_name: str) -> None:
    await _run_command(["systemctl", "--user", "stop", service_name], timeout_s=120)
    await _run_command(["systemctl", "--user", "reset-failed", service_name], timeout_s=30)


async def _unload_other_model(model: str) -> None:
    if model == "minicpm-v45-ray":
        try:
            await asyncio.to_thread(
                requests.post,
                "http://100.96.79.21:18082/admin/unload",
                timeout=(3, 120),
            )
        except requests.RequestException:
            pass
    elif model == "qwen36-tq3":
        await _stop_service("qwen36-tq3-worker.service")


def _primary_health_ok(config: GatewayConfig) -> bool:
    try:
        response = requests.get(f"{config.primary.url}/health", timeout=(2, config.health_timeout_s))
        return response.status_code < 500
    except requests.RequestException:
        return False


async def _ensure_primary_loaded() -> None:
    if state is None:
        raise RuntimeError("Gateway not initialized")
    async with state.primary_lock:
        if await asyncio.to_thread(_primary_health_ok, state.config):
            mark_loaded(TTS_MODEL)
            return

        result = await _run_command(["systemctl", "--user", "start", state.worker_service], timeout_s=120)
        if result.returncode != 0:
            message = (result.stderr or result.stdout or "systemctl start failed").strip()
            mark_error(TTS_MODEL, message)
            raise RuntimeError(message)

        deadline = time.monotonic() + state.startup_timeout_s
        while time.monotonic() < deadline:
            if await asyncio.to_thread(_primary_health_ok, state.config):
                mark_loaded(TTS_MODEL)
                return
            await asyncio.sleep(1)

        mark_error(TTS_MODEL, "Timed out waiting for local TTS primary")
        raise TimeoutError("Timed out waiting for local TTS primary")


async def _unload_self() -> None:
    if state is None:
        return
    await _stop_service(state.worker_service)
    mark_unloaded(TTS_MODEL)


async def _idle_watcher() -> None:
    while True:
        await asyncio.sleep(1)
        if state is None:
            continue
        if should_idle_unload(TTS_MODEL, state.idle_timeout_s):
            await _unload_self()


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
    primary_lease = None
    local_tts_request = _is_local_tts_request(request.method, path)
    primary_allowed = state.primary_available() or local_tts_request

    if local_tts_request and primary_allowed:
        try:
            primary_lease = await acquire_model(TTS_MODEL, _unload_other_model)
            await _ensure_primary_loaded()
        except ResourceBusy as exc:
            body_detail, _headers = problem_detail(requested_model=TTS_MODEL, exc=exc, instance=str(request.url))
            reason = body_detail["detail"]
            tried.append({"backend": config.primary.name, "url": config.primary.url, "reason": reason})
            state.mark_primary_failed(reason)
            primary_allowed = False
        except Exception as exc:
            if primary_lease is not None:
                await primary_lease.release()
                primary_lease = None
            reason = f"{type(exc).__name__}: {exc}"
            tried.append({"backend": config.primary.name, "url": config.primary.url, "reason": reason})
            state.mark_primary_failed(reason)
            primary_allowed = False

    if primary_allowed:
        primary_response, reason = await _try_backend(
            config.primary,
            request,
            path,
            body,
            config.primary_timeout_s,
        )
        tried.append({"backend": config.primary.name, "url": config.primary.url, "reason": reason})
        if primary_response is not None and reason is None:
            try:
                return Response(
                    content=primary_response.content,
                    status_code=primary_response.status_code,
                    headers=_response_headers(primary_response, config.primary, False),
                    media_type=primary_response.headers.get("content-type"),
                )
            finally:
                if primary_lease is not None:
                    await primary_lease.release()
        if local_tts_request:
            state.mark_primary_failed(reason or "primary failed")
        if primary_lease is not None:
            await primary_lease.release()
            primary_lease = None

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
            "router": read_state(),
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
            "local_idle_timeout_s": state.idle_timeout_s,
            "router": read_state(),
            "backends": rows,
        },
        status_code=200 if any_ok else 503,
    )


@app.get("/api/tts/validation/recent")
async def tts_validation_recent(request: Request, limit: int = 20) -> Response:
    local_results = recent_validation_results(limit)
    if local_results:
        return JSONResponse({"success": True, "enabled": validation_enabled(), "results": local_results})
    return await _proxy(request, "/api/tts/validation/recent")


@app.get("/api/tts/validation/{validation_id}")
async def tts_validation_result(request: Request, validation_id: str, wait_ms: int = 0) -> Response:
    deadline = time.monotonic() + max(0, min(wait_ms, 30000)) / 1000.0
    result = get_validation_result(validation_id)
    while result is not None and result.get("status") in {"queued", "running"} and time.monotonic() < deadline:
        await asyncio.sleep(0.1)
        result = get_validation_result(validation_id)
    if result is not None:
        return JSONResponse(result)
    return await _proxy(request, f"/api/tts/validation/{validation_id}")


@app.post("/api/tts/play")
async def tts_play(request: Request) -> JSONResponse:
    if state is None:
        return JSONResponse({"success": False, "error": "Gateway not initialized"}, status_code=503)

    try:
        data = await request.json()
    except Exception:
        data = {}
    text = _normalize_playback_text(data.get("text"))
    if not text:
        return JSONResponse({"success": False, "error": "No text provided"}, status_code=400)

    trace_id = str(data.get("trace_id") or f"tts-play-{int(time.time() * 1000)}").strip()
    mode = str(data.get("mode") or "replace").strip().lower()
    if mode != "replace":
        mode = "replace"
    speaker = str(data.get("speaker") or "").strip() or None
    language = str(data.get("language") or "Auto").strip() or "Auto"
    instruction = str(data.get("instruction") or "").strip() or None
    speed = _normalize_float(data.get("speed"), 1.0, 0.5, 2.0)
    max_chars = int(_normalize_float(data.get("max_chars_per_chunk"), 45.0, 20.0, 240.0))
    prefetch_window = int(_normalize_float(data.get("prefetch_window"), 3.0, 1.0, 6.0))
    retry_count = int(_normalize_float(data.get("retry_count"), 1.0, 0.0, 3.0))

    chunks, truncated = await _plan_playback_chunks(text, trace_id, max_chars)
    if not chunks:
        return JSONResponse({"success": False, "error": "No playable chunks"}, status_code=400)

    job_id = f"tts-play-{uuid.uuid4().hex[:12]}"
    stop_event = asyncio.Event()

    async with state.playback_lock:
        if state.playback_stop_event is not None:
            state.playback_stop_event.set()
        if state.playback_task is not None and not state.playback_task.done():
            state.playback_task.cancel()
        task = asyncio.create_task(
            _run_playback_job(
                job_id=job_id,
                trace_id=trace_id,
                chunks=chunks,
                speaker=speaker,
                language=language,
                speed=speed,
                instruction=instruction,
                prefetch_window=prefetch_window,
                retry_count=retry_count,
                stop_event=stop_event,
            )
        )
        state.playback_job_id = job_id
        state.playback_stop_event = stop_event
        state.playback_task = task

    task.add_done_callback(lambda _task, current_job_id=job_id: _clear_playback_if_current(current_job_id))
    return JSONResponse(
        {
            "success": True,
            "job_id": job_id,
            "trace_id": trace_id,
            "status": "queued",
            "chunks": chunks,
            "total_chunks": len(chunks),
            "truncated": truncated,
            "mode": mode,
        }
    )


@app.post("/api/tts/play/stop")
async def tts_play_stop() -> JSONResponse:
    if state is None:
        return JSONResponse({"success": False, "error": "Gateway not initialized"}, status_code=503)
    async with state.playback_lock:
        job_id = state.playback_job_id
        if state.playback_stop_event is not None:
            state.playback_stop_event.set()
        if state.playback_task is not None and not state.playback_task.done():
            state.playback_task.cancel()
        state.playback_job_id = ""
        state.playback_task = None
        state.playback_stop_event = None
    return JSONResponse({"success": True, "stopped": bool(job_id), "job_id": job_id})


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
    parser.add_argument("--primary-timeout-s", type=float, default=float(os.getenv("QWEN_TTS_PRIMARY_TIMEOUT_S", "60")))
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
