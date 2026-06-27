#!/usr/bin/env python3
"""
Faster Qwen3-TTS Demo Server

Usage:
    python demo/server.py
    python demo/server.py --model Qwen/Qwen3-TTS-12Hz-1.7B-Base --port 7860
    python demo/server.py --no-preload  # skip startup model load
"""

import argparse
import asyncio
import base64
from collections import OrderedDict
import hashlib
import hmac
import io
import json
import os
import secrets
import sqlite3
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import soundfile as sf
import torch
import torchaudio
import uvicorn
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response, StreamingResponse

# Allow running from any directory
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from faster_qwen3_tts import FasterQwen3TTS, get_optimal_device, device_supports_cuda_graphs
except ImportError:
    print("Error: faster_qwen3_tts not found.")
    print("Install with:  pip install -e .  (from the repo root)")
    sys.exit(1)

try:
    from huggingface_hub import attach_huggingface_oauth, parse_huggingface_oauth
except ImportError:
    print("Error: Hugging Face OAuth helpers not found.")
    print("Install with:  pip install 'huggingface-hub[oauth]>=0.36.0'")
    sys.exit(1)

try:
    from authlib.integrations.base_client.errors import OAuthError
except ImportError:
    OAuthError = None

from nano_parakeet import from_pretrained as _parakeet_from_pretrained

device = get_optimal_device("auto")
dtype = torch.bfloat16 if device_supports_cuda_graphs(device) else torch.float32


_ALL_MODELS = [
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
]

_active_models_env = os.environ.get("ACTIVE_MODELS", "")
if _active_models_env:
    _allowed = {m.strip() for m in _active_models_env.split(",") if m.strip()}
    AVAILABLE_MODELS = [m for m in _ALL_MODELS if m in _allowed]
else:
    AVAILABLE_MODELS = list(_ALL_MODELS)

BASE_DIR = Path(__file__).resolve().parent
INDEX_HTML = BASE_DIR / "index.html"
# Assets that need to be downloaded at runtime go to a writable directory.
# /app is read-only in HF Spaces; fall back to /tmp.
_ASSET_DIR = Path(os.environ.get("ASSET_DIR", "/tmp/faster-qwen3-tts-assets"))
USAGE_BUCKET_MOUNT_PATH = Path(os.environ.get("USAGE_BUCKET_MOUNT_PATH", "/data"))
USAGE_DB_FILENAME = os.environ.get("USAGE_DB_FILENAME", "faster-qwen3-tts-usage.sqlite3")


def _default_usage_db_path() -> Path:
    if USAGE_BUCKET_MOUNT_PATH.exists() and os.access(USAGE_BUCKET_MOUNT_PATH, os.W_OK):
        return USAGE_BUCKET_MOUNT_PATH / USAGE_DB_FILENAME
    return _ASSET_DIR / "usage.sqlite3"


USAGE_DB_PATH = Path(os.environ.get("USAGE_DB_PATH", str(_default_usage_db_path())))
PRESET_TRANSCRIPTS = _ASSET_DIR / "samples" / "parity" / "icl_transcripts.txt"
PRESET_REFS = [
    ("ref_audio_3", _ASSET_DIR / "ref_audio_3.wav", "Clone 1"),
    ("ref_audio_2", _ASSET_DIR / "ref_audio_2.wav", "Clone 2"),
    ("ref_audio", _ASSET_DIR / "ref_audio.wav", "Clone 3"),
]

_GITHUB_RAW = "https://raw.githubusercontent.com/andimarafioti/faster-qwen3-tts/main"
_PRESET_REMOTE = {
    "ref_audio":   f"{_GITHUB_RAW}/ref_audio.wav",
    "ref_audio_2": f"{_GITHUB_RAW}/ref_audio_2.wav",
    "ref_audio_3": f"{_GITHUB_RAW}/ref_audio_3.wav",
}
_TRANSCRIPT_REMOTE = f"{_GITHUB_RAW}/samples/parity/icl_transcripts.txt"


def _fetch_preset_assets() -> None:
    """Download preset wav files and transcripts from GitHub if not present locally."""
    import urllib.request
    _ASSET_DIR.mkdir(parents=True, exist_ok=True)
    PRESET_TRANSCRIPTS.parent.mkdir(parents=True, exist_ok=True)
    if not PRESET_TRANSCRIPTS.exists():
        try:
            urllib.request.urlretrieve(_TRANSCRIPT_REMOTE, PRESET_TRANSCRIPTS)
        except Exception as e:
            print(f"Warning: could not fetch transcripts: {e}")
    for key, path, _ in PRESET_REFS:
        if not path.exists() and key in _PRESET_REMOTE:
            try:
                urllib.request.urlretrieve(_PRESET_REMOTE[key], path)
                print(f"Downloaded {path.name}")
            except Exception as e:
                print(f"Warning: could not fetch {key}: {e}")

_preset_refs: dict[str, dict] = {}


def _load_preset_transcripts() -> dict[str, str]:
    if not PRESET_TRANSCRIPTS.exists():
        return {}
    transcripts = {}
    for line in PRESET_TRANSCRIPTS.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key_part, text = line.split(":", 1)
        key = key_part.split("(")[0].strip()
        transcripts[key] = text.strip()
    return transcripts


def _load_preset_refs() -> None:
    transcripts = _load_preset_transcripts()
    for key, path, label in PRESET_REFS:
        if not path.exists():
            continue
        content = path.read_bytes()
        cached_path = _get_cached_ref_path(content)
        _preset_refs[key] = {
            "id": key,
            "label": label,
            "filename": path.name,
            "path": cached_path,
            "ref_text": transcripts.get(key, ""),
            "audio_b64": base64.b64encode(content).decode(),
        }


def _prime_preset_voice_cache(model: FasterQwen3TTS) -> None:
    if not _preset_refs:
        return
    for preset in _preset_refs.values():
        ref_path = preset["path"]
        ref_text = preset["ref_text"]
        for xvec_only in (True, False):
            try:
                model._prepare_generation(
                    text="Hello.",
                    ref_audio=ref_path,
                    ref_text=ref_text,
                    language="English",
                    xvec_only=xvec_only,
                    non_streaming_mode=True,
                )
            except Exception:
                continue


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() not in {"0", "false", "no", "off", ""}


WEB_ONLY_MODE = _env_flag("DEMO_WEB_ONLY", "0")
REQUIRE_LOGIN = _env_flag("DEMO_REQUIRE_LOGIN", "1" if WEB_ONLY_MODE else "0")
WEB_TOKEN_TTL_SECONDS = int(os.environ.get("DEMO_WEB_TOKEN_TTL_SECONDS", "7200"))
WEB_TOKEN_HEADER = "x-fqtts-web-token"
DAILY_FREE_REQUESTS = int(os.environ.get("DEMO_DAILY_FREE_REQUESTS", "10"))
_web_gate_secret = os.environ.get("DEMO_WEB_GATE_SECRET")
if _web_gate_secret:
    _WEB_GATE_SECRET_BYTES = _web_gate_secret.encode("utf-8")
else:
    _WEB_GATE_SECRET_BYTES = secrets.token_bytes(32)
_usage_hash_secret = os.environ.get("DEMO_USAGE_HASH_SECRET") or _web_gate_secret
if _usage_hash_secret:
    _USAGE_HASH_SECRET_BYTES = _usage_hash_secret.encode("utf-8")
else:
    # Private but not stable across restarts. Set DEMO_USAGE_HASH_SECRET as a
    # Space Secret when using a persistent usage bucket.
    _USAGE_HASH_SECRET_BYTES = _WEB_GATE_SECRET_BYTES
_usage_lock = threading.Lock()
_usage_db_initialized = False


def _host_netloc(value: str | None) -> str:
    return (value or "").split("/", 1)[0].lower()


def _request_netloc(request: Request) -> str:
    return _host_netloc(request.headers.get("host"))


def _cors_origins() -> list[str]:
    origins = {"http://localhost:7860", "http://127.0.0.1:7860"}
    space_host = os.environ.get("SPACE_HOST")
    if space_host:
        origins.add(f"https://{space_host}")
    return sorted(origins)


def _client_fingerprint(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for", "")
    if forwarded_for:
        client_ip = forwarded_for.split(",", 1)[0].strip()
    elif request.client:
        client_ip = request.client.host
    else:
        client_ip = ""
    user_agent = request.headers.get("user-agent", "")[:256]
    return f"{client_ip}|{user_agent}"


def _sign_web_token(ts: str, nonce: str, request: Request) -> str:
    msg = f"{ts}.{nonce}.{_client_fingerprint(request)}".encode("utf-8")
    digest = hmac.new(_WEB_GATE_SECRET_BYTES, msg, hashlib.sha256).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def _make_web_token(request: Request) -> str:
    ts = str(int(time.time()))
    nonce = secrets.token_urlsafe(18)
    sig = _sign_web_token(ts, nonce, request)
    return f"{ts}.{nonce}.{sig}"


def _verify_web_token(token: str, request: Request) -> bool:
    try:
        ts, nonce, sig = token.split(".", 2)
        issued_at = int(ts)
    except (ValueError, TypeError):
        return False

    now = int(time.time())
    if issued_at > now + 60 or now - issued_at > WEB_TOKEN_TTL_SECONDS:
        return False

    expected = _sign_web_token(ts, nonce, request)
    return hmac.compare_digest(sig, expected)


def _same_origin_or_referer(request: Request) -> bool:
    allowed = {_request_netloc(request)}
    space_host = os.environ.get("SPACE_HOST")
    if space_host:
        allowed.add(_host_netloc(space_host))

    origin = request.headers.get("origin")
    if origin:
        return _host_netloc(urlparse(origin).netloc) in allowed

    referer = request.headers.get("referer")
    if referer:
        return _host_netloc(urlparse(referer).netloc) in allowed

    return False


async def require_web_client(request: Request) -> None:
    if not WEB_ONLY_MODE:
        return

    fetch_site = request.headers.get("sec-fetch-site")
    if fetch_site and fetch_site not in {"same-origin", "same-site", "none"}:
        raise HTTPException(status_code=403, detail="Use the web UI to run this demo.")

    if not _same_origin_or_referer(request):
        raise HTTPException(status_code=403, detail="Use the web UI to run this demo.")

    token = request.headers.get(WEB_TOKEN_HEADER, "")
    if not _verify_web_token(token, request):
        raise HTTPException(status_code=403, detail="Open the demo page before making requests.")


def _oauth_info(request: Request):
    return parse_huggingface_oauth(request)


def _oauth_user(oauth_info):
    return getattr(oauth_info, "user_info", None) if oauth_info is not None else None


def _user_sub(user) -> str:
    sub = getattr(user, "sub", None) or getattr(user, "preferred_username", None)
    return str(sub or "")


def _hash_user_identifier(user_id: str) -> str:
    digest = hmac.new(_USAGE_HASH_SECRET_BYTES, user_id.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"hfuser_{digest}"


def _user_key(user) -> str:
    return _hash_user_identifier(_user_sub(user))


def _user_name(user) -> str:
    return str(getattr(user, "preferred_username", None) or getattr(user, "name", None) or _user_sub(user))


def _is_pro_user(user) -> bool:
    return bool(getattr(user, "is_pro", False))


async def require_authenticated_user(request: Request):
    if not REQUIRE_LOGIN:
        return None

    oauth_info = _oauth_info(request)
    user = _oauth_user(oauth_info)
    if user is None or not _user_sub(user):
        raise HTTPException(status_code=401, detail="Sign in with Hugging Face to use this demo.")
    return oauth_info


def _today_key() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _ensure_usage_db_locked() -> None:
    global _usage_db_initialized
    if _usage_db_initialized:
        return
    USAGE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(USAGE_DB_PATH, timeout=30) as con:
        _ensure_usage_schema(con)
    _usage_db_initialized = True


def _create_usage_table(con: sqlite3.Connection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS usage_daily (
            user_key TEXT NOT NULL,
            day TEXT NOT NULL,
            is_pro INTEGER NOT NULL DEFAULT 0,
            count INTEGER NOT NULL DEFAULT 0,
            updated_at INTEGER NOT NULL,
            PRIMARY KEY (user_key, day)
        )
        """
    )


def _create_usage_users_table(con: sqlite3.Connection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS usage_users (
            user_key TEXT PRIMARY KEY,
            username TEXT NOT NULL,
            is_pro INTEGER NOT NULL DEFAULT 0,
            first_seen_at INTEGER NOT NULL,
            last_seen_at INTEGER NOT NULL
        )
        """
    )


def _usage_columns(con: sqlite3.Connection) -> set[str]:
    return {row[1] for row in con.execute("PRAGMA table_info(usage_daily)").fetchall()}


def _ensure_usage_schema(con: sqlite3.Connection) -> None:
    _create_usage_users_table(con)
    columns = _usage_columns(con)
    if not columns:
        _create_usage_table(con)
        return

    expected = {"user_key", "day", "is_pro", "count", "updated_at"}
    if columns == expected:
        return

    legacy_name = "usage_daily_legacy_privacy"
    con.execute(f"DROP TABLE IF EXISTS {legacy_name}")
    con.execute(f"ALTER TABLE usage_daily RENAME TO {legacy_name}")
    _create_usage_table(con)

    legacy_columns = {row[1] for row in con.execute(f"PRAGMA table_info({legacy_name})").fetchall()}
    if {"user_sub", "day", "is_pro", "count", "updated_at"}.issubset(legacy_columns):
        rows = con.execute(
            f"SELECT user_sub, day, is_pro, count, updated_at FROM {legacy_name}"
        ).fetchall()
        for user_sub, day, is_pro, count, updated_at in rows:
            user_key = _hash_user_identifier(str(user_sub))
            con.execute(
                """
                INSERT INTO usage_daily (user_key, day, is_pro, count, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(user_key, day) DO UPDATE SET
                    is_pro = excluded.is_pro,
                    count = MAX(usage_daily.count, excluded.count),
                    updated_at = MAX(usage_daily.updated_at, excluded.updated_at)
                """,
                (user_key, day, int(is_pro), int(count), int(updated_at)),
            )
        if "username" in legacy_columns:
            users = con.execute(
                f"""
                SELECT user_sub, username, MAX(is_pro), MIN(updated_at), MAX(updated_at)
                FROM {legacy_name}
                GROUP BY user_sub, username
                """
            ).fetchall()
            for user_sub, username, is_pro, first_seen_at, last_seen_at in users:
                _record_usage_user(
                    con,
                    _hash_user_identifier(str(user_sub)),
                    str(username or user_sub),
                    bool(is_pro),
                    int(last_seen_at),
                    first_seen_at=int(first_seen_at),
                )
    elif {"user_key", "day", "is_pro", "count", "updated_at"}.issubset(legacy_columns):
        rows = con.execute(
            f"SELECT user_key, day, is_pro, count, updated_at FROM {legacy_name}"
        ).fetchall()
        con.executemany(
            """
            INSERT OR REPLACE INTO usage_daily (user_key, day, is_pro, count, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            rows,
        )

    con.execute(f"DROP TABLE {legacy_name}")


def _record_usage_user(
    con: sqlite3.Connection,
    user_key: str,
    username: str,
    is_pro: bool,
    seen_at: int,
    *,
    first_seen_at: int | None = None,
) -> None:
    first_seen = seen_at if first_seen_at is None else first_seen_at
    existing = con.execute(
        "SELECT first_seen_at, last_seen_at FROM usage_users WHERE user_key = ?",
        (user_key,),
    ).fetchone()
    if existing:
        con.execute(
            """
            UPDATE usage_users
            SET username = ?,
                is_pro = ?,
                first_seen_at = ?,
                last_seen_at = ?
            WHERE user_key = ?
            """,
            (
                username,
                1 if is_pro else 0,
                min(int(existing[0]), int(first_seen)),
                max(int(existing[1]), int(seen_at)),
                user_key,
            ),
        )
    else:
        con.execute(
            """
            INSERT INTO usage_users (user_key, username, is_pro, first_seen_at, last_seen_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (user_key, username, 1 if is_pro else 0, int(first_seen), int(seen_at)),
        )


def _usage_payload_for_count(user, day: str, count: int) -> dict:
    is_pro = _is_pro_user(user)
    limit = None if is_pro else DAILY_FREE_REQUESTS
    remaining = None if is_pro else max(0, DAILY_FREE_REQUESTS - count)
    return {
        "day": day,
        "used_today": count,
        "limit": limit,
        "remaining": remaining,
        "is_pro": is_pro,
    }


def _get_usage(oauth_info) -> dict:
    user = _oauth_user(oauth_info)
    day = _today_key()
    user_key = _user_key(user)
    is_pro = _is_pro_user(user)
    now = int(time.time())
    with _usage_lock:
        _ensure_usage_db_locked()
        with sqlite3.connect(USAGE_DB_PATH, timeout=30) as con:
            _record_usage_user(con, user_key, _user_name(user), is_pro, now)
            row = con.execute(
                "SELECT count FROM usage_daily WHERE user_key = ? AND day = ?",
                (user_key, day),
            ).fetchone()
    count = int(row[0]) if row else 0
    return _usage_payload_for_count(user, day, count)


def _consume_generation_quota(oauth_info) -> dict:
    user = _oauth_user(oauth_info)
    day = _today_key()
    user_key = _user_key(user)
    is_pro = _is_pro_user(user)
    now = int(time.time())

    with _usage_lock:
        _ensure_usage_db_locked()
        with sqlite3.connect(USAGE_DB_PATH, timeout=30) as con:
            _record_usage_user(con, user_key, _user_name(user), is_pro, now)
            row = con.execute(
                "SELECT count FROM usage_daily WHERE user_key = ? AND day = ?",
                (user_key, day),
            ).fetchone()
            count = int(row[0]) if row else 0
            if not is_pro and count >= DAILY_FREE_REQUESTS:
                raise HTTPException(
                    status_code=429,
                    detail=f"Daily free limit reached ({DAILY_FREE_REQUESTS} generations/day). Hugging Face PRO users have unlimited access.",
                )

            count += 1
            con.execute(
                """
                INSERT INTO usage_daily (user_key, day, is_pro, count, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(user_key, day) DO UPDATE SET
                    is_pro = excluded.is_pro,
                    count = excluded.count,
                    updated_at = excluded.updated_at
                """,
                (user_key, day, 1 if is_pro else 0, count, now),
            )

    return _usage_payload_for_count(user, day, count)


def _user_payload(oauth_info) -> dict:
    user = _oauth_user(oauth_info)
    return {
        "username": _user_name(user),
        "is_pro": _is_pro_user(user),
    }


def _login_page() -> HTMLResponse:
    return HTMLResponse(
        """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Sign in - Faster Qwen3-TTS</title>
<style>
* { box-sizing: border-box; }
html, body { height: 100%; margin: 0; }
body {
  display: grid; place-items: center; padding: 24px;
  background: #09090b; color: #fafafa;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
}
.panel {
  width: min(360px, 100%); border: 1px solid #27272a; border-radius: 8px;
  background: #18181b; padding: 22px; box-shadow: 0 1px 3px rgba(0,0,0,0.4);
}
h1 { margin: 0 0 8px; font-size: 18px; font-weight: 650; }
p { margin: 0 0 18px; color: #a1a1aa; font-size: 14px; line-height: 1.45; }
a {
  display: inline-flex; align-items: center; justify-content: center; width: 100%;
  height: 40px; border-radius: 6px; background: #fafafa; color: #09090b;
  text-decoration: none; font-weight: 650; font-size: 14px;
}
</style>
</head>
<body>
  <main class="panel">
    <h1>faster-qwen3-tts</h1>
    <p>Sign in with Hugging Face to use the demo.</p>
    <a href="/oauth/huggingface/login" target="_blank" rel="noopener">Sign in with Hugging Face</a>
  </main>
</body>
</html>
        """,
        headers={"Cache-Control": "no-store"},
    )


app = FastAPI(title="Faster Qwen3-TTS Demo")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_methods=["*"],
    allow_headers=["content-type", WEB_TOKEN_HEADER],
)
attach_huggingface_oauth(app)
if OAuthError is not None:
    @app.exception_handler(OAuthError)
    async def oauth_error_handler(request: Request, exc) -> RedirectResponse:
        return RedirectResponse("/", status_code=303)

_model_cache: OrderedDict[str, FasterQwen3TTS] = OrderedDict()
_model_cache_max: int = int(os.environ.get("MODEL_CACHE_SIZE", "2"))
_active_model_name: str | None = None
_loading = False
_ref_cache: dict[str, str] = {}
_ref_cache_lock = threading.Lock()
_parakeet = None
_generation_lock = asyncio.Lock()
_generation_waiters: int = 0  # requests waiting for or holding the generation lock

# Guard against inputs that would overflow the static KV cache (max_seq_len=2048).
# At ~3-4 chars/token for English the overhead of system/ref tokens leaves room
# for roughly 1000 chars before we approach the limit.
MAX_TEXT_CHARS = 1000
# ~10 MB covers 1 minute of 44.1 kHz stereo 16-bit WAV.
MAX_AUDIO_BYTES = 10 * 1024 * 1024
_AUDIO_TOO_LARGE_MSG = (
    "Audio file too large ({size_mb:.1f} MB). "
    "Voice cloning works best with short clips under 1 minute — please upload a shorter recording."
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _to_wav_b64(audio: np.ndarray, sr: int) -> str:
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    if audio.ndim > 1:
        audio = audio.squeeze()
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return b64


def _concat_audio(audio_list) -> np.ndarray:
    if isinstance(audio_list, np.ndarray):
        return audio_list.astype(np.float32).squeeze()
    parts = [np.array(a, dtype=np.float32).squeeze() for a in audio_list if len(a) > 0]
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)

def _get_cached_ref_path(content: bytes) -> str:
    digest = hashlib.sha1(content).hexdigest()
    with _ref_cache_lock:
        cached = _ref_cache.get(digest)
        if cached and os.path.exists(cached):
            return cached
        tmp_dir = Path(tempfile.gettempdir())
        path = tmp_dir / f"faster_qwen3_tts_ref_{digest}.wav"
        if not path.exists():
            path.write_bytes(content)
        _ref_cache[digest] = str(path)
        return str(path)


def _default_non_streaming_mode_for_mode(mode: str) -> bool:
    return mode != "voice_clone"


# ─── Routes ───────────────────────────────────────────────────────────────────

_fetch_preset_assets()
_load_preset_refs()

@app.get("/")
async def root(request: Request):
    oauth_info = _oauth_info(request) if REQUIRE_LOGIN else None
    if REQUIRE_LOGIN and _oauth_user(oauth_info) is None:
        return _login_page()

    if not WEB_ONLY_MODE:
        return FileResponse(INDEX_HTML)

    html = INDEX_HTML.read_text(encoding="utf-8")
    token = _make_web_token(request)
    bootstrap = f"<script>window.__FQTTS_WEB_TOKEN__ = {json.dumps(token)};</script>"
    html = html.replace("</head>", f"{bootstrap}\n</head>", 1)
    return HTMLResponse(html, headers={"Cache-Control": "no-store"})


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)


@app.post("/transcribe")
async def transcribe_audio(
    _user = Depends(require_authenticated_user),
    _web_client: None = Depends(require_web_client),
    audio: UploadFile = File(...),
):
    """Transcribe reference audio using nano-parakeet."""
    if _parakeet is None:
        raise HTTPException(status_code=503, detail="Transcription model not loaded")

    content = await audio.read()
    if len(content) > MAX_AUDIO_BYTES:
        raise HTTPException(
            status_code=400,
            detail=_AUDIO_TOO_LARGE_MSG.format(size_mb=len(content) / 1024 / 1024),
        )

    def run():
        wav, sr = sf.read(io.BytesIO(content), dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        wav_t = torch.from_numpy(wav)
        if sr != 16000:
            wav_t = torchaudio.functional.resample(wav_t.unsqueeze(0), sr, 16000).squeeze(0)
        return _parakeet.transcribe(wav_t.to(device))

    text = await asyncio.to_thread(run)
    return {"text": text}


@app.get("/status")
async def get_status(oauth_info = Depends(require_authenticated_user)):
    speakers = []
    model_type = None
    active = _model_cache.get(_active_model_name) if _active_model_name else None
    if active is not None:
        try:
            model_type = active.model.model.tts_model_type
            speakers = active.model.get_supported_speakers() or []
        except Exception:
            speakers = []
    user_payload = _user_payload(oauth_info) if REQUIRE_LOGIN else None
    usage_payload = _get_usage(oauth_info) if REQUIRE_LOGIN else None
    return {
        "loaded": active is not None,
        "model": _active_model_name,
        "loading": _loading,
        "available_models": AVAILABLE_MODELS,
        "model_type": model_type,
        "speakers": speakers,
        "transcription_available": _parakeet is not None,
        "preset_refs": [
            {"id": p["id"], "label": p["label"], "ref_text": p["ref_text"]}
            for p in _preset_refs.values()
        ],
        "user": user_payload,
        "usage": usage_payload,
        "queue_depth": _generation_waiters,
        "cached_models": list(_model_cache.keys()),
    }


@app.get("/preset_ref/{preset_id}")
async def get_preset_ref(
    preset_id: str,
    _user = Depends(require_authenticated_user),
):
    preset = _preset_refs.get(preset_id)
    if not preset:
        raise HTTPException(status_code=404, detail="Preset not found")
    return {
        "id": preset["id"],
        "label": preset["label"],
        "filename": preset["filename"],
        "ref_text": preset["ref_text"],
        "audio_b64": preset["audio_b64"],
    }


@app.post("/load")
async def load_model(
    _user = Depends(require_authenticated_user),
    _web_client: None = Depends(require_web_client),
    model_id: str = Form(...),
):
    global _active_model_name, _loading

    # Already in cache — instant switch, no GPU work needed
    if model_id in _model_cache:
        _active_model_name = model_id
        _model_cache.move_to_end(model_id)
        return {"status": "already_loaded", "model": model_id}

    _loading = True

    def _do_load():
        global _active_model_name, _loading
        try:
            if len(_model_cache) >= _model_cache_max:
                evicted, _ = _model_cache.popitem(last=False)
                print(f"Model cache full — evicted: {evicted}")
            new_model = FasterQwen3TTS.from_pretrained(
                model_id,
                device=device,
                dtype=dtype,
            )
            print("Capturing CUDA graphs…")
            new_model._warmup(prefill_len=100)
            _model_cache[model_id] = new_model
            _model_cache.move_to_end(model_id)
            _active_model_name = model_id
            _prime_preset_voice_cache(new_model)
            print("CUDA graphs captured — model ready.")
        finally:
            _loading = False

    # Hold the generation lock while loading to prevent OOM from concurrent inference
    async with _generation_lock:
        await asyncio.to_thread(_do_load)
    return {"status": "loaded", "model": model_id}


@app.post("/generate/stream")
async def generate_stream(
    oauth_info = Depends(require_authenticated_user),
    _web_client: None = Depends(require_web_client),
    text: str = Form(...),
    language: str = Form("English"),
    mode: str = Form("voice_clone"),
    ref_text: str = Form(""),
    speaker: str = Form(""),
    instruct: str = Form(""),
    xvec_only: bool = Form(True),
    chunk_size: int = Form(8),
    temperature: float = Form(0.9),
    top_k: int = Form(50),
    repetition_penalty: float = Form(1.05),
    non_streaming_mode: bool | None = Form(None),
    ref_preset: str = Form(""),
    ref_audio: UploadFile = File(None),
):
    if not _active_model_name or _active_model_name not in _model_cache:
        raise HTTPException(status_code=400, detail="Model not loaded. Click 'Load' first.")
    if len(text) > MAX_TEXT_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"Text too long ({len(text)} chars). Maximum is {MAX_TEXT_CHARS} characters.",
        )
    if REQUIRE_LOGIN:
        _consume_generation_quota(oauth_info)

    tmp_path = None
    tmp_is_cached = False

    if ref_preset and ref_preset in _preset_refs:
        preset = _preset_refs[ref_preset]
        tmp_path = preset["path"]
        tmp_is_cached = True
        if not ref_text:
            ref_text = preset["ref_text"]
    elif ref_audio and ref_audio.filename:
        content = await ref_audio.read()
        if len(content) > MAX_AUDIO_BYTES:
            raise HTTPException(
                status_code=400,
                detail=_AUDIO_TOO_LARGE_MSG.format(size_mb=len(content) / 1024 / 1024),
            )
        tmp_path = _get_cached_ref_path(content)
        tmp_is_cached = True

    if non_streaming_mode is None:
        non_streaming_mode = _default_non_streaming_mode_for_mode(mode)

    loop = asyncio.get_event_loop()
    queue: asyncio.Queue[str | None] = asyncio.Queue()

    def run_generation():
        try:
            # Resolve the model after the generation lock is held so we always
            # use the currently active model, not a stale reference captured
            # before a concurrent /load request changed the active model.
            model = _model_cache.get(_active_model_name)
            if model is None:
                raise RuntimeError("No model loaded. Please load a model first.")

            t0 = time.perf_counter()
            total_audio_s = 0.0
            voice_clone_ms = 0.0

            if mode == "voice_clone":
                gen = model.generate_voice_clone_streaming(
                    text=text,
                    language=language,
                    ref_audio=tmp_path,
                    ref_text=ref_text,
                    xvec_only=xvec_only,
                    non_streaming_mode=non_streaming_mode,
                    chunk_size=chunk_size,
                    temperature=temperature,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    max_new_tokens=360,  # cap at 30s (12 Hz codec)
                )
            elif mode == "custom":
                if not speaker:
                    raise ValueError("Speaker ID is required for custom voice")
                gen = model.generate_custom_voice_streaming(
                    text=text,
                    speaker=speaker,
                    language=language,
                    instruct=instruct,
                    non_streaming_mode=non_streaming_mode,
                    chunk_size=chunk_size,
                    temperature=temperature,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    max_new_tokens=360,
                )
            else:
                gen = model.generate_voice_design_streaming(
                    text=text,
                    instruct=instruct,
                    language=language,
                    non_streaming_mode=non_streaming_mode,
                    chunk_size=chunk_size,
                    temperature=temperature,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    max_new_tokens=360,
                )

            # Use timing data from the generator itself (measured after voice-clone
            # encoding, so TTFA and RTF reflect pure LLM generation latency).
            ttfa_ms = None
            total_gen_ms = 0.0

            # Prime generator to capture wall-clock time to first chunk
            first_audio = next(gen, None)
            if first_audio is not None:
                audio_chunk, sr, timing = first_audio
                wall_first_ms = (time.perf_counter() - t0) * 1000
                model_ms = timing.get("prefill_ms", 0) + timing.get("decode_ms", 0)
                voice_clone_ms = max(0.0, wall_first_ms - model_ms)
                total_gen_ms += timing.get('prefill_ms', 0) + timing.get('decode_ms', 0)
                if ttfa_ms is None:
                    ttfa_ms = total_gen_ms

                audio_chunk = _concat_audio(audio_chunk)
                dur = len(audio_chunk) / sr
                total_audio_s += dur
                rtf = total_audio_s / (total_gen_ms / 1000) if total_gen_ms > 0 else 0.0

                audio_b64 = _to_wav_b64(audio_chunk, sr)
                payload = {
                    "type": "chunk",
                    "audio_b64": audio_b64,
                    "sample_rate": sr,
                    "ttfa_ms": round(ttfa_ms),
                    "voice_clone_ms": round(voice_clone_ms),
                    "rtf": round(rtf, 3),
                    "total_audio_s": round(total_audio_s, 3),
                    "elapsed_ms": round(time.perf_counter() - t0, 3) * 1000,
                }
                loop.call_soon_threadsafe(queue.put_nowait, json.dumps(payload))

            for audio_chunk, sr, timing in gen:
                # prefill_ms is non-zero only on the first chunk
                total_gen_ms += timing.get('prefill_ms', 0) + timing.get('decode_ms', 0)
                if ttfa_ms is None:
                    ttfa_ms = total_gen_ms  # already in ms

                audio_chunk = _concat_audio(audio_chunk)
                dur = len(audio_chunk) / sr
                total_audio_s += dur
                rtf = total_audio_s / (total_gen_ms / 1000) if total_gen_ms > 0 else 0.0

                audio_b64 = _to_wav_b64(audio_chunk, sr)
                payload = {
                    "type": "chunk",
                    "audio_b64": audio_b64,
                    "sample_rate": sr,
                    "ttfa_ms": round(ttfa_ms),
                    "voice_clone_ms": round(voice_clone_ms),
                    "rtf": round(rtf, 3),
                    "total_audio_s": round(total_audio_s, 3),
                    "elapsed_ms": round(time.perf_counter() - t0, 3) * 1000,
                }
                loop.call_soon_threadsafe(queue.put_nowait, json.dumps(payload))

            rtf = total_audio_s / (total_gen_ms / 1000) if total_gen_ms > 0 else 0.0
            done_payload = {
                "type": "done",
                "ttfa_ms": round(ttfa_ms) if ttfa_ms else 0,
                "voice_clone_ms": round(voice_clone_ms),
                "rtf": round(rtf, 3),
                "total_audio_s": round(total_audio_s, 3),
                "total_ms": round((time.perf_counter() - t0) * 1000),
            }
            loop.call_soon_threadsafe(queue.put_nowait, json.dumps(done_payload))

        except Exception as e:
            import traceback
            err = {"type": "error", "message": str(e), "detail": traceback.format_exc()}
            loop.call_soon_threadsafe(queue.put_nowait, json.dumps(err))
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)
            if tmp_path and os.path.exists(tmp_path) and not tmp_is_cached:
                os.unlink(tmp_path)

    async def sse():
        global _generation_waiters
        lock_acquired = False
        _generation_waiters += 1
        people_ahead = _generation_waiters - 1 + (1 if _generation_lock.locked() else 0)
        try:
            if people_ahead > 0:
                yield f"data: {json.dumps({'type': 'queued', 'position': people_ahead})}\n\n"

            await _generation_lock.acquire()
            lock_acquired = True
            _generation_waiters -= 1

            thread = threading.Thread(target=run_generation, daemon=True)
            thread.start()

            while True:
                msg = await queue.get()
                if msg is None:
                    break
                yield f"data: {msg}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            if lock_acquired:
                _generation_lock.release()
            else:
                _generation_waiters -= 1

    return StreamingResponse(
        sse(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )




@app.post("/generate")
async def generate_non_streaming(
    oauth_info = Depends(require_authenticated_user),
    _web_client: None = Depends(require_web_client),
    text: str = Form(...),
    language: str = Form("English"),
    mode: str = Form("voice_clone"),
    ref_text: str = Form(""),
    speaker: str = Form(""),
    instruct: str = Form(""),
    xvec_only: bool = Form(True),
    temperature: float = Form(0.9),
    top_k: int = Form(50),
    repetition_penalty: float = Form(1.05),
    non_streaming_mode: bool | None = Form(None),
    ref_preset: str = Form(""),
    ref_audio: UploadFile = File(None),
):
    if not _active_model_name or _active_model_name not in _model_cache:
        raise HTTPException(status_code=400, detail="Model not loaded. Click 'Load' first.")
    if len(text) > MAX_TEXT_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"Text too long ({len(text)} chars). Maximum is {MAX_TEXT_CHARS} characters.",
        )
    if REQUIRE_LOGIN:
        _consume_generation_quota(oauth_info)

    tmp_path = None
    tmp_is_cached = False

    if ref_preset and ref_preset in _preset_refs:
        preset = _preset_refs[ref_preset]
        tmp_path = preset["path"]
        tmp_is_cached = True
        if not ref_text:
            ref_text = preset["ref_text"]
    elif ref_audio and ref_audio.filename:
        content = await ref_audio.read()
        if len(content) > MAX_AUDIO_BYTES:
            raise HTTPException(
                status_code=400,
                detail=_AUDIO_TOO_LARGE_MSG.format(size_mb=len(content) / 1024 / 1024),
            )
        tmp_path = _get_cached_ref_path(content)
        tmp_is_cached = True

    if non_streaming_mode is None:
        non_streaming_mode = _default_non_streaming_mode_for_mode(mode)

    def run():
        # Resolve the model after the generation lock is held.
        model = _model_cache.get(_active_model_name)
        if model is None:
            raise RuntimeError("No model loaded. Please load a model first.")
        t0 = time.perf_counter()
        if mode == "voice_clone":
            audio_list, sr = model.generate_voice_clone(
                text=text,
                language=language,
                ref_audio=tmp_path,
                ref_text=ref_text,
                xvec_only=xvec_only,
                non_streaming_mode=non_streaming_mode,
                temperature=temperature,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                max_new_tokens=360,  # cap at 30s (12 Hz codec)
            )
        elif mode == "custom":
            if not speaker:
                raise ValueError("Speaker ID is required for custom voice")
            audio_list, sr = model.generate_custom_voice(
                text=text,
                speaker=speaker,
                language=language,
                instruct=instruct,
                non_streaming_mode=non_streaming_mode,
                temperature=temperature,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                max_new_tokens=360,
            )
        else:
            audio_list, sr = model.generate_voice_design(
                text=text,
                instruct=instruct,
                language=language,
                non_streaming_mode=non_streaming_mode,
                temperature=temperature,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                max_new_tokens=360,
            )
        elapsed = time.perf_counter() - t0
        audio = _concat_audio(audio_list)
        dur = len(audio) / sr
        return audio, sr, elapsed, dur

    global _generation_waiters
    _generation_waiters += 1
    lock_acquired = False
    try:
        await _generation_lock.acquire()
        lock_acquired = True
        _generation_waiters -= 1
        audio, sr, elapsed, dur = await asyncio.to_thread(run)
        rtf = dur / elapsed if elapsed > 0 else 0.0
        return JSONResponse({
            "audio_b64": _to_wav_b64(audio, sr),
            "sample_rate": sr,
            "metrics": {
                "total_ms": round(elapsed * 1000),
                "audio_duration_s": round(dur, 3),
                "rtf": round(rtf, 3),
            },
        })
    finally:
        if lock_acquired:
            _generation_lock.release()
        else:
            _generation_waiters -= 1
        if tmp_path and os.path.exists(tmp_path) and not tmp_is_cached:
            os.unlink(tmp_path)


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Faster Qwen3-TTS Demo Server")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        help="Model to preload at startup (default: 1.7B-Base)",
    )
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 7860)))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument(
        "--no-preload",
        action="store_true",
        help="Skip model loading at startup (load via UI instead)",
    )
    args = parser.parse_args()

    if not args.no_preload:
        global _active_model_name, _parakeet
        print(f"Loading model: {args.model}")
        _startup_model = FasterQwen3TTS.from_pretrained(
            args.model,
            device=device,
            dtype=dtype,
        )
        print("Capturing CUDA graphs…")
        _startup_model._warmup(prefill_len=100)
        _model_cache[args.model] = _startup_model
        _active_model_name = args.model
        _prime_preset_voice_cache(_startup_model)
        print("TTS model ready.")

        print("Loading transcription model (nano-parakeet)…")
        if device_supports_cuda_graphs(device):
            _parakeet = _parakeet_from_pretrained(device="cuda")
            print("Transcription model ready.")
        else:
            print("Transcription model skipped (requires CUDA).")

        print(f"Ready. Open http://localhost:{args.port}")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
