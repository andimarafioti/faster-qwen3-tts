#!/usr/bin/env python3
"""Side-by-side comparison UI for the three local TTS services."""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import wave

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field


NOTES_API_BASE = os.environ.get("NOTES_API_BASE", "http://localhost:9999").rstrip("/")
SERVICES = {
    "qwen-clone": {"port": 7860, "name": "Qwen3-TTS 0.6B Base", "kind": "Voice clone · Clone 1"},
    "qwen-design": {"port": 7861, "name": "Qwen3-TTS 1.7B VoiceDesign", "kind": "Designed narrator"},
    "chatterbox": {"port": 7862, "name": "Chatterbox Turbo 350M", "kind": "Voice clone · Clone 1"},
}


class CompareRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)


def _post_note(text: str) -> None:
    try:
        body = json.dumps({"channel": "app", "text": text[:500]}).encode()
        request = urllib.request.Request(
            f"{NOTES_API_BASE}/note", data=body,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        urllib.request.urlopen(request, timeout=2).close()
    except Exception:
        pass


def _multipart(fields: dict[str, str]) -> tuple[bytes, str]:
    boundary = f"----ttscompare{uuid.uuid4().hex}"
    parts = []
    for name, value in fields.items():
        parts.extend([
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
            str(value).encode(), b"\r\n",
        ])
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def _request_json(url: str, data: bytes | None = None, content_type: str | None = None) -> dict:
    headers = {"Content-Type": content_type} if content_type else {}
    request = urllib.request.Request(url, data=data, headers=headers, method="POST" if data else "GET")
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        try:
            detail = json.loads(detail).get("detail", detail)
        except json.JSONDecodeError:
            pass
        raise RuntimeError(str(detail)) from exc


def _fetch_audio(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


def _split_text(text: str, max_chars: int = 900) -> list[str]:
    chunks: list[str] = []
    current = ""
    for sentence in re.split(r"(?<=[.!?])\s+", " ".join(text.split())):
        while len(sentence) > max_chars:
            split_at = sentence.rfind(" ", 0, max_chars)
            split_at = split_at if split_at > 0 else max_chars
            piece, sentence = sentence[:split_at].strip(), sentence[split_at:].strip()
            if current:
                chunks.append(current)
                current = ""
            if piece:
                chunks.append(piece)
        candidate = f"{current} {sentence}".strip()
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _join_wavs(encoded_wavs: list[str]) -> str:
    output = io.BytesIO()
    parameters = None
    frames = []
    for encoded in encoded_wavs:
        with wave.open(io.BytesIO(base64.b64decode(encoded)), "rb") as source:
            current = source.getparams()
            signature = (current.nchannels, current.sampwidth, current.framerate, current.comptype)
            if parameters is None:
                parameters = current
            elif signature != (parameters.nchannels, parameters.sampwidth, parameters.framerate, parameters.comptype):
                raise RuntimeError("Port 7860 returned incompatible WAV chunks")
            frames.append(source.readframes(source.getnframes()))
    with wave.open(output, "wb") as target:
        target.setparams(parameters)
        target.writeframes(b"".join(frames))
    return base64.b64encode(output.getvalue()).decode()


def _generate(service: str, text: str) -> dict:
    started = time.perf_counter()
    if service == "qwen-clone":
        results = []
        for chunk in _split_text(text):
            body, content_type = _multipart({
                "text": chunk, "language": "English", "mode": "voice_clone",
                "ref_preset": "ref_audio_3", "xvec_only": "true",
            })
            results.append(_request_json("http://127.0.0.1:7860/generate", body, content_type))
        audio_b64 = _join_wavs([result["audio_b64"] for result in results])
        metrics = {
            "audio_duration_s": sum(result.get("metrics", {}).get("audio_duration_s", 0) for result in results),
            "chunk_count": len(results),
        }
    elif service == "qwen-design":
        payload = json.dumps({
            "text": text,
            "instruction": "Warm, natural narrator with clear diction, calm confidence, and a steady pace.",
        }).encode()
        result = _request_json("http://127.0.0.1:7861/api/speak", payload, "application/json")
        audio_b64 = base64.b64encode(_fetch_audio(f"http://127.0.0.1:7861{result['url']}")).decode()
        metrics = {"audio_duration_s": result.get("duration_s"), "rtf": result.get("rtf")}
    elif service == "chatterbox":
        body, content_type = _multipart({"text": text, "preset": "clone_1", "temperature": "0.8"})
        result = _request_json("http://127.0.0.1:7862/generate", body, content_type)
        audio_b64 = base64.b64encode(_fetch_audio(f"http://127.0.0.1:7862{result['url']}")).decode()
        metrics = {"audio_duration_s": result.get("duration_s"), "chunk_count": result.get("chunk_count")}
    else:
        raise ValueError("Unknown service")
    return {
        "audio_b64": audio_b64,
        "elapsed_s": round(time.perf_counter() - started, 2),
        "metrics": metrics,
    }


app = FastAPI(title="TTS Compare")


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(INDEX_HTML)


@app.get("/api/status")
def status():
    states = {}
    for key, config in SERVICES.items():
        try:
            endpoint = "/api/status" if config["port"] == 7861 else "/status"
            states[key] = {"available": True, **_request_json(f"http://127.0.0.1:{config['port']}{endpoint}")}
        except Exception as exc:
            states[key] = {"available": False, "error": str(exc)}
    return states


@app.post("/api/generate/{service}")
def generate(service: str, request: CompareRequest):
    if service not in SERVICES:
        raise HTTPException(status_code=404, detail="Unknown TTS service")
    try:
        _post_note(f"[tts-compare] generating -- {SERVICES[service]['name']}")
        return _generate(service, request.text)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


INDEX_HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TTS Listening Room</title><style>
:root{color-scheme:dark;--bg:#0b0d12;--panel:#151922;--line:#293142;--text:#f3f5f8;--muted:#99a4b5;--accent:#8cc8ff;--good:#72dda5;--bad:#ff8b94}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 50% -20%,#20304b 0,transparent 42%),var(--bg);color:var(--text);font:15px/1.5 system-ui,sans-serif}
main{max-width:1450px;margin:auto;padding:48px 24px}header{display:flex;justify-content:space-between;gap:24px;align-items:end;margin-bottom:26px}h1{font-size:clamp(30px,5vw,58px);line-height:1;margin:0;letter-spacing:-.04em}.eyebrow{color:var(--accent);text-transform:uppercase;letter-spacing:.16em;font-size:12px;font-weight:700}.intro{color:var(--muted);max-width:620px;margin:12px 0 0}
.composer{background:#11151d;border:1px solid var(--line);border-radius:18px;padding:18px;box-shadow:0 24px 60px #0005}textarea{width:100%;min-height:140px;resize:vertical;border:0;background:transparent;color:var(--text);font:20px/1.55 Georgia,serif;outline:0}button{border:0;border-radius:10px;background:var(--accent);color:#07111c;font-weight:800;padding:12px 18px;cursor:pointer}button:disabled{opacity:.45;cursor:wait}.composer-foot{display:flex;justify-content:space-between;align-items:center;color:var(--muted);border-top:1px solid var(--line);padding-top:14px}
.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px;margin-top:20px}.card{min-width:0;background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:20px;position:relative;overflow:hidden}.card:before{content:"";position:absolute;left:0;top:0;right:0;height:3px;background:var(--accent);transform:scaleX(0);transform-origin:left}.card.working:before{animation:load 1.2s ease-in-out infinite}.card.done:before{transform:scaleX(1);background:var(--good)}.card.error:before{transform:scaleX(1);background:var(--bad)}@keyframes load{0%{transform:scaleX(0)}50%{transform:scaleX(.7)}100%{transform:translateX(100%) scaleX(.3)}}
.port{font:12px ui-monospace,monospace;color:var(--accent)}h2{font-size:20px;margin:5px 0}.kind,.status,.metrics{color:var(--muted)}.status{min-height:48px;margin:20px 0}.metrics{display:flex;gap:12px;font-size:12px;margin-top:12px}audio{width:100%;height:42px}.links{margin-top:16px}.links a{color:var(--accent);text-decoration:none;font-size:13px}@media(max-width:850px){header{display:block}.grid{grid-template-columns:1fr}main{padding:28px 16px}}
</style></head><body><main><header><div><div class="eyebrow">Three engines · one script</div><h1>TTS Listening Room</h1><p class="intro">Send identical text to every local voice engine and compare the results as each one arrives.</p></div></header>
<section class="composer"><textarea id="text" maxlength="5000">The best voice is not always the loudest one. Sometimes it is the voice that makes every word feel inevitable.</textarea><div class="composer-foot"><span id="count">0 characters</span><button id="run">Generate all three</button></div></section>
<section class="grid">
<article class="card" data-service="qwen-clone"><span class="port">:7860</span><h2>Qwen3-TTS 0.6B Base</h2><div class="kind">Voice clone · Clone 1</div><p class="status">Ready</p><audio controls></audio><div class="metrics"></div><div class="links"><a data-port="7860" target="_blank">Open original ↗</a></div></article>
<article class="card" data-service="qwen-design"><span class="port">:7861</span><h2>Qwen3-TTS 1.7B VoiceDesign</h2><div class="kind">Designed narrator</div><p class="status">Ready</p><audio controls></audio><div class="metrics"></div><div class="links"><a data-port="7861" target="_blank">Open original ↗</a></div></article>
<article class="card" data-service="chatterbox"><span class="port">:7862</span><h2>Chatterbox Turbo 350M</h2><div class="kind">Voice clone · Clone 1</div><p class="status">Ready</p><audio controls></audio><div class="metrics"></div><div class="links"><a data-port="7862" target="_blank">Open original ↗</a></div></article>
</section></main><script>
document.querySelectorAll('a[data-port]').forEach(link=>link.href=`${location.protocol}//${location.hostname}:${link.dataset.port}`);
const text=document.querySelector('#text'),run=document.querySelector('#run'),count=document.querySelector('#count');
function recount(){count.textContent=`${text.value.length.toLocaleString()} characters`}text.addEventListener('input',recount);recount();
function message(value){if(Array.isArray(value))return value.map(x=>x.msg||JSON.stringify(x)).join('; ');return String(value||'Generation failed')}
async function generate(card){const service=card.dataset.service,status=card.querySelector('.status'),audio=card.querySelector('audio'),metrics=card.querySelector('.metrics');card.className='card working';status.textContent='Generating…';metrics.textContent='';audio.removeAttribute('src');audio.load();const started=performance.now();try{const response=await fetch(`/api/generate/${service}`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:text.value.trim()})});const data=await response.json();if(!response.ok)throw new Error(message(data.detail));audio.src=`data:audio/wav;base64,${data.audio_b64}`;const duration=data.metrics?.audio_duration_s;metrics.textContent=`Wall ${data.elapsed_s}s${duration?` · Audio ${Number(duration).toFixed(1)}s`:''}`;status.textContent='Complete — ready to play';card.className='card done'}catch(error){status.textContent=error.message;card.className='card error'}}
run.addEventListener('click',async()=>{if(!text.value.trim())return;text.disabled=true;run.disabled=true;run.textContent='Generating…';await Promise.allSettled([...document.querySelectorAll('.card')].map(generate));text.disabled=false;run.disabled=false;run.textContent='Generate all three'});
</script></body></html>'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7850)
    args = parser.parse_args()
    _post_note(f"[tts-compare] ready -- side-by-side listening room on :{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
