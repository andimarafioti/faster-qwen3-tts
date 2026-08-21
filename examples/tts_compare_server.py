#!/usr/bin/env python3
"""Side-by-side comparison UI for the three local TTS services."""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
from pathlib import Path
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import wave

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field


NOTES_API_BASE = os.environ.get("NOTES_API_BASE", "http://localhost:9999").rstrip("/")
AUDIO_DIR = Path(os.environ.get("TTS_COMPARE_AUDIO_DIR", "/tmp/tts-compare-audio"))
AUDIO_MAX_AGE_S = int(os.environ.get("TTS_COMPARE_AUDIO_MAX_AGE_S", "21600"))
SERVICES = {
    "qwen-clone": {"port": 7860, "name": "Qwen3-TTS 0.6B Base", "kind": "Voice clone · Clone 1"},
    "qwen-design": {"port": 7861, "name": "Qwen3-TTS 1.7B VoiceDesign", "kind": "Designed narrator"},
    "chatterbox": {"port": 7862, "name": "Chatterbox Turbo 350M", "kind": "Voice clone · Clone 1"},
}
VOICE_PRESETS = {
    "clone_1": {"label": "Clone 1", "filename": "ref_audio_3.wav", "qwen": "ref_audio_3"},
    "clone_2": {"label": "Clone 2", "filename": "ref_audio_2.wav", "qwen": "ref_audio_2"},
    "clone_3": {"label": "Clone 3", "filename": "ref_audio.wav", "qwen": "ref_audio"},
}
DESIGN_PRESETS = {
    "warm_narrator": "Warm, natural narrator with clear diction, calm confidence, and a steady pace.",
    "deep_male": "Adult male narrator with a lower pitch, resonant tone, calm confidence, and clear diction.",
    "bright_female": "Adult female narrator with a bright tone, friendly energy, clear diction, and a natural conversational pace.",
    "news_anchor": "Professional broadcast news anchor, neutral accent, crisp articulation, steady pacing, and composed delivery.",
    "audiobook": "Expressive audiobook narrator with gentle character nuance, warm tone, measured pacing, and clear emotional phrasing.",
    "podcast_host": "Conversational podcast host, relaxed and engaging, medium pace, natural pauses, and friendly confidence.",
    "calm_bedtime": "Soft calming bedtime storyteller, low energy, slower pace, gentle warmth, and smooth phrasing.",
    "executive": "Polished executive presenter, confident tone, precise diction, medium-low pitch, and concise professional delivery.",
    "energetic_promo": "Energetic promotional voice, upbeat delivery, bright tone, strong emphasis, and lively pacing.",
    "documentary": "Serious documentary narrator, thoughtful tone, controlled pacing, subtle gravity, and clear articulation.",
    "androgynous": "Androgynous adult voice with neutral pitch, balanced warmth, precise diction, and calm professional delivery.",
}


class CompareRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)
    voice: str = "clone_1"
    design_voice: str = "warm_narrator"


class DialogRow(BaseModel):
    qwen_clone: str = Field(default="", max_length=5000)
    qwen_clone_pause: float = Field(default=0, ge=0, le=60)
    qwen_design: str = Field(default="", max_length=5000)
    qwen_design_pause: float = Field(default=0, ge=0, le=60)
    chatterbox: str = Field(default="", max_length=5000)
    chatterbox_pause: float = Field(default=0, ge=0, le=60)


class DialogRequest(BaseModel):
    rows: list[DialogRow] = Field(min_length=1, max_length=10)
    qwen_voice: str = "clone_1"
    design_voice: str = "warm_narrator"
    chatterbox_voice: str = "clone_1"


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


def _split_text(text: str, max_chars: int = 220) -> list[str]:
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


def _join_wavs(encoded_wavs: list[str]) -> bytes:
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
    return output.getvalue()


def _store_audio(audio: bytes) -> str:
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    cutoff = time.time() - AUDIO_MAX_AGE_S
    for old_file in AUDIO_DIR.glob("*.wav"):
        try:
            if old_file.stat().st_mtime < cutoff:
                old_file.unlink()
        except OSError:
            pass
    audio_id = uuid.uuid4().hex
    target = AUDIO_DIR / f"{audio_id}.wav"
    temporary = AUDIO_DIR / f".{audio_id}.tmp"
    temporary.write_bytes(audio)
    temporary.replace(target)
    return f"/audio/{audio_id}.wav"


def _synthesize(service: str, text: str, voice: str = "clone_1", design_voice: str = "warm_narrator") -> tuple[bytes, dict, float]:
    preset = VOICE_PRESETS.get(voice)
    if preset is None:
        raise ValueError(f"Unknown source voice: {voice}")
    started = time.perf_counter()
    if service == "qwen-clone":
        results = []
        for chunk in _split_text(text):
            body, content_type = _multipart({
                "text": chunk, "language": "English", "mode": "voice_clone",
                "ref_preset": preset["qwen"], "xvec_only": "true",
            })
            results.append(_request_json("http://127.0.0.1:7860/generate", body, content_type))
        audio = _join_wavs([result["audio_b64"] for result in results])
        metrics = {
            "audio_duration_s": sum(result.get("metrics", {}).get("audio_duration_s", 0) for result in results),
            "chunk_count": len(results),
        }
    elif service == "qwen-design":
        instruction = DESIGN_PRESETS.get(design_voice)
        if instruction is None:
            raise ValueError(f"Unknown VoiceDesign preset: {design_voice}")
        payload = json.dumps({
            "text": text,
            "instruction": instruction,
        }).encode()
        result = _request_json("http://127.0.0.1:7861/api/speak", payload, "application/json")
        audio = _fetch_audio(f"http://127.0.0.1:7861{result['url']}")
        metrics = {"audio_duration_s": result.get("duration_s"), "rtf": result.get("rtf")}
    elif service == "chatterbox":
        body, content_type = _multipart({"text": text, "preset": voice, "temperature": "0.8"})
        result = _request_json("http://127.0.0.1:7862/generate", body, content_type)
        audio = _fetch_audio(f"http://127.0.0.1:7862{result['url']}")
        metrics = {"audio_duration_s": result.get("duration_s"), "chunk_count": result.get("chunk_count")}
    else:
        raise ValueError("Unknown service")
    return audio, metrics, time.perf_counter() - started


def _generate(service: str, text: str, voice: str = "clone_1", design_voice: str = "warm_narrator") -> dict:
    audio, metrics, elapsed = _synthesize(service, text, voice, design_voice)
    return {"audio_url": _store_audio(audio), "elapsed_s": round(elapsed, 2), "metrics": metrics}


def _join_dialog(parts: list[tuple[bytes, float]]) -> tuple[bytes, float]:
    output = io.BytesIO()
    parameters = None
    frames: list[bytes] = []
    total_frames = 0
    for audio, pause_s in parts:
        with wave.open(io.BytesIO(audio), "rb") as source:
            current = source.getparams()
            signature = (current.nchannels, current.sampwidth, current.framerate, current.comptype)
            if parameters is None:
                parameters = current
            elif signature != (parameters.nchannels, parameters.sampwidth, parameters.framerate, parameters.comptype):
                raise RuntimeError("Dialog sources returned incompatible WAV formats")
            pause_frames = round(current.framerate * pause_s)
            if pause_frames:
                frames.append(b"\0" * pause_frames * current.nchannels * current.sampwidth)
                total_frames += pause_frames
            source_frames = source.getnframes()
            frames.append(source.readframes(source_frames))
            total_frames += source_frames
    if parameters is None:
        raise ValueError("Enter text in at least one dialog cell")
    with wave.open(output, "wb") as target:
        target.setparams(parameters)
        target.writeframes(b"".join(frames))
    return output.getvalue(), total_frames / parameters.framerate


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


@app.get("/audio/{audio_id}.wav")
def audio_file(audio_id: str):
    if not re.fullmatch(r"[0-9a-f]{32}", audio_id):
        raise HTTPException(status_code=404, detail="Audio not found")
    path = AUDIO_DIR / f"{audio_id}.wav"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Audio not found")
    return FileResponse(path, media_type="audio/wav")


@app.post("/api/generate/{service}")
def generate(service: str, request: CompareRequest):
    if service not in SERVICES:
        raise HTTPException(status_code=404, detail="Unknown TTS service")
    try:
        _post_note(f"[tts-compare] generating -- {SERVICES[service]['name']}")
        return _generate(service, request.text, request.voice, request.design_voice)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc



@app.post("/api/generate-dialog")
def generate_dialog(request: DialogRequest):
    started = time.perf_counter()
    parts: list[tuple[bytes, float]] = []
    part_count = 0
    try:
        for row in request.rows:
            cells = (
                ("qwen-clone", row.qwen_clone, row.qwen_clone_pause, request.qwen_voice),
                ("qwen-design", row.qwen_design, row.qwen_design_pause, request.qwen_voice),
                ("chatterbox", row.chatterbox, row.chatterbox_pause, request.chatterbox_voice),
            )
            for service, text, pause_s, voice in cells:
                text = text.strip()
                if not text:
                    continue
                _post_note(f"[tts-compare/dialog] generating part {part_count + 1} -- {SERVICES[service]['name']}")
                audio, _metrics, _elapsed = _synthesize(service, text, voice, request.design_voice)
                parts.append((audio, pause_s))
                part_count += 1
        audio, duration_s = _join_dialog(parts)
        return {
            "audio_url": _store_audio(audio),
            "elapsed_s": round(time.perf_counter() - started, 2),
            "audio_duration_s": round(duration_s, 2),
            "part_count": part_count,
        }
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


INDEX_HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TTS Listening Room</title><style>
:root{color-scheme:dark;--bg:#090b10;--panel:#141820;--panel-2:#0f131a;--line:#29303b;--line-soft:#202630;--text:#f4f7fb;--muted:#8e99a8;--accent:#8b7cf6;--accent-strong:#a99df8;--good:#68d9a0;--bad:#ff858f}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 50% -28%,#202b42 0,transparent 39%),var(--bg);color:var(--text);font:13px/1.4 Inter,ui-sans-serif,system-ui,sans-serif}
main{max-width:1380px;margin:auto;padding:26px 22px 48px}header{display:flex;justify-content:space-between;gap:20px;align-items:end;margin-bottom:16px}h1{font-size:clamp(27px,3vw,40px);line-height:1.05;margin:0;letter-spacing:-.035em;font-weight:720}.eyebrow{color:var(--accent-strong);text-transform:uppercase;letter-spacing:.15em;font-size:10px;font-weight:750}.intro{color:var(--muted);max-width:650px;margin:7px 0 0;font-size:12px}
.composer{background:linear-gradient(180deg,#121720,var(--panel-2));border:1px solid var(--line);border-radius:12px;padding:12px 14px;box-shadow:0 16px 42px #0004}textarea{width:100%;min-height:92px;resize:vertical;border:0;background:transparent;color:var(--text);font:15px/1.48 ui-sans-serif,system-ui,sans-serif;outline:0}textarea::placeholder{color:#687383}.composer-foot{display:flex;flex-wrap:wrap;justify-content:space-between;gap:10px;align-items:center;color:var(--muted);border-top:1px solid var(--line-soft);padding-top:9px;font-size:11px}button{border:1px solid transparent;border-radius:7px;background:var(--accent);color:#0a0c12;font-size:12px;font-weight:750;padding:8px 12px;cursor:pointer;transition:filter .15s,transform .15s}button:hover{filter:brightness(1.1)}button:active{transform:translateY(1px)}button#clear{background:transparent;border-color:var(--line);color:var(--muted)}button:disabled{opacity:.45;cursor:wait}
.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin-top:12px}.card{min-width:0;background:linear-gradient(180deg,var(--panel),#11151c);border:1px solid var(--line);border-radius:11px;padding:14px;position:relative;overflow:hidden;box-shadow:0 8px 24px #0002}.card:before{content:"";position:absolute;left:0;top:0;right:0;height:2px;background:var(--accent);transform:scaleX(0);transform-origin:left}.card.working:before{animation:load 1.2s ease-in-out infinite}.card.done:before{transform:scaleX(1);background:var(--good)}.card.error:before{transform:scaleX(1);background:var(--bad)}@keyframes load{0%{transform:scaleX(0)}50%{transform:scaleX(.7)}100%{transform:translateX(100%) scaleX(.3)}}
.port{font:10px ui-monospace,SFMono-Regular,monospace;color:var(--accent-strong)}h2{font-size:15px;line-height:1.2;margin:3px 0 1px;letter-spacing:-.01em}.kind,.status,.metrics{color:var(--muted)}.kind{font-size:11px}.voice-picker{display:grid;grid-template-columns:auto minmax(0,1fr);align-items:center;gap:8px;margin-top:10px;padding-top:9px;border-top:1px solid var(--line-soft)}.voice-picker label{font-size:10px;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:.06em}select{min-width:0;width:100%;background:var(--panel-2);color:var(--text);border:1px solid var(--line);border-radius:6px;padding:6px 24px 6px 8px;font-size:11px}.status{min-height:28px;margin:11px 0 7px;font-size:11px}.metrics{display:flex;gap:8px;font-size:10px;margin-top:6px}audio{width:100%;height:32px}.links{margin-top:8px;text-align:right}.links a{color:var(--accent-strong);text-decoration:none;font-size:10px}.links a:hover{text-decoration:underline}@media(max-width:850px){header{display:block}.grid{grid-template-columns:1fr}main{padding:20px 14px 36px}.composer{padding:11px}textarea{min-height:84px}}
.dialog{margin-top:18px;background:linear-gradient(180deg,#121720,var(--panel-2));border:1px solid var(--line);border-radius:12px;padding:14px;box-shadow:0 16px 42px #0003}.dialog-head,.dialog-foot{display:flex;align-items:center;justify-content:space-between;gap:12px}.dialog-head{margin-bottom:10px}.dialog-head h2{font-size:17px;margin:0}.dialog-head p{font-size:11px;color:var(--muted);margin:2px 0 0}.dialog-scroll{overflow-x:auto}.dialog-grid{display:grid;grid-template-columns:34px repeat(3,minmax(260px,1fr));gap:6px;min-width:900px}.dialog-column{font-size:10px;font-weight:750;color:var(--accent-strong);text-transform:uppercase;letter-spacing:.06em;padding:4px 7px}.dialog-row-number{display:flex;align-items:center;justify-content:center;color:var(--muted);font:10px ui-monospace,monospace}.dialog-cell{display:grid;grid-template-columns:48px minmax(0,1fr);gap:5px}.pause-wrap{position:relative}.pause-wrap:after{content:"s";position:absolute;right:6px;top:8px;color:var(--muted);font-size:9px;pointer-events:none}.dialog-pause,.dialog-text{width:100%;border:1px solid var(--line-soft);background:#0c1016;color:var(--text);border-radius:6px;font:11px/1.35 ui-sans-serif,system-ui,sans-serif}.dialog-pause{height:34px;padding:6px 15px 6px 6px}.dialog-text{min-height:34px;height:34px;padding:7px;resize:vertical}.dialog-pause:focus,.dialog-text:focus{outline:1px solid var(--accent);border-color:var(--accent)}.dialog-foot{margin-top:10px;padding-top:10px;border-top:1px solid var(--line-soft)}.dialog-result{display:flex;align-items:center;gap:10px;min-width:0;flex:1}.dialog-result audio{max-width:420px}.dialog-status{font-size:11px;color:var(--muted)}.generate-one{width:100%;margin-top:8px;padding:6px 9px;background:transparent;border-color:var(--line);color:var(--accent-strong);font-size:10px}#clearDialog{background:transparent;border-color:var(--line);color:var(--muted)}@media(max-width:700px){.dialog-foot{align-items:flex-start;flex-direction:column}.dialog-result{width:100%;flex-direction:column;align-items:stretch}}
</style></head><body><main><header><div><div class="eyebrow">Three engines · one script</div><h1>TTS Listening Room</h1><p class="intro">Send identical text to every local voice engine and compare the results as each one arrives.</p></div></header>
<section class="composer"><textarea id="text" maxlength="5000" placeholder="Enter text to synthesize..."></textarea><div class="composer-foot"><span id="count">0 characters · Shift+Enter to generate</span><div><button id="clear">Clear</button> <button id="run">Generate all three</button></div></div></section>
<section class="grid">
<article class="card" data-service="qwen-clone"><span class="port">:7860</span><h2>Qwen3-TTS 0.6B Base</h2><div class="kind">Voice clone</div><div class="voice-picker"><label>Source voice</label><select class="voice"><option value="clone_1">Clone 1 · ref_audio_3.wav</option><option value="clone_2">Clone 2 · ref_audio_2.wav</option><option value="clone_3">Clone 3 · ref_audio.wav</option></select></div><p class="status">Ready</p><audio controls></audio><div class="metrics"></div><button class="generate-one" type="button">Generate this voice</button><div class="links"><a data-port="7860" target="_blank">Open original ↗</a></div></article>
<article class="card" data-service="qwen-design"><span class="port">:7861</span><h2>Qwen3-TTS 1.7B VoiceDesign</h2><div class="kind">Designed voice</div><div class="voice-picker"><label>Voice preset</label><select class="design-voice"><option value="warm_narrator">Warm narrator</option><option value="deep_male">Deep male</option><option value="bright_female">Bright female</option><option value="news_anchor">News anchor</option><option value="audiobook">Audiobook</option><option value="podcast_host">Podcast host</option><option value="calm_bedtime">Calm bedtime</option><option value="executive">Executive</option><option value="energetic_promo">Energetic promo</option><option value="documentary">Documentary</option><option value="androgynous">Androgynous</option></select></div><p class="status">Ready</p><audio controls></audio><div class="metrics"></div><button class="generate-one" type="button">Generate this voice</button><div class="links"><a data-port="7861" target="_blank">Open original ↗</a></div></article>
<article class="card" data-service="chatterbox"><span class="port">:7862</span><h2>Chatterbox Turbo 350M</h2><div class="kind">Voice clone</div><div class="voice-picker"><label>Source voice</label><select class="voice"><option value="clone_1">Clone 1 · ref_audio_3.wav</option><option value="clone_2">Clone 2 · ref_audio_2.wav</option><option value="clone_3">Clone 3 · ref_audio.wav</option></select></div><p class="status">Ready</p><audio controls></audio><div class="metrics"></div><button class="generate-one" type="button">Generate this voice</button><div class="links"><a data-port="7862" target="_blank">Open original ↗</a></div></article>
</section>
<section class="dialog" aria-labelledby="dialogTitle"><div class="dialog-head"><div><h2 id="dialogTitle">Dialog Builder</h2><p>Playback order runs left to right, then down. Empty cells are skipped; each pause occurs before its line.</p></div></div><div class="dialog-scroll"><div id="dialogGrid" class="dialog-grid"></div></div><div class="dialog-foot"><div class="dialog-result"><audio id="dialogAudio" controls></audio><span id="dialogStatus" class="dialog-status">Ready</span></div><div><button id="clearDialog" type="button">Clear dialog</button> <button id="runDialog" type="button">Generate dialog</button></div></div></section>
</main><script>
document.querySelectorAll('a[data-port]').forEach(link=>link.href=`${location.protocol}//${location.hostname}:${link.dataset.port}`);
const text=document.querySelector('#text'),run=document.querySelector('#run'),count=document.querySelector('#count');
const dialogGrid=document.querySelector('#dialogGrid'),runDialog=document.querySelector('#runDialog'),dialogStatus=document.querySelector('#dialogStatus'),dialogAudio=document.querySelector('#dialogAudio');
const dialogColumns=[['qwen_clone','Qwen Clone · :7860'],['qwen_design','VoiceDesign · :7861'],['chatterbox','Chatterbox · :7862']];
dialogGrid.insertAdjacentHTML('beforeend','<div></div>'+dialogColumns.map(([,label])=>`<div class="dialog-column">${label}</div>`).join(''));
for(let row=0;row<10;row++){dialogGrid.insertAdjacentHTML('beforeend',`<div class="dialog-row-number">${row+1}</div>`+dialogColumns.map(([service,label])=>`<div class="dialog-cell" data-row="${row}" data-service="${service}"><label class="pause-wrap" title="Pause before this line"><input class="dialog-pause" type="number" min="0" max="60" step="0.1" value="0" aria-label="Pause before ${label}, row ${row+1}"></label><textarea class="dialog-text" maxlength="5000" rows="1" placeholder="Dialog line…" aria-label="${label}, row ${row+1}"></textarea></div>`).join(''));}
function dialogRows(){return Array.from({length:10},(_,row)=>{const result={};dialogColumns.forEach(([service])=>{const cell=dialogGrid.querySelector(`[data-row="${row}"][data-service="${service}"]`);result[service]=cell.querySelector('.dialog-text').value.trim();result[service+'_pause']=Number(cell.querySelector('.dialog-pause').value)||0});return result})}
function selectedVoice(service){const card=document.querySelector(`[data-service="${service}"]`);return card.querySelector('.voice')?.value||'clone_1'}
document.querySelectorAll('.generate-one').forEach(button=>button.addEventListener('click',()=>{if(text.value.trim())generate(button.closest('.card'))}));
document.querySelector('#clearDialog').addEventListener('click',()=>{dialogGrid.querySelectorAll('.dialog-text').forEach(input=>input.value='');dialogGrid.querySelectorAll('.dialog-pause').forEach(input=>input.value='0');dialogAudio.removeAttribute('src');dialogAudio.load();dialogStatus.textContent='Ready'});
runDialog.addEventListener('click',async()=>{const rows=dialogRows();if(!rows.some(row=>dialogColumns.some(([service])=>row[service])))return;runDialog.disabled=true;runDialog.textContent='Generating…';dialogStatus.textContent='Generating left to right, then down…';dialogAudio.removeAttribute('src');dialogAudio.load();try{const designCard=document.querySelector('[data-service="qwen-design"]');const response=await fetch('/api/generate-dialog',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({rows,qwen_voice:selectedVoice('qwen-clone'),design_voice:designCard.querySelector('.design-voice').value,chatterbox_voice:selectedVoice('chatterbox')})});const data=await response.json();if(!response.ok)throw new Error(message(data.detail));dialogAudio.src=data.audio_url;dialogStatus.textContent=`Complete · ${data.part_count} part${data.part_count===1?'':'s'} · ${data.audio_duration_s.toFixed(1)}s audio · ${data.elapsed_s}s wall`}catch(error){dialogStatus.textContent=error.message}finally{runDialog.disabled=false;runDialog.textContent='Generate dialog'}});
function recount(){count.textContent=`${text.value.length.toLocaleString()} characters · Shift+Enter to generate`}text.addEventListener('input',recount);recount();
document.querySelector("#clear").addEventListener("click",()=>{text.value="";recount();text.focus()});
function message(value){if(Array.isArray(value))return value.map(x=>x.msg||JSON.stringify(x)).join('; ');return String(value||'Generation failed')}
async function generate(card){const service=card.dataset.service,status=card.querySelector('.status'),audio=card.querySelector('audio'),metrics=card.querySelector('.metrics'),voice=card.querySelector('.voice')?.value||'clone_1',designVoice=card.querySelector('.design-voice')?.value||'warm_narrator';card.className='card working';status.textContent='Generating…';metrics.textContent='';audio.removeAttribute('src');audio.load();const started=performance.now();try{const response=await fetch(`/api/generate/${service}`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:text.value.trim(),voice,design_voice:designVoice})});const data=await response.json();if(!response.ok)throw new Error(message(data.detail));audio.src=data.audio_url;const duration=data.metrics?.audio_duration_s;metrics.textContent=`Wall ${data.elapsed_s}s${duration?` · Audio ${Number(duration).toFixed(1)}s`:''}`;status.textContent='Complete — ready to play';card.className='card done'}catch(error){status.textContent=error.message;card.className='card error'}}
run.addEventListener('click',async()=>{if(!text.value.trim())return;text.disabled=true;run.disabled=true;run.textContent='Generating…';await Promise.allSettled([...document.querySelectorAll('.card')].map(generate));text.disabled=false;run.disabled=false;run.textContent='Generate all three'});
text.addEventListener("keydown",event=>{if(event.shiftKey&&event.key==="Enter"){event.preventDefault();run.click()}});
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
