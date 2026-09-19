#!/usr/bin/env python3
"""Chatterbox listening room with saved workspaces, dialogue, and long narration."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

import argparse
import base64
from datetime import datetime
import html
import io
import json
import os
from pathlib import Path
import re
import time
import threading
import urllib.error
import urllib.parse
import urllib.request
import uuid
import wave

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from chatterbox_settings import ChatterboxSettings, CONFIG_PRESETS
from pydantic import BaseModel, Field


NOTES_API_BASE = os.environ.get("NOTES_API_BASE", "http://localhost:9999").rstrip("/")
NOTES_PROJECT = os.environ.get("NOTES_PROJECT", "tts-compare")
AUDIO_DIR = Path(os.environ.get("TTS_COMPARE_AUDIO_DIR", "/tmp/tts-compare-audio"))
AUDIO_MAX_AGE_S = int(os.environ.get("TTS_COMPARE_AUDIO_MAX_AGE_S", "21600"))
WORKSPACE_DIR = Path(os.environ.get("TTS_COMPARE_WORKSPACE_DIR", Path(__file__).resolve().parent.parent / "listening_room_workspaces"))
VOICE_DIR = Path(os.environ.get("TTS_SAVED_VOICES_DIR", Path(__file__).resolve().parent.parent / "tts_history" / "voices"))
LONGFORM_STATUS_DIR = Path(os.environ.get("TTS_LONGFORM_STATUS_DIR", "/tmp/tts-longform-status"))
LONGFORM_AUDIO_DIR = Path(os.environ.get(
    "TTS_LONGFORM_AUDIO_DIR", Path(__file__).resolve().parent.parent / "tts_history" / "long_content",
))
ENSEMBLE_PROGRESS: dict[str, dict] = {}
ENSEMBLE_PROGRESS_LOCK = threading.Lock()
DIALOG_PROGRESS: dict[str, dict] = {}
DIALOG_PROGRESS_LOCK = threading.Lock()
LONGFORM_JOBS: dict[str, dict] = {}
LONGFORM_JOBS_LOCK = threading.Lock()
LONGFORM_EXECUTOR = ThreadPoolExecutor(max_workers=1)
SERVICES = {
    "chatterbox": {"port": 7862, "name": "Chatterbox Turbo 350M", "kind": "Voice clone  -  Clone 1"},
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
    config: ChatterboxSettings = Field(default_factory=ChatterboxSettings)
    text: str = Field(min_length=1, max_length=50000)
    voice: str = "clone_1"
    design_voice: str = "warm_narrator"


class EnsembleRequest(BaseModel):
    config: ChatterboxSettings = Field(default_factory=ChatterboxSettings)
    text: str = Field(min_length=1, max_length=50000)
    voice: str = "clone_1"
    design_voice: str = "warm_narrator"
    chatterbox_voice: str = "clone_2"
    job_id: str = ""


class DialogRow(BaseModel):
    qwen_clone: str = Field(default="", max_length=5000)
    qwen_clone_pause: float = Field(default=0, ge=0, le=60)
    qwen_design: str = Field(default="", max_length=5000)
    qwen_design_pause: float = Field(default=0, ge=0, le=60)
    chatterbox: str = Field(default="", max_length=5000)
    chatterbox_pause: float = Field(default=0, ge=0, le=60)


class DialogRequest(BaseModel):
    voice_b: str = "clone_2"
    config: ChatterboxSettings = Field(default_factory=ChatterboxSettings)
    rows: list[DialogRow] = Field(min_length=1, max_length=10)
    qwen_voice: str = "clone_1"
    design_voice: str = "warm_narrator"
    chatterbox_voice: str = "clone_2"
    job_id: str = ""


class WorkspaceName(BaseModel):
    name: str = Field(min_length=1, max_length=80)


class WorkspaceSave(WorkspaceName):
    narration_voice: str = "clone_1"
    voice_b: str = "clone_2"
    config: ChatterboxSettings = Field(default_factory=ChatterboxSettings)
    text: str = Field(default="", max_length=50000)
    rows: list[DialogRow] = Field(min_length=1, max_length=10)
    qwen_voice: str = "clone_1"
    design_voice: str = "warm_narrator"
    chatterbox_voice: str = "clone_2"


class LongContentRequest(BaseModel):
    config: ChatterboxSettings = Field(default_factory=ChatterboxSettings)
    text: str = Field(min_length=1, max_length=200000)
    name: str = Field(default="Long content", min_length=1, max_length=100)
    engine: str = Field(default="chatterbox", pattern="^chatterbox$")
    voice: str = "clone_1"
    design_voice: str = "warm_narrator"


def _post_note(text: str) -> None:
    try:
        body = json.dumps({"channel": "app", "project": NOTES_PROJECT, "text": text[:500]}).encode()
        request = urllib.request.Request(
            f"{NOTES_API_BASE}/note", data=body,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        urllib.request.urlopen(request, timeout=2).close()
    except Exception:
        pass


def _longform_status_path(job_id: str) -> Path:
    return LONGFORM_STATUS_DIR / f"{job_id}.json"


def _save_longform_job(job: dict) -> None:
    job["updated_at"] = time.time()
    LONGFORM_STATUS_DIR.mkdir(parents=True, exist_ok=True)
    temporary = _longform_status_path(job["id"]).with_suffix(".tmp")
    temporary.write_text(json.dumps(job, indent=2), encoding="utf-8")
    temporary.replace(_longform_status_path(job["id"]))
    with LONGFORM_JOBS_LOCK:
        LONGFORM_JOBS[job["id"]] = dict(job)


def _load_longform_jobs() -> None:
    LONGFORM_STATUS_DIR.mkdir(parents=True, exist_ok=True)
    for path in LONGFORM_STATUS_DIR.glob("*.json"):
        try:
            job = json.loads(path.read_text(encoding="utf-8"))
            if job.get("stage") in {"queued", "generating", "stitching"}:
                job["stage"] = "interrupted"
                job["message"] = "Server restarted before this job completed"
                _save_longform_job(job)
            else:
                LONGFORM_JOBS[job["id"]] = job
        except (OSError, json.JSONDecodeError, KeyError):
            continue


def _balanced_slices(text: str, count: int) -> list[str]:
    sentences = [
        part.strip() for part in re.split(r"(?<=[.!?])\s+", " ".join(text.split())) if part.strip()
    ]
    if not sentences:
        return [text.strip()]
    total_words = sum(len(sentence.split()) for sentence in sentences)
    target = total_words / count
    slices: list[list[str]] = [[]]
    words = 0
    for sentence in sentences:
        sentence_words = len(sentence.split())
        if len(slices) < count and slices[-1] and words + sentence_words > target:
            slices.append([])
            words = 0
        slices[-1].append(sentence)
        words += sentence_words
    return [" ".join(parts) for parts in slices if parts]


def _run_longform_job(job_id: str, request: LongContentRequest) -> None:
    with LONGFORM_JOBS_LOCK:
        job = dict(LONGFORM_JOBS[job_id])
    parts = _split_text(request.text, max_chars=4500)
    started = time.perf_counter()
    try:
        rendered = []
        job.update(stage="generating", total=len(parts), completed=0, active_family="chatterbox")
        _save_longform_job(job)
        for index, part in enumerate(parts):
            job["message"] = f"Generating section {index + 1} of {len(parts)}"
            _save_longform_job(job)
            _post_note(f"[long-content/{job_id[:8]}] generating -- section {index + 1}/{len(parts)} on Chatterbox")
            audio, _, _ = _synthesize("chatterbox", part, request.voice, config=request.config)
            rendered.append((audio, 0.0 if index == 0 else 0.42))
            job["completed"] = index + 1
            _save_longform_job(job)
        job.update(stage="stitching", message="Joining narration sections")
        _save_longform_job(job)
        audio, duration = _join_dialog(rendered)
        elapsed = time.perf_counter() - started
        result = {"audio_url": _store_audio(audio, directory=LONGFORM_AUDIO_DIR),
                  "duration_s": round(duration, 2), "generation_s": round(elapsed, 2),
                  "realtime_ratio": round(duration / elapsed, 3)}
        job.update(stage="complete", message="Chatterbox narration is ready", results={"chatterbox": result},
                   elapsed_s=round(elapsed, 2), finished_at=time.time(), config=request.config.model_dump())
        _save_longform_job(job)
        _post_note(f"[long-content/{job_id[:8]}] complete -- narration ready")
    except Exception as exc:
        job.update(stage="error", message=str(exc), error=str(exc), finished_at=time.time())
        _save_longform_job(job)
        _post_note(f"[long-content/{job_id[:8]}] failed -- {str(exc)[:350]}")


def _multipart(fields: dict[str, str], reference: tuple[str, bytes] | None = None) -> tuple[bytes, str]:
    boundary = f"----ttscompare{uuid.uuid4().hex}"
    parts = []
    for name, value in fields.items():
        parts.extend([
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
            str(value).encode(), b"\r\n",
        ])
    if reference is not None:
        filename, audio = reference
        parts.extend([f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="reference"; filename="{Path(filename).name}"\r\n'.encode(),
            b"Content-Type: audio/wav\r\n\r\n", audio, b"\r\n"])
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


def _split_chapters(text: str, target_words: int = 330, minimum_words: int = 240) -> list[str]:
    chapters: list[str] = []
    sections = re.split(r"/\*\s*new\s+voice\s*\*/", text, flags=re.IGNORECASE)
    for section in sections:
        normalized = " ".join(section.split())
        if not normalized:
            continue
        sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", normalized) if part.strip()]
        units: list[str] = []
        for sentence in sentences:
            words = sentence.split()
            units.extend(" ".join(words[index:index + target_words]) for index in range(0, len(words), target_words))
        current: list[str] = []
        word_count = 0
        for unit in units:
            unit_words = len(unit.split())
            if current and word_count >= minimum_words and word_count + unit_words > target_words:
                chapters.append(" ".join(current))
                current, word_count = [], 0
            current.append(unit)
            word_count += unit_words
        if current:
            chapters.append(" ".join(current))
    return chapters

def _set_ensemble_progress(job_id: str, **values) -> None:
    with ENSEMBLE_PROGRESS_LOCK:
        ENSEMBLE_PROGRESS[job_id] = {**ENSEMBLE_PROGRESS.get(job_id, {}), **values}




def _store_audio(audio: bytes, *, directory: Path = AUDIO_DIR) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    if directory == AUDIO_DIR:
        cutoff = time.time() - AUDIO_MAX_AGE_S
        for old_file in AUDIO_DIR.glob("*.wav"):
            try:
                if old_file.stat().st_mtime < cutoff:
                    old_file.unlink()
            except OSError:
                pass
    audio_id = uuid.uuid4().hex
    target = directory / f"{audio_id}.wav"
    temporary = directory / f".{audio_id}.tmp"
    temporary.write_bytes(audio)
    temporary.replace(target)
    return f"/audio/{audio_id}.wav"


def _synthesize(service: str, text: str, voice: str = "clone_1", design_voice: str = "warm_narrator",
                config: ChatterboxSettings | None = None,
                reference: tuple[str, bytes] | None = None) -> tuple[bytes, dict, float]:
    if service != "chatterbox":
        raise ValueError("Only Chatterbox is enabled")
    if reference is None and voice not in VOICE_PRESETS:
        raise ValueError(f"Unknown source voice: {voice}")
    if not text.strip():
        raise ValueError("Enter text to generate")
    settings = config or ChatterboxSettings()
    started = time.perf_counter()
    chunks = [text] if len(text) <= 4500 else _split_text(text, max_chars=4500)
    rendered = []
    chunk_count = 0
    for chunk in chunks:
        fields = {"text": chunk, "preset": voice,
                  **{key: str(value) for key, value in settings.model_dump().items() if value is not None}}
        body, content_type = _multipart(fields, reference)
        result = _request_json("http://127.0.0.1:7862/generate", body, content_type)
        rendered.append((_fetch_audio(f"http://127.0.0.1:7862{result['url']}"), 0.0))
        chunk_count += result.get("chunk_count", 1)
    audio, duration = _join_dialog(rendered)
    elapsed = time.perf_counter() - started
    metrics = {"audio_duration_s": duration, "chunk_count": chunk_count,
               "realtime_ratio": duration / elapsed, "settings": settings.model_dump()}
    return audio, metrics, elapsed


def _generate(service: str, text: str, voice: str = "clone_1", design_voice: str = "warm_narrator",
              config: ChatterboxSettings | None = None) -> dict:
    audio, metrics, elapsed = _synthesize(service, text, voice, design_voice, config)
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


_load_longform_jobs()
app = FastAPI(title="Chatterbox Listening Room")
EXPERIMENT_DIR = Path(__file__).resolve().parent.parent / "chatterbox_experiments"
EXPERIMENT_DIR.mkdir(exist_ok=True)
app.mount("/experiments", StaticFiles(directory=EXPERIMENT_DIR, html=True), name="experiments")

@app.get("/api/configs")
def configurations():
    return {"default": "balanced", "presets": CONFIG_PRESETS}



@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(Path(__file__).with_name("chatterbox_studio.html").read_text(encoding="utf-8"))


@app.post("/api/long-content")
def submit_long_content(request: LongContentRequest):
    if request.voice not in VOICE_PRESETS:
        raise HTTPException(status_code=400, detail="Unknown source voice")
    job_id = uuid.uuid4().hex
    job = {
        "id": job_id, "name": request.name.strip(), "engine": request.engine,
        "stage": "queued", "completed": 0, "total": 0,
        "message": "Waiting for the long-content worker", "created_at": time.time(),
        "character_count": len(request.text), "word_count": len(request.text.split()),
    }
    _save_longform_job(job)
    LONGFORM_EXECUTOR.submit(_run_longform_job, job_id, request)
    _post_note(f"[long-content/{job_id[:8]}] queued -- {job['name']}; {job['word_count']} words")
    return job


@app.get("/api/long-content")
def list_long_content():
    with LONGFORM_JOBS_LOCK:
        jobs = sorted(
            (dict(item) for item in LONGFORM_JOBS.values()),
            key=lambda item: item.get("created_at", 0), reverse=True,
        )
    for job in jobs:
        job["results"] = {
            engine: {**result, "audio_available": _audio_path(result.get("audio_url", "")) is not None}
            for engine, result in job.get("results", {}).items()
        }
    return {"jobs": jobs}


@app.get("/api/long-content/{job_id}")
def get_long_content(job_id: str):
    if not re.fullmatch(r"[0-9a-f]{32}", job_id):
        raise HTTPException(status_code=404, detail="Job not found")
    with LONGFORM_JOBS_LOCK:
        job = LONGFORM_JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/long-content", response_class=HTMLResponse)
def long_content_page():
    return HTMLResponse('''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Long Content TTS</title>
<style>
:root{color-scheme:dark;--bg:#090b10;--panel:#141820;--line:#29303b;--text:#f4f7fb;--muted:#8e99a8;--accent:#a99df8;--good:#68d9a0;--bad:#ff858f}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.4 ui-sans-serif,system-ui,sans-serif}main{max-width:1050px;margin:auto;padding:28px 20px 60px}
header{display:flex;justify-content:space-between;gap:16px;align-items:end;margin-bottom:20px}h1{font-size:36px;margin:0}h2{margin-top:30px}a{color:var(--accent)}
.panel,.job{border:1px solid var(--line);border-radius:11px;background:var(--panel);padding:16px}.row{display:grid;grid-template-columns:2fr 1fr 1fr;gap:10px;margin-bottom:12px}
label{display:grid;gap:5px;color:var(--muted);font-size:12px}input,select,textarea,button{font:inherit}input,select,textarea{width:100%;border:1px solid var(--line);border-radius:7px;background:#0d1117;color:var(--text);padding:9px}
textarea{min-height:320px;resize:vertical;line-height:1.5}button{border:0;border-radius:7px;background:var(--accent);color:#090b10;padding:10px 15px;font-weight:750;cursor:pointer}.actions{display:flex;justify-content:space-between;align-items:center;gap:12px;margin-top:10px}
.job{margin:10px 0}.job-head{display:flex;justify-content:space-between;gap:12px}.muted{color:var(--muted)}progress{width:100%;accent-color:var(--accent);margin:10px 0}.results{display:grid;gap:10px}.result{display:grid;grid-template-columns:minmax(220px,1fr) minmax(260px,1.5fr) auto;align-items:center;gap:10px}audio{width:100%;height:34px}.error{color:var(--bad)}
@media(max-width:700px){.row,.result{grid-template-columns:1fr}header{align-items:start;flex-direction:column}}
</style></head><body><main><header><div><h1>Long Content TTS</h1><div class="muted">Queued narration with Chatterbox Turbo</div></div><div><a href="/">Listening Room</a> · <a href="/generation-status">System status</a> · <a href="/history">History</a></div></header>
<section class="panel"><div class="row"><label>Job name<input id="name" value="Long narration" maxlength="100"></label><label>Engine<select id="engine"><option value="chatterbox">Chatterbox · fastest</option></select></label><label>Load text file<input id="file" type="file" accept=".txt,.md,text/plain,text/markdown"></label></div>
<div class="row"><label>Clone voice<select id="voice"><option value="clone_1">Clone 1</option><option value="clone_2">Clone 2</option><option value="clone_3">Clone 3</option></select></label><label>Configuration<select id="config"><option value="balanced">Balanced</option></select></label><span></span></div>
<textarea id="text" maxlength="200000" placeholder="Paste long content here, or choose a .txt/.md file above."></textarea><div class="actions"><span id="count" class="muted">0 words</span><button id="submit">Queue generation</button></div><div id="formStatus" class="muted"></div></section>
<h2>Jobs</h2><section id="jobs"></section>
<script>
const q=s=>document.querySelector(s),text=q('#text'),file=q('#file'),count=q('#count'),jobs=q('#jobs'),submit=q('#submit'),formStatus=q('#formStatus');
let configs={balanced:{settings:{}}};fetch('/api/configs').then(r=>r.json()).then(d=>{configs=d.presets;const select=q('#config');select.replaceChildren(...Object.entries(configs).map(([id,c])=>{const o=document.createElement('option');o.value=id;o.textContent=c.label;return o}))});
const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function recount(){const words=text.value.trim()?text.value.trim().split(/\s+/).length:0;count.textContent=words.toLocaleString()+' words · '+text.value.length.toLocaleString()+' characters'}text.addEventListener('input',recount);recount();
file.addEventListener('change',async()=>{if(file.files[0]){text.value=await file.files[0].text();if(!q('#name').value||q('#name').value==='Long narration')q('#name').value=file.files[0].name.replace(/\.[^.]+$/,'');recount()}});
submit.addEventListener('click',async()=>{if(!text.value.trim()){formStatus.textContent='Paste or load text first.';return}submit.disabled=true;formStatus.textContent='Submitting…';try{const r=await fetch('/api/long-content',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:q('#name').value||'Long narration',engine:q('#engine').value,voice:q('#voice').value,config:configs[q('#config').value].settings,text:text.value})});const d=await r.json();if(!r.ok)throw new Error(d.detail||'Submission failed');formStatus.textContent='Queued job '+d.id.slice(0,8);await refresh()}catch(e){formStatus.textContent=e.message}finally{submit.disabled=false}});
function resultHtml(job){return Object.entries(job.results||{}).map(([engine,r])=>{const gen=Number(r.generation_s||job.elapsed_s||0),ratio=Number(r.realtime_ratio||(gen?r.duration_s/gen:0)),media=r.audio_available?'<audio controls preload="metadata" src="'+r.audio_url+'"></audio><a href="'+r.audio_url+'" download>Download WAV</a>':'<span class="muted">Audio expired; regenerate to listen</span><span></span>';return '<div class="result"><div><strong>'+esc(engine)+'</strong><div class="muted">'+Number(r.duration_s).toFixed(1)+'s audio · '+gen.toFixed(1)+'s TTS · '+ratio.toFixed(2)+'× real time</div></div>'+media+'</div>'}).join('')}
let lastJobsHtml='';
async function refresh(){try{const r=await fetch('/api/long-content',{cache:'no-store'}),d=await r.json(),next=d.jobs.length?d.jobs.map(j=>'<article class="job"><div class="job-head"><strong>'+esc(j.name)+'</strong><span class="muted">'+esc(j.engine)+' · '+esc(j.stage)+'</span></div><div class="'+(j.stage==='error'?'error':'muted')+'">'+esc(j.message)+'</div><progress max="'+Math.max(1,j.total||1)+'" value="'+(j.completed||0)+'"></progress><div class="muted">'+(j.word_count||0).toLocaleString()+' words'+(j.elapsed_s?' · '+j.elapsed_s+'s wall':'')+'</div><div class="results">'+resultHtml(j)+'</div></article>').join(''):'<p class="muted">No jobs yet.</p>';if(next!==lastJobsHtml&&!Array.from(jobs.querySelectorAll('audio')).some(player=>!player.paused)){jobs.innerHTML=next;lastJobsHtml=next}}catch(e){if(!lastJobsHtml)jobs.innerHTML='<p class="error">'+esc(e.message)+'</p>'}}refresh();setInterval(refresh,2000);
</script></main></body></html>''')


@app.get("/api/generation-status")
def generation_status():
    workers = []
    for family, ports, path in (
        ("Chatterbox", (7862,), "/status"),
    ):
        for index, port in enumerate(ports, 1):
            try:
                state = _request_json(f"http://127.0.0.1:{port}{path}")
                workers.append({"family": family, "worker": index, "port": port, "online": True, "state": state})
            except Exception as exc:
                workers.append({"family": family, "worker": index, "port": port, "online": False, "error": str(exc)})
    jobs = []
    LONGFORM_STATUS_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(LONGFORM_STATUS_DIR.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True)[:20]:
        try:
            jobs.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    activity = []
    try:
        note_history = _request_json(f"{NOTES_API_BASE}/note/history?channel=app").get("history", [])
        activity = [
            item for item in note_history
            if any(tag in item.get("text", "") for tag in ("[tts-9-worker-job]", "[chatterbox-turbo/generate]"))
        ][-20:]
        if not jobs:
            for item in reversed(activity):
                note_text = item.get("text", "")
                if "[tts-9-worker-job]" not in note_text:
                    continue
                match = re.search(r"\[tts-9-worker-job\] (\w+) -- (\d+)/(\d+) slices complete;?\s*(.*)", note_text)
                if match:
                    jobs.append({
                        "id": "legacy-active", "name": "Current long-form generation",
                        "stage": match.group(1), "completed": int(match.group(2)),
                        "total": int(match.group(3)), "message": match.group(4),
                    })
                else:
                    stage_match = re.search(r"\[tts-9-worker-job\] (\w+) -- (.*)", note_text)
                    if stage_match:
                        jobs.append({
                            "id": "legacy-latest", "name": "Latest long-form generation",
                            "stage": stage_match.group(1), "completed": 0, "total": 9,
                            "message": stage_match.group(2),
                        })
                break
    except Exception:
        pass
    audio = []
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(AUDIO_DIR.glob("*.wav"), key=lambda item: item.stat().st_mtime, reverse=True)[:12]:
        stat = path.stat()
        audio.append({"url": f"/audio/{path.stem}.wav", "created_at": stat.st_mtime, "size": stat.st_size})
    return {"workers": workers, "jobs": jobs, "activity": activity, "recent_audio": audio, "updated_at": time.time()}


@app.get("/generation-status", response_class=HTMLResponse)
def generation_status_page():
    return HTMLResponse('''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>TTS Generation Status</title>
<style>
:root{color-scheme:dark;--bg:#090b10;--panel:#141820;--line:#29303b;--text:#f4f7fb;--muted:#8e99a8;--accent:#a99df8;--good:#68d9a0;--bad:#ff858f}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.4 ui-sans-serif,system-ui,sans-serif}main{max-width:1100px;margin:auto;padding:28px 20px 60px}
header{display:flex;justify-content:space-between;align-items:end;gap:16px}h1{margin:0;font-size:34px}h2{margin-top:28px}a{color:var(--accent)}#stamp,.muted{color:var(--muted)}
.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}.card,.job{border:1px solid var(--line);border-radius:10px;background:var(--panel);padding:13px}.online{color:var(--good)}.offline{color:var(--bad)}
.job{margin-bottom:10px}.job progress{width:100%;accent-color:var(--accent)}.parts{display:flex;gap:8px;flex-wrap:wrap;font-size:12px;color:var(--muted)}
.audio{display:grid;grid-template-columns:1fr minmax(260px,430px);gap:12px;align-items:center;margin:8px 0}audio{width:100%;height:34px}
@media(max-width:720px){.grid{grid-template-columns:1fr}.audio{grid-template-columns:1fr}}
</style></head><body><main><header><div><h1>TTS Generation Status</h1><div id="stamp">Loading…</div></div><div><a href="/">Listening Room</a> · <a href="/history">History</a></div></header>
<h2>Workers</h2><div id="workers" class="grid"></div><h2>Long-form jobs</h2><div id="jobs"></div><h2>Live activity</h2><div id="activity"></div><h2>Recent audio</h2><div id="audio"></div>
<script>
const stamp=document.querySelector('#stamp'),workers=document.querySelector('#workers'),jobs=document.querySelector('#jobs'),activity=document.querySelector('#activity'),audio=document.querySelector('#audio');
const esc=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
async function refresh(){try{const r=await fetch('/api/generation-status',{cache:'no-store'}),d=await r.json();
stamp.textContent='Updated '+new Date(d.updated_at*1000).toLocaleTimeString()+' · refreshes every 2 seconds';
workers.innerHTML=d.workers.map(w=>'<article class="card"><strong>'+esc(w.family)+' #'+w.worker+'</strong><div>:'+w.port+'</div><div class="'+(w.online?'online':'offline')+'">'+(w.online?'Online · loaded':'Offline')+'</div></article>').join('');
jobs.innerHTML=d.jobs.length?d.jobs.map(j=>{const total=j.total||9,done=j.completed||0;return '<article class="job"><strong>'+esc(j.name||j.id||'Long-form job')+'</strong><div class="muted">'+esc(j.stage||'unknown')+' · '+done+'/'+total+' slices</div><progress max="'+total+'" value="'+done+'"></progress><div class="parts">'+esc(j.message||'')+'</div></article>'}).join(''):'<p class="muted">No structured jobs recorded yet.</p>';
activity.innerHTML=(d.activity||[]).slice().reverse().map(a=>'<div class="job"><span class="muted">'+new Date(a.updatedAt).toLocaleTimeString()+'</span> '+esc(a.text)+'</div>').join('');
audio.innerHTML=d.recent_audio.map(a=>'<div class="audio"><span>'+new Date(a.created_at*1000).toLocaleString()+' · '+(a.size/1048576).toFixed(1)+' MB</span><audio controls preload="metadata" src="'+a.url+'"></audio></div>').join('');
}catch(e){stamp.textContent='Status unavailable: '+e.message}}refresh();setInterval(refresh,2000);
</script></main></body></html>''')


@app.get("/history", response_class=HTMLResponse)
def history():
    """Show every WAV currently retained by the comparison server."""
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    recordings = []
    for path in AUDIO_DIR.glob("*.wav"):
        try:
            stat = path.stat()
        except OSError:
            continue
        duration_s = None
        try:
            with wave.open(str(path), "rb") as source:
                duration_s = source.getnframes() / source.getframerate()
        except (OSError, EOFError, wave.Error, ZeroDivisionError):
            pass
        recordings.append((stat.st_mtime, path.stem, stat.st_size, duration_s))
    recordings.sort(reverse=True)

    cards = []
    for modified, audio_id, size_bytes, duration_s in recordings:
        timestamp = datetime.fromtimestamp(modified).astimezone().strftime("%b %-d, %Y at %-I:%M:%S %p %Z")
        audio_url = f"/audio/{audio_id}.wav"
        duration = f"{duration_s / 60:.1f} min &middot; " if duration_s is not None else ""
        cards.append(f'''<article class="recording">
<div class="meta"><strong>{html.escape(timestamp)}</strong><span>{duration}{size_bytes / 1024 / 1024:.1f} MB</span></div>
<audio controls preload="metadata" src="{audio_url}"></audio>
<a href="{audio_url}" download="tts-{audio_id}.wav">Download WAV</a>
</article>''')
    content = "\n".join(cards) if cards else '<p class="empty">No retained generations yet.</p>'
    return HTMLResponse(f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Generation History</title>
<style>
:root{{color-scheme:dark;--bg:#090b10;--panel:#141820;--line:#29303b;--text:#f4f7fb;--muted:#8e99a8;--accent:#a99df8}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.4 ui-sans-serif,system-ui,sans-serif}}
main{{max-width:920px;margin:auto;padding:30px 20px 60px}}header{{display:flex;align-items:end;justify-content:space-between;gap:20px;margin-bottom:22px}}
h1{{margin:0;font-size:clamp(28px,5vw,42px)}}header p,.empty{{color:var(--muted)}}a{{color:var(--accent)}}
.recording{{display:grid;grid-template-columns:minmax(210px,1fr) minmax(260px,420px) auto;align-items:center;gap:16px;padding:16px;border:1px solid var(--line);border-radius:11px;background:var(--panel);margin-bottom:10px}}
.meta{{display:grid;gap:4px}}.meta span{{color:var(--muted);font-size:12px}}audio{{width:100%;height:34px}}.recording>a{{white-space:nowrap;font-size:12px}}
@media(max-width:720px){{header{{align-items:start;flex-direction:column}}.recording{{grid-template-columns:1fr}}}}
</style></head><body><main><header><div><h1>Generation History</h1><p>{len(recordings)} retained recording{'s' if len(recordings) != 1 else ''}, newest first</p></div><a href="/">Back to Listening Room</a></header>{content}</main></body></html>''')




def _workspace_path(name: str) -> Path:
    clean_name = name.strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 _.-]{0,79}", clean_name):
        raise HTTPException(
            status_code=400,
            detail="Use 1-80 letters, numbers, spaces, dots, underscores, or hyphens; start with a letter or number",
        )
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    return WORKSPACE_DIR / f"{clean_name}.json"




@app.get("/api/design-presets")
def design_presets():
    return {"presets": []}

@app.get("/api/workspaces")
def list_workspaces():
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    items = []
    for path in sorted(WORKSPACE_DIR.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        items.append({"name": path.stem, "modified": int(path.stat().st_mtime)})
    return {"items": items}


@app.post("/api/workspaces/save")
def save_workspace(request: WorkspaceSave):
    path = _workspace_path(request.name)
    payload = request.model_dump() if hasattr(request, "model_dump") else request.dict()
    payload["name"] = request.name.strip()
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)
    _post_note(f"[listening-room/workspaces] saved -- {payload['name']}")
    return {"name": payload["name"], "saved": True}


@app.post("/api/workspaces/load")
def load_workspace(request: WorkspaceName):
    path = _workspace_path(request.name)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Workspace not found")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail="Workspace file could not be read") from exc

@app.get("/api/status")
def status():
    states = {}
    for key, config in SERVICES.items():
        try:
            endpoint = "/status"
            states[key] = {"available": True, **_request_json(f"http://127.0.0.1:{config['port']}{endpoint}")}
        except Exception as exc:
            states[key] = {"available": False, "error": str(exc)}
    return states


def _audio_path(audio_url: str) -> Path | None:
    match = re.fullmatch(r"/audio/([0-9a-f]{32})\.wav", audio_url)
    if not match:
        return None
    for directory in (LONGFORM_AUDIO_DIR, AUDIO_DIR):
        path = directory / f"{match.group(1)}.wav"
        if path.is_file():
            return path
    return None


@app.get("/audio/{audio_id}.wav")
def audio_file(audio_id: str):
    if not re.fullmatch(r"[0-9a-f]{32}", audio_id):
        raise HTTPException(status_code=404, detail="Audio not found")
    path = _audio_path(f"/audio/{audio_id}.wav")
    if path is None:
        raise HTTPException(status_code=404, detail="Audio not found")
    return FileResponse(path, media_type="audio/wav")


@app.post("/api/generate/{service}")
def generate(service: str, request: CompareRequest):
    if service not in SERVICES:
        raise HTTPException(status_code=404, detail="Unknown TTS service")
    try:
        _post_note(f"[tts-compare] generating -- {SERVICES[service]['name']}")
        return _generate(service, request.text, request.voice, request.design_voice, request.config)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


def _voice_path(voice_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{32}", voice_id):
        raise HTTPException(status_code=404, detail="Saved voice not found")
    path = VOICE_DIR / f"{voice_id}.wav"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Saved voice not found")
    return path


def _saved_voice_items() -> list[dict]:
    items = []
    for path in VOICE_DIR.glob("*.json") if VOICE_DIR.is_dir() else []:
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
            if _voice_path(item["id"]).is_file():
                items.append(item)
        except (OSError, ValueError, KeyError, HTTPException):
            continue
    return sorted(items, key=lambda item: item.get("created_at", 0), reverse=True)


@app.get("/api/voices")
def saved_voices():
    return {"items": _saved_voice_items()}


@app.post("/api/voices")
def save_voice(name: str = Form(...), reference: UploadFile = File(...)):
    name = name.strip()
    if not name or len(name) > 80:
        raise HTTPException(status_code=400, detail="Voice name must be 1-80 characters")
    if Path(reference.filename or "").suffix.lower() != ".wav":
        raise HTTPException(status_code=400, detail="Upload a WAV file")
    audio = reference.file.read(20 * 1024 * 1024 + 1)
    if not audio or len(audio) > 20 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="WAV must be 1 byte to 20 MB")
    try:
        with wave.open(io.BytesIO(audio), "rb") as source:
            if source.getnframes() == 0 or source.getnchannels() not in (1, 2):
                raise ValueError("Voice WAV must contain mono or stereo audio")
            duration_s = source.getnframes() / source.getframerate()
            if duration_s > 120:
                raise ValueError("Voice sample must be under 120 seconds")
    except (wave.Error, EOFError, ZeroDivisionError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid voice WAV: {exc}") from exc
    VOICE_DIR.mkdir(parents=True, exist_ok=True)
    voice_id = uuid.uuid4().hex
    item = {"id": voice_id, "name": name, "duration_s": round(duration_s, 1), "created_at": time.time()}
    path = VOICE_DIR / f"{voice_id}.wav"
    path.write_bytes(audio)
    (VOICE_DIR / f"{voice_id}.json").write_text(json.dumps(item), encoding="utf-8")
    return item


@app.get("/api/voices/{voice_id}.wav")
def saved_voice_audio(voice_id: str):
    return FileResponse(_voice_path(voice_id), media_type="audio/wav")


@app.delete("/api/voices/{voice_id}")
def delete_voice(voice_id: str):
    path = _voice_path(voice_id)
    path.unlink()
    path.with_suffix(".json").unlink(missing_ok=True)
    return {"deleted": voice_id}


@app.post("/api/generate-custom")
def generate_custom(text: str = Form(...), config: str = Form(...),
                    reference: UploadFile | None = File(None), voice_id: str = Form("")):
    if not text.strip() or len(text) > 50000:
        raise HTTPException(status_code=400, detail="Text must be 1-50,000 characters")
    if voice_id:
        path = _voice_path(voice_id)
        audio = path.read_bytes()
        filename = path.name
    elif reference is not None and Path(reference.filename or "").suffix.lower() == ".wav":
        audio = reference.file.read(20 * 1024 * 1024 + 1)
        filename = reference.filename
    else:
        raise HTTPException(status_code=400, detail="Choose a saved voice or upload a WAV")
    if not audio or len(audio) > 20 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="WAV must be 1 byte to 20 MB")
    try:
        settings = ChatterboxSettings.model_validate_json(config)
        rendered, metrics, elapsed = _synthesize("chatterbox", text, "", config=settings,
                                                  reference=(filename, audio))
        return {"audio_url": _store_audio(rendered), "elapsed_s": round(elapsed, 2), "metrics": metrics}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/api/ensemble-progress/{job_id}")
def ensemble_progress(job_id: str):
    if not re.fullmatch(r"[0-9a-f]{32}", job_id):
        raise HTTPException(status_code=404, detail="Progress not found")
    with ENSEMBLE_PROGRESS_LOCK:
        progress_state = ENSEMBLE_PROGRESS.get(job_id)
    if progress_state is None:
        raise HTTPException(status_code=404, detail="Progress not found")
    return progress_state


@app.post("/api/generate-ensemble")
def generate_ensemble(request: EnsembleRequest):
    chapters = _split_chapters(request.text.strip())
    job_id = request.job_id if re.fullmatch(r"[0-9a-f]{32}", request.job_id) else uuid.uuid4().hex
    services = ("chatterbox",)
    started = time.perf_counter()
    rendered: list[bytes | None] = [None] * len(chapters)
    model_sequence = [SERVICES[services[index % len(services)]]["name"] for index in range(len(chapters))]
    completed = 0
    _set_ensemble_progress(job_id, stage="queued", completed=0, total=len(chapters), endpoint="")

    def render_chapter(index: int) -> tuple[int, bytes, str]:
        service = services[index % len(services)]
        endpoint = SERVICES[service]["name"]
        voice = request.chatterbox_voice if service == "chatterbox" else request.voice
        audio, _metrics, _elapsed = _synthesize(service, chapters[index], voice, request.design_voice, request.config)
        return index, audio, endpoint

    try:
        for wave_start in range(0, len(chapters), len(services)):
            wave_indexes = range(wave_start, min(wave_start + len(services), len(chapters)))
            labels = ", ".join(f"{index + 1}:{model_sequence[index]}" for index in wave_indexes)
            _set_ensemble_progress(
                job_id, stage="generating", completed=completed, total=len(chapters),
                endpoint=f"parallel wave {wave_start // len(services) + 1}",
            )
            _post_note(f"[listening-room/ensemble] parallel wave -- {labels}")
            with ThreadPoolExecutor(max_workers=len(services)) as executor:
                futures = [executor.submit(render_chapter, index) for index in wave_indexes]
                for future in as_completed(futures):
                    index, chapter_audio, endpoint = future.result()
                    rendered[index] = chapter_audio
                    completed += 1
                    _set_ensemble_progress(
                        job_id, stage="generating", completed=completed,
                        total=len(chapters), endpoint=f"finished {index + 1}: {endpoint}",
                    )
        parts = [(chapter_audio, 0.0 if index == 0 else 0.8) for index, chapter_audio in enumerate(rendered)]
        audio, duration_s = _join_dialog(parts)
        result = {"audio_url": _store_audio(audio), "elapsed_s": round(time.perf_counter() - started, 2), "audio_duration_s": round(duration_s, 2), "chapter_count": len(chapters), "model_sequence": model_sequence}
        _set_ensemble_progress(job_id, stage="done", completed=len(chapters), total=len(chapters), endpoint="")
        _post_note(f"[listening-room/ensemble] complete -- {len(chapters)} parallel chapters joined in source order")
        return result
    except Exception as exc:
        _set_ensemble_progress(job_id, stage="error", error=str(exc))
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/api/dialog-progress/{job_id}")
def dialog_progress(job_id: str):
    if not re.fullmatch(r"[0-9a-f]{32}", job_id):
        raise HTTPException(status_code=404, detail="Progress not found")
    with DIALOG_PROGRESS_LOCK:
        progress_state = DIALOG_PROGRESS.get(job_id)
    if progress_state is None:
        raise HTTPException(status_code=404, detail="Progress not found")
    return progress_state


@app.post("/api/generate-dialog")
def generate_dialog(request: DialogRequest):
    started = time.perf_counter()
    job_id = request.job_id if re.fullmatch(r"[0-9a-f]{32}", request.job_id) else uuid.uuid4().hex
    tasks: list[tuple[int, str, str, float, str]] = []
    for row in request.rows:
        cells = (
            ("chatterbox", row.qwen_clone, row.qwen_clone_pause, request.qwen_voice),
            ("chatterbox", row.qwen_design, row.qwen_design_pause, request.voice_b),
            ("chatterbox", row.chatterbox, row.chatterbox_pause, request.chatterbox_voice),
        )
        for service, text, pause_s, voice in cells:
            if text.strip():
                tasks.append((len(tasks), service, text.strip(), pause_s, voice))
    if not tasks:
        raise HTTPException(status_code=400, detail="Enter text in at least one dialog cell")

    lanes = {service: [task for task in tasks if task[1] == service] for service in SERVICES}
    active_lanes = {service: lane for service, lane in lanes.items() if lane}
    model_progress = {
        service: {"name": SERVICES[service]["name"], "completed": 0, "total": len(lane)}
        for service, lane in active_lanes.items()
    }
    with DIALOG_PROGRESS_LOCK:
        DIALOG_PROGRESS[job_id] = {
            "stage": "queued", "completed": 0, "total": len(tasks),
            "models": model_progress, "last": "",
        }

    def render_lane(service: str, lane: list[tuple[int, str, str, float, str]]):
        lane_results = []
        for index, _service, text, pause_s, voice in lane:
            _post_note(f"[listening-room/dialog] generating part {index + 1}/{len(tasks)} -- {SERVICES[service]['name']}")
            audio, _metrics, _elapsed = _synthesize(service, text, voice, request.design_voice, request.config)
            lane_results.append((index, audio, pause_s))
            with DIALOG_PROGRESS_LOCK:
                state = DIALOG_PROGRESS[job_id]
                state["stage"] = "generating"
                state["completed"] += 1
                state["models"][service]["completed"] += 1
                state["last"] = f"Part {index + 1} finished on {SERVICES[service]['name']}"
        return lane_results

    try:
        ordered: list[tuple[bytes, float] | None] = [None] * len(tasks)
        with ThreadPoolExecutor(max_workers=len(active_lanes)) as executor:
            futures = [executor.submit(render_lane, service, lane) for service, lane in active_lanes.items()]
            for future in as_completed(futures):
                for index, audio, pause_s in future.result():
                    ordered[index] = (audio, pause_s)
        audio, duration_s = _join_dialog(ordered)
        result = {
            "audio_url": _store_audio(audio),
            "elapsed_s": round(time.perf_counter() - started, 2),
            "audio_duration_s": round(duration_s, 2),
            "part_count": len(tasks),
        }
        with DIALOG_PROGRESS_LOCK:
            DIALOG_PROGRESS[job_id]["stage"] = "done"
            DIALOG_PROGRESS[job_id]["last"] = "Stitched in spreadsheet order"
        _post_note(f"[listening-room/dialog] complete -- {len(tasks)} parts from {len(active_lanes)} parallel lanes stitched in order")
        return result
    except Exception as exc:
        with DIALOG_PROGRESS_LOCK:
            DIALOG_PROGRESS[job_id]["stage"] = "error"
            DIALOG_PROGRESS[job_id]["error"] = str(exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7850)
    args = parser.parse_args()
    _post_note(f"[chatterbox-studio] ready -- Chatterbox listening room on :{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
