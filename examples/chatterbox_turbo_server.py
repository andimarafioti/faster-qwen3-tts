#!/usr/bin/env python3
"""Minimal Chatterbox Turbo voice-cloning web service for NVIDIA Spark."""
from __future__ import annotations

import argparse
import html
import json
import os
import re
import tempfile
import threading
import time
import urllib.request
import uuid
from pathlib import Path

import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from chatterbox.tts_turbo import ChatterboxTurboTTS


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = Path(os.environ.get("CHATTERBOX_HISTORY_DIR", BASE_DIR / "chatterbox_history"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
NOTES_API_BASE = os.environ.get("NOTES_API_BASE", "http://localhost:9999").rstrip("/")
REFERENCE_PRESETS = {
    "clone_1": BASE_DIR / "ref_audio_3.wav",
    "clone_2": BASE_DIR / "ref_audio_2.wav",
    "clone_3": BASE_DIR / "ref_audio.wav",
}
MODEL = None
MODEL_LOCK = threading.Lock()
GENERATE_LOCK = threading.Lock()
PROGRESS_LOCK = threading.Lock()
PROGRESS: dict[str, dict] = {}


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


def _load_model():
    global MODEL
    if MODEL is None:
        with MODEL_LOCK:
            if MODEL is None:
                _post_note("[chatterbox-turbo/server] loading -- downloading/loading Turbo on GB10")
                MODEL = ChatterboxTurboTTS.from_pretrained(device="cuda")
                _post_note("[chatterbox-turbo/server] ready -- Turbo loaded; web UI available")
    return MODEL


def _history() -> list[dict]:
    items = []
    for path in sorted(OUTPUT_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:100]:
        try:
            items.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return items


def _split_long_text(text: str, max_chars: int = 280) -> list[tuple[str, float]]:
    """Return narration-sized chunks with the pause that should follow each one."""
    paragraphs = [" ".join(part.split()) for part in re.split(r"\n\s*\n", text) if part.strip()]
    chunks: list[tuple[str, float]] = []
    for paragraph in paragraphs:
        sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", paragraph) if part.strip()]
        paragraph_chunks: list[str] = []
        current = ""
        for sentence in sentences:
            pieces = [sentence]
            if len(sentence) > max_chars:
                pieces = []
                piece = ""
                for word in sentence.split():
                    candidate = f"{piece} {word}".strip()
                    if piece and len(candidate) > max_chars:
                        pieces.append(piece)
                        piece = word
                    else:
                        piece = candidate
                if piece:
                    pieces.append(piece)
            for piece in pieces:
                candidate = f"{current} {piece}".strip()
                if current and len(candidate) > max_chars:
                    paragraph_chunks.append(current)
                    current = piece
                else:
                    current = candidate
        if current:
            paragraph_chunks.append(current)
        for index, chunk in enumerate(paragraph_chunks):
            pause = 0.42 if index == len(paragraph_chunks) - 1 else 0.18
            chunks.append((chunk, pause))
    if chunks:
        chunks[-1] = (chunks[-1][0], 0.0)
    return chunks


def _set_progress(job_id: str, **values) -> None:
    with PROGRESS_LOCK:
        PROGRESS[job_id] = {**PROGRESS.get(job_id, {}), **values}
        if len(PROGRESS) > 200:
            PROGRESS.pop(next(iter(PROGRESS)))


app = FastAPI(title="Chatterbox Turbo on Spark")


@app.get("/", response_class=HTMLResponse)
def index():
    rows = "".join(
        f'<li><a href="{html.escape(item["url"])}">{html.escape(item["text"][:80])}</a> '
        f'({item["duration_s"]:.1f}s)</li>' for item in _history()
    ) or "<li>No generations yet.</li>"
    return HTMLResponse(f"""<!doctype html><html><head><meta charset="utf-8"><title>Chatterbox Turbo</title>
<style>body{{font:16px system-ui;max-width:850px;margin:40px auto;padding:0 20px;background:#111;color:#eee}}
textarea,input,select,button{{box-sizing:border-box;width:100%;padding:10px;margin:6px 0;background:#222;color:#eee;border:1px solid #555;border-radius:6px}}
button{{cursor:pointer;background:#3858d8}} label{{display:block;margin-top:14px}} a{{color:#8ab4ff}}</style></head>
<body><h1>Chatterbox Turbo 350M</h1><p>English zero-shot voice cloning. Try tags such as <code>[laugh]</code>, <code>[chuckle]</code>, and <code>[cough]</code>.</p>
<form id="form"><label>Sample voice<select name="preset"><option value="clone_1">Clone 1</option><option value="clone_2">Clone 2</option><option value="clone_3">Clone 3</option><option value="">Custom upload</option></select></label>
<label>Custom reference WAV (optional)<input name="reference" type="file" accept="audio/*"></label>
<label>Text<textarea name="text" rows="7" required>Hello from Chatterbox Turbo [chuckle]. This voice was cloned from your reference recording.</textarea></label>
<label>Temperature<input name="temperature" type="number" min="0.1" max="2" step="0.05" value="0.8"></label><button>Generate WAV</button></form>
<p id="status"></p><progress id="bar" value="0" max="1" style="width:100%;display:none"></progress><audio id="player" controls style="width:100%"></audio><h2>Server history</h2><ul>{rows}</ul>
<script>form.onsubmit=async(e)=>{{e.preventDefault();const jobId=crypto.randomUUID().replaceAll("-","");const data=new FormData(form);data.append("job_id",jobId);bar.style.display="block";bar.value=0;bar.max=1;status.textContent="Preparing voice...";const timer=setInterval(async()=>{{const p=await fetch("/progress/"+jobId).then(r=>r.ok?r.json():null).catch(()=>null);if(!p)return;bar.max=Math.max(1,p.total||1);bar.value=p.completed||0;status.textContent=p.stage==="queued"?"Waiting for generator...":"Generating chunk "+(p.completed||0)+" of "+(p.total||0)+"...";}},350);const r=await fetch("/generate",{{method:"POST",body:data}});clearInterval(timer);const d=await r.json();if(!r.ok){{status.textContent=d.detail||"Error";return}}bar.max=d.chunk_count;bar.value=d.chunk_count;player.src=d.url;player.play();status.textContent="Done: "+d.duration_s.toFixed(1)+"s audio in "+d.elapsed_s.toFixed(1)+"s across "+d.chunk_count+" chunk(s)";}};</script></body></html>""")


@app.get("/status")
def status():
    return {"loaded": MODEL is not None, "model": "Chatterbox Turbo 350M", "device": "cuda"}


@app.get("/progress/{job_id}")
def generation_progress(job_id: str):
    if not job_id.isalnum():
        raise HTTPException(status_code=404, detail="Progress not found")
    with PROGRESS_LOCK:
        progress_state = PROGRESS.get(job_id)
    if progress_state is None:
        raise HTTPException(status_code=404, detail="Progress not found")
    return progress_state


@app.post("/generate")
def generate(
    text: str = Form(...), temperature: float = Form(0.8), preset: str = Form("clone_1"), job_id: str = Form(""),
    reference: UploadFile | None = File(None),
):
    if not text.strip() or len(text) > 5000:
        raise HTTPException(status_code=400, detail="Text must be 1-5000 characters")
    temporary_ref = False
    if reference is not None and reference.filename:
        suffix = Path(reference.filename).suffix or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(reference.file.read())
            ref_path = tmp.name
        temporary_ref = True
    else:
        preset_path = REFERENCE_PRESETS.get(preset)
        if preset_path is None or not preset_path.is_file():
            raise HTTPException(status_code=400, detail="Choose a sample voice or upload a reference WAV")
        ref_path = str(preset_path)
    try:
        started = time.perf_counter()
        chunks = _split_long_text(text.strip())
        job_id = job_id if job_id.isalnum() else uuid.uuid4().hex
        _set_progress(job_id, stage="queued", completed=0, total=len(chunks))
        with GENERATE_LOCK:
            _set_progress(job_id, stage="conditioning", completed=0, total=len(chunks))
            model = _load_model()
            model.prepare_conditionals(ref_path)
            sample_rate = int(model.sr)
            audio_parts = []
            _post_note(f"[chatterbox-turbo/generate] running -- long-form request split into {len(chunks)} chunks")
            for index, (chunk, pause_s) in enumerate(chunks):
                _set_progress(job_id, stage="generating", completed=index, total=len(chunks))
                wav = model.generate(chunk, temperature=float(temperature)).reshape(-1)
                audio_parts.append(wav)
                if pause_s:
                    audio_parts.append(torch.zeros(round(sample_rate * pause_s), dtype=wav.dtype))
                if len(chunks) > 3 and index + 1 == len(chunks) // 2:
                    _post_note(f"[chatterbox-turbo/generate] running -- {index + 1}/{len(chunks)} chunks complete")
                _set_progress(job_id, stage="generating", completed=index + 1, total=len(chunks))
            wav = torch.cat(audio_parts)
        elapsed = time.perf_counter() - started
        audio = wav.detach().float().cpu().numpy()
        audio_id = uuid.uuid4().hex
        output_path = OUTPUT_DIR / f"{audio_id}.wav"
        sf.write(output_path, audio, sample_rate, subtype="PCM_16")
        item = {
            "id": audio_id, "url": f"/audio/{audio_id}.wav", "text": text.strip(),
            "duration_s": len(audio) / sample_rate, "elapsed_s": elapsed,
            "timestamp": int(time.time() * 1000), "temperature": float(temperature),
            "chunk_count": len(chunks),
        }
        (OUTPUT_DIR / f"{audio_id}.json").write_text(json.dumps(item, indent=2), encoding="utf-8")
        _set_progress(job_id, stage="done", completed=len(chunks), total=len(chunks))
        _post_note(
            f"[chatterbox-turbo/generate] complete -- {len(chunks)} chunks, "
            f"{item['duration_s']:.1f}s audio in {elapsed:.1f}s"
        )
        return item
    finally:
        if temporary_ref:
            Path(ref_path).unlink(missing_ok=True)


@app.get("/audio/{audio_id}.wav")
def audio(audio_id: str):
    if not audio_id.isalnum():
        raise HTTPException(status_code=404, detail="Audio not found")
    path = OUTPUT_DIR / f"{audio_id}.wav"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Audio not found")
    return FileResponse(path, media_type="audio/wav", filename=path.name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7862)
    parser.add_argument("--no-preload", action="store_true")
    args = parser.parse_args()
    if not args.no_preload:
        _load_model()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
