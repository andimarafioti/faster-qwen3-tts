#!/usr/bin/env python3
"""Benchmark throughput: CUDA graphs using the FasterQwen3TTS wrapper."""
import torch
import time
import os
import numpy as np
import soundfile as sf
from faster_qwen3_tts import FasterQwen3TTS

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_SIZE = os.environ.get('MODEL_SIZE', '0.6B')
MODEL_ID = f'Qwen/Qwen3-TTS-12Hz-{MODEL_SIZE}-Base'
text = "Ladies and gentlemen, I have just been informed that this speech is being generated faster than I can speak it. The robots have officially won. Please remain calm."
ref_audio = os.path.join(PROJECT_DIR, 'ref_audio.wav')
ref_text = "I'm confused why some people have super short timelines, yet at the same time are bullish on scaling up reinforcement learning atop LLMs. If we're actually close to a human-like learner, then this whole approach of training on verifiable outcomes."

print("Loading model...")
model = FasterQwen3TTS.from_pretrained(
    MODEL_ID,
    device='cuda',
    dtype=torch.bfloat16,
    attn_implementation='eager',
    max_seq_len=2048,
)

print("\nWarmup run (includes CUDA graph capture)...")
start = time.perf_counter()
audio_list, sr = model.generate_voice_clone(
    text=text[:50],  # Short warmup
    language="English",
    ref_audio=ref_audio,
    ref_text=ref_text,
    max_new_tokens=20,
)
warmup_time = time.perf_counter() - start
print(f"Warmup: {warmup_time:.2f}s")

# TTFA (Time to First Audio) via streaming
CHUNK_SIZES = [4, 8, 12]
PRIMARY_CHUNK_SIZE = 8
ttfa_by_chunk = {}

print("\nMeasuring streaming TTFA (5 runs per chunk size)...")
for chunk_size in CHUNK_SIZES:
    ttfa_results = []
    for i in range(5):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        gen = model.generate_voice_clone_streaming(
            text=text,
            language="English",
            ref_audio=ref_audio,
            ref_text=ref_text,
            chunk_size=chunk_size,
        )
        first_chunk, sr, timing = next(gen)
        torch.cuda.synchronize()
        ttfa_ms = (time.perf_counter() - t0) * 1000
        ttfa_results.append(ttfa_ms)
        gen.close()
        if chunk_size == PRIMARY_CHUNK_SIZE:
            print(f"  Run {i+1}: {ttfa_ms:.1f}ms")

    mean = np.mean(ttfa_results)
    std = np.std(ttfa_results)
    ttfa_by_chunk[chunk_size] = {'mean': mean, 'std': std}
    marker = " <<" if chunk_size == PRIMARY_CHUNK_SIZE else ""
    print(f"  chunk_size={chunk_size:2d}: TTFA={mean:.0f}ms ± {std:.0f}ms{marker}")

ttfa_mean = ttfa_by_chunk[PRIMARY_CHUNK_SIZE]['mean']
ttfa_std = ttfa_by_chunk[PRIMARY_CHUNK_SIZE]['std']


def measure_streaming_ttfa(parity_mode: bool, chunk_size: int, runs: int = 5):
    results = []
    for _ in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        gen = model.generate_voice_clone_streaming(
            text=text,
            language="English",
            ref_audio=ref_audio,
            ref_text=ref_text,
            chunk_size=chunk_size,
            parity_mode=parity_mode,
        )
        _first_chunk, _sr, _timing = next(gen)
        torch.cuda.synchronize()
        results.append((time.perf_counter() - t0) * 1000)
        gen.close()
    return float(np.mean(results)), float(np.std(results))


def measure_streaming_rtf(parity_mode: bool, chunk_size: int, runs: int = 3):
    rtfs = []
    for _ in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        chunks = []
        sr = None
        for audio_chunk, sr, _timing in model.generate_voice_clone_streaming(
            text=text,
            language="English",
            ref_audio=ref_audio,
            ref_text=ref_text,
            chunk_size=chunk_size,
            parity_mode=parity_mode,
        ):
            chunks.append(audio_chunk)
        torch.cuda.synchronize()
        total_time = time.perf_counter() - t0
        if chunks and sr is not None:
            audio = np.concatenate(chunks)
            audio_duration = len(audio) / sr
            rtfs.append(audio_duration / total_time if total_time > 0 else 0.0)
    return float(np.mean(rtfs)) if rtfs else 0.0, float(np.std(rtfs)) if rtfs else 0.0


def generate_streaming_audio(parity_mode: bool, chunk_size: int):
    chunks = []
    sr = None
    for audio_chunk, sr, _timing in model.generate_voice_clone_streaming(
        text=text,
        language="English",
        ref_audio=ref_audio,
        ref_text=ref_text,
        chunk_size=chunk_size,
        parity_mode=parity_mode,
    ):
        chunks.append(audio_chunk)
    if chunks and sr is not None:
        return np.concatenate(chunks), sr
    return None, None


# Streaming baseline (dynamic cache) vs fast path (CUDA graphs)
print("\nStreaming baseline vs fast path (chunk_size=8)...")
baseline_ttfa_mean, baseline_ttfa_std = measure_streaming_ttfa(parity_mode=True, chunk_size=PRIMARY_CHUNK_SIZE)
fast_ttfa_mean, fast_ttfa_std = measure_streaming_ttfa(parity_mode=False, chunk_size=PRIMARY_CHUNK_SIZE)
baseline_rtf_mean, baseline_rtf_std = measure_streaming_rtf(parity_mode=True, chunk_size=PRIMARY_CHUNK_SIZE)
fast_rtf_mean, fast_rtf_std = measure_streaming_rtf(parity_mode=False, chunk_size=PRIMARY_CHUNK_SIZE)

print(
    f"  baseline: TTFA={baseline_ttfa_mean:.0f}ms ± {baseline_ttfa_std:.0f}ms, "
    f"RTF={baseline_rtf_mean:.3f} ± {baseline_rtf_std:.3f}"
)
print(
    f"  fast:     TTFA={fast_ttfa_mean:.0f}ms ± {fast_ttfa_std:.0f}ms, "
    f"RTF={fast_rtf_mean:.3f} ± {fast_rtf_std:.3f}"
)

# Save audio sample from fast streaming path
try:
    audio, sr = generate_streaming_audio(parity_mode=False, chunk_size=PRIMARY_CHUNK_SIZE)
    if audio is not None and sr is not None:
        out_wav = os.path.join(PROJECT_DIR, f'sample_{MODEL_SIZE}.wav')
        sf.write(out_wav, audio, sr)
        print(f"\nSaved sample audio to {out_wav}")
except Exception as e:
    print(f"Audio save failed: {e}")

# Save results as JSON
import json
import subprocess
gpu_name = "Unknown"
try:
    out = subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                                  stderr=subprocess.DEVNULL, text=True)
    gpu_name = out.strip().split('\n')[0].replace(' ', '_')
except:
    pass

bench_data = {
    'model': MODEL_SIZE,
    'gpu': gpu_name,
    'ttfa_ms': ttfa_mean,
    'ttfa_std_ms': ttfa_std,
    'streaming_baseline_ttfa_ms': baseline_ttfa_mean,
    'streaming_baseline_ttfa_std_ms': baseline_ttfa_std,
    'streaming_baseline_rtf': baseline_rtf_mean,
    'streaming_baseline_rtf_std': baseline_rtf_std,
    'streaming_fast_ttfa_ms': fast_ttfa_mean,
    'streaming_fast_ttfa_std_ms': fast_ttfa_std,
    'streaming_fast_rtf': fast_rtf_mean,
    'streaming_fast_rtf_std': fast_rtf_std,
    'ttfa_by_chunk_size': {str(k): v for k, v in ttfa_by_chunk.items()},
}

json_path = f'bench_results_{gpu_name}.json'
with open(json_path, 'r+' if os.path.exists(json_path) else 'w') as f:
    if os.path.exists(json_path) and os.path.getsize(json_path) > 0:
        f.seek(0)
        try:
            data = json.load(f)
        except:
            data = {}
    else:
        data = {}
    data[MODEL_SIZE] = bench_data
    f.seek(0)
    f.truncate()
    json.dump(data, f, indent=2)

print(f"Results saved to {json_path}")
