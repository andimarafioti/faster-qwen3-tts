#!/usr/bin/env python3
"""Measure Time to First Audio Chunk (TTFA/TTFT) for Qwen3-TTS with CUDA graphs."""
import torch
import time
import os
import numpy as np
from qwen3_tts_cuda_graphs import Qwen3TTSCudaGraphs

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_SIZE = os.environ.get('MODEL_SIZE', '0.6B')
MODEL_PATH = os.path.join(SCRIPT_DIR, 'models', f'Qwen3-TTS-12Hz-{MODEL_SIZE}-Base')
text = 'The quick brown fox jumps over the lazy dog.'
ref_audio = os.path.join(SCRIPT_DIR, 'ref_audio.wav')
ref_text = 'This is a reference audio sample.'

print(f"=== TTFA Benchmark for {MODEL_SIZE} ===\n")

print("Loading model...")
model = Qwen3TTSCudaGraphs.from_pretrained(
    MODEL_PATH,
    device='cuda',
    dtype=torch.bfloat16,
    attn_implementation='eager',
    max_seq_len=2048,
)

print("Warmup (includes CUDA graph capture)...")
model.generate_voice_clone(
    text=text[:20],
    language="English",
    ref_audio=ref_audio,
    ref_text=ref_text,
    max_new_tokens=5,
)

print("\nMeasuring TTFA (20 runs)...")
ttfa_times = []

for i in range(20):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    
    audio_list, sr = model.generate_voice_clone(
        text=text,
        language="English",
        ref_audio=ref_audio,
        ref_text=ref_text,
        max_new_tokens=1,  # Just first token
    )
    
    torch.cuda.synchronize()
    ttfa_ms = (time.perf_counter() - t0) * 1000
    ttfa_times.append(ttfa_ms)
    
    if (i + 1) % 5 == 0:
        print(f"  Runs {i-3}-{i+1}: {np.mean(ttfa_times[-5:]):.1f}ms ± {np.std(ttfa_times[-5:]):.1f}ms")

print(f"\n=== Results ===")
print(f"Mean TTFA: {np.mean(ttfa_times):.1f}ms")
print(f"Std Dev:   {np.std(ttfa_times):.1f}ms")
print(f"Min:       {np.min(ttfa_times):.1f}ms")
print(f"Max:       {np.max(ttfa_times):.1f}ms")
print(f"Median:    {np.median(ttfa_times):.1f}ms")
print(f"P95:       {np.percentile(ttfa_times, 95):.1f}ms")
print(f"P99:       {np.percentile(ttfa_times, 99):.1f}ms")
