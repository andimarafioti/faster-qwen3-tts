#!/usr/bin/env python3
"""
Benchmark: does nano-parakeet coexistence slow down Qwen3-TTS?

Tests four conditions:
  A) TTS alone (baseline)
  B) Parakeet on GPU, then TTS (shared VRAM)
  C) Parakeet transcription run 2s before TTS (no offload)
  D) Parakeet offloaded to CPU before TTS
  E) Parakeet on GPU + 2s pause + offload to CPU before TTS

Run: python benchmarks/parakeet_coexistence.py
"""
import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_SIZE  = os.environ.get("MODEL_SIZE", "0.6B")
MODEL_ID    = f"Qwen/Qwen3-TTS-12Hz-{MODEL_SIZE}-Base"
REF_AUDIO   = os.path.join(PROJECT_DIR, "ref_audio.wav")
REF_TEXT    = "I'm confused why some people have super short timelines, yet at the same time are bullish on scaling up reinforcement learning atop LLMs."
TTS_TEXT    = "Ladies and gentlemen, I have just been informed that this speech is being generated faster than I can speak it. The robots have officially won. Please remain calm."


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def vram_gb():
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / 1e9


def run_tts_bench(model, n_runs=5, max_new_tokens=128, label=""):
    """Run TTS n_runs times, return mean ms/step and RTF."""
    rtfs = []
    for i in range(n_runs):
        torch.cuda.synchronize()
        audio_list, sr = model.generate_voice_clone(
            text=TTS_TEXT,
            language="English",
            ref_audio=REF_AUDIO,
            ref_text=REF_TEXT,
            max_new_tokens=max_new_tokens,
        )
        torch.cuda.synchronize()
        # Grab timing from the last generate call via logging or re-derive from audio
        # We measure wall time ourselves
    # Do it properly with timing
    step_times_ms = []
    rtfs = []
    for i in range(n_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        audio_list, sr = model.generate_voice_clone(
            text=TTS_TEXT,
            language="English",
            ref_audio=REF_AUDIO,
            ref_text=REF_TEXT,
            max_new_tokens=max_new_tokens,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        audio_dur = len(audio_list[0]) / sr
        n_steps = int(audio_dur * 12)  # 12 Hz codec
        ms_per_step = elapsed * 1000 / n_steps if n_steps > 0 else 0
        rtf = audio_dur / elapsed if elapsed > 0 else 0
        step_times_ms.append(ms_per_step)
        rtfs.append(rtf)
        print(f"    [{label}] run {i+1}: {ms_per_step:.1f}ms/step, RTF={rtf:.2f}, "
              f"audio={audio_dur:.1f}s, elapsed={elapsed:.2f}s")
    return np.mean(step_times_ms), np.std(step_times_ms), np.mean(rtfs)


def run_tts_step_bench(model, n_runs=5, max_new_tokens=128, label=""):
    """Benchmark using the generate timing dict (more precise step timing)."""
    rtfs = []
    for i in range(n_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        audio_list, sr = model.generate_voice_clone(
            text=TTS_TEXT,
            language="English",
            ref_audio=REF_AUDIO,
            ref_text=REF_TEXT,
            max_new_tokens=max_new_tokens,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        audio_dur = len(audio_list[0]) / sr
        rtf = audio_dur / elapsed if elapsed > 0 else 0
        rtfs.append(rtf)
    return np.mean(rtfs), np.std(rtfs)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    from faster_qwen3_tts import FasterQwen3TTS

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"TTS model: {MODEL_ID}")
    print()

    # ── Load TTS ──────────────────────────────────────────────────────────────
    print("Loading TTS model...")
    tts = FasterQwen3TTS.from_pretrained(
        MODEL_ID,
        device="cuda",
        dtype=torch.bfloat16,
        attn_implementation="eager",
        max_seq_len=2048,
    )
    print(f"TTS loaded. VRAM: {vram_gb():.2f} GB")

    # Warmup (captures CUDA graphs)
    print("Warming up TTS (CUDA graph capture)...")
    tts.generate_voice_clone(
        text=TTS_TEXT[:50],
        language="English",
        ref_audio=REF_AUDIO,
        ref_text=REF_TEXT,
        max_new_tokens=20,
    )
    torch.cuda.synchronize()
    print(f"Warmup done. VRAM: {vram_gb():.2f} GB")

    N_RUNS = 5
    MAX_TOKENS = 120  # ~10s of audio, fast enough for multiple runs

    results = {}

    # ── A: TTS alone baseline ─────────────────────────────────────────────────
    print("\n" + "="*60)
    print("A) TTS alone (baseline)")
    print("="*60)
    step_ms, step_std, rtf = run_tts_bench(tts, N_RUNS, MAX_TOKENS, label="A-alone")
    results["A_alone"] = (step_ms, step_std, rtf)
    print(f"  => mean {step_ms:.1f}ms/step ±{step_std:.1f}, RTF={rtf:.2f}")
    vram_tts_only = vram_gb()
    print(f"  VRAM: {vram_tts_only:.2f} GB")

    # ── Load Parakeet ─────────────────────────────────────────────────────────
    print("\nLoading nano-parakeet on GPU...")
    from nano_parakeet import from_pretrained as parakeet_from_pretrained
    parakeet = parakeet_from_pretrained(device="cuda")
    torch.cuda.synchronize()
    vram_both = vram_gb()
    print(f"Parakeet loaded. VRAM: {vram_both:.2f} GB "
          f"(+{vram_both - vram_tts_only:.2f} GB for parakeet)")

    print("Warming up Parakeet...")
    parakeet.warmup(duration_s=3.0)
    torch.cuda.synchronize()
    print(f"Parakeet warmed up. VRAM: {vram_gb():.2f} GB")

    # Run a quick transcription on ref audio so parakeet is "used"
    import soundfile as sf
    ref_wav, ref_sr = sf.read(REF_AUDIO, dtype="float32")
    if ref_wav.ndim > 1:
        ref_wav = ref_wav.mean(axis=1)
    # Resample to 16kHz if needed (parakeet needs 16kHz)
    if ref_sr != 16000:
        import torchaudio
        ref_wav_t = torch.from_numpy(ref_wav).unsqueeze(0)
        ref_wav_t = torchaudio.functional.resample(ref_wav_t, ref_sr, 16000)
        ref_wav = ref_wav_t.squeeze().numpy()
    ref_wav_tensor = torch.from_numpy(ref_wav).cuda()

    print("Running parakeet transcription (warmup)...")
    transcript = parakeet.transcribe(ref_wav_tensor)
    print(f"  Transcript: '{transcript[:60]}...'")

    # ── B: Parakeet on GPU, TTS runs (no recent parakeet call) ────────────────
    print("\n" + "="*60)
    print("B) Parakeet on GPU (idle), TTS runs")
    print("="*60)
    step_ms, step_std, rtf = run_tts_bench(tts, N_RUNS, MAX_TOKENS, label="B-parakeet-idle")
    results["B_parakeet_idle"] = (step_ms, step_std, rtf)
    print(f"  => mean {step_ms:.1f}ms/step ±{step_std:.1f}, RTF={rtf:.2f}")
    print(f"  VRAM: {vram_gb():.2f} GB")

    # ── C: Parakeet transcription 2s before TTS (no offload) ──────────────────
    print("\n" + "="*60)
    print("C) Parakeet transcription, 2s pause, then TTS")
    print("="*60)
    step_times_ms = []
    rtfs_c = []
    for i in range(N_RUNS):
        # Run parakeet
        _ = parakeet.transcribe(ref_wav_tensor)
        torch.cuda.synchronize()
        time.sleep(2.0)

        t0 = time.perf_counter()
        audio_list, sr = tts.generate_voice_clone(
            text=TTS_TEXT,
            language="English",
            ref_audio=REF_AUDIO,
            ref_text=REF_TEXT,
            max_new_tokens=MAX_TOKENS,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        audio_dur = len(audio_list[0]) / sr
        n_steps = int(audio_dur * 12)
        ms_per_step = elapsed * 1000 / n_steps if n_steps > 0 else 0
        rtf = audio_dur / elapsed if elapsed > 0 else 0
        step_times_ms.append(ms_per_step)
        rtfs_c.append(rtf)
        print(f"    [C-2s-pause] run {i+1}: {ms_per_step:.1f}ms/step, RTF={rtf:.2f}")
    results["C_2s_pause"] = (np.mean(step_times_ms), np.std(step_times_ms), np.mean(rtfs_c))
    print(f"  => mean {np.mean(step_times_ms):.1f}ms/step ±{np.std(step_times_ms):.1f}, RTF={np.mean(rtfs_c):.2f}")

    # ── D: Parakeet offloaded to CPU before TTS ───────────────────────────────
    print("\n" + "="*60)
    print("D) Parakeet transcription, then offload to CPU, then TTS")
    print("="*60)
    step_times_ms = []
    rtfs_d = []
    for i in range(N_RUNS):
        # Run parakeet on GPU
        _ = parakeet.transcribe(ref_wav_tensor)
        torch.cuda.synchronize()

        # Offload parakeet to CPU
        parakeet.cpu()
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        audio_list, sr = tts.generate_voice_clone(
            text=TTS_TEXT,
            language="English",
            ref_audio=REF_AUDIO,
            ref_text=REF_TEXT,
            max_new_tokens=MAX_TOKENS,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        audio_dur = len(audio_list[0]) / sr
        n_steps = int(audio_dur * 12)
        ms_per_step = elapsed * 1000 / n_steps if n_steps > 0 else 0
        rtf = audio_dur / elapsed if elapsed > 0 else 0
        step_times_ms.append(ms_per_step)
        rtfs_d.append(rtf)
        print(f"    [D-cpu-offload] run {i+1}: {ms_per_step:.1f}ms/step, RTF={rtf:.2f}")

        # Move back to GPU for next iteration
        parakeet.cuda()
        torch.cuda.synchronize()
    results["D_cpu_offload"] = (np.mean(step_times_ms), np.std(step_times_ms), np.mean(rtfs_d))
    print(f"  => mean {np.mean(step_times_ms):.1f}ms/step ±{np.std(step_times_ms):.1f}, RTF={np.mean(rtfs_d):.2f}")

    # ── E: Parakeet on GPU + 2s pause + offload, then TTS ────────────────────
    print("\n" + "="*60)
    print("E) Parakeet transcription, 2s pause, offload, then TTS")
    print("="*60)
    step_times_ms = []
    rtfs_e = []
    for i in range(N_RUNS):
        _ = parakeet.transcribe(ref_wav_tensor)
        torch.cuda.synchronize()
        time.sleep(2.0)

        parakeet.cpu()
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        audio_list, sr = tts.generate_voice_clone(
            text=TTS_TEXT,
            language="English",
            ref_audio=REF_AUDIO,
            ref_text=REF_TEXT,
            max_new_tokens=MAX_TOKENS,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        audio_dur = len(audio_list[0]) / sr
        n_steps = int(audio_dur * 12)
        ms_per_step = elapsed * 1000 / n_steps if n_steps > 0 else 0
        rtf = audio_dur / elapsed if elapsed > 0 else 0
        step_times_ms.append(ms_per_step)
        rtfs_e.append(rtf)
        print(f"    [E-2s+offload] run {i+1}: {ms_per_step:.1f}ms/step, RTF={rtf:.2f}")

        parakeet.cuda()
        torch.cuda.synchronize()
    results["E_2s_offload"] = (np.mean(step_times_ms), np.std(step_times_ms), np.mean(rtfs_e))
    print(f"  => mean {np.mean(step_times_ms):.1f}ms/step ±{np.std(step_times_ms):.1f}, RTF={np.mean(rtfs_e):.2f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    baseline_ms = results["A_alone"][0]
    baseline_rtf = results["A_alone"][2]
    print(f"{'Condition':<40} {'ms/step':>8} {'±':>6} {'RTF':>6} {'vs baseline':>12}")
    print("-"*75)
    labels = {
        "A_alone":         "A) TTS alone (baseline)",
        "B_parakeet_idle": "B) Parakeet idle on GPU",
        "C_2s_pause":      "C) Parakeet → 2s pause → TTS",
        "D_cpu_offload":   "D) Parakeet → CPU offload → TTS",
        "E_2s_offload":    "E) Parakeet → 2s + CPU offload → TTS",
    }
    for key, label in labels.items():
        ms, std, rtf = results[key]
        delta = (ms - baseline_ms) / baseline_ms * 100
        sign = "+" if delta >= 0 else ""
        print(f"  {label:<38} {ms:>8.1f} {std:>6.1f} {rtf:>6.2f} {sign}{delta:>+10.1f}%")
    print()
    print(f"Baseline: {baseline_ms:.1f}ms/step, RTF={baseline_rtf:.2f}")


if __name__ == "__main__":
    main()
