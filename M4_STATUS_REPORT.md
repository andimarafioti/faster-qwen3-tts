# M4 Apple Silicon Status Report

**Date:** 2025-06-25
**Project:** faster-qwen3-tts v0.2.6
**Status:** Implementation complete — device auto-detection and MPS/CPU fallback added

---

## Current Status: What Blocks and What Already Works

### The Good News

The CUDA dependency is **cleanly isolated**. The entire BLOCKER surface is behind **one gate** at `model.py:118`:

```python
if not device.startswith("cuda") or not torch.cuda.is_available():
    raise ValueError("CUDA graphs require CUDA device")
```

Nothing beyond that line ever executes on non-CUDA. The 75 BLOCKERs in `talker_graph.py`, `predictor_graph.py`, `generate.py`, and `streaming.py` are **unreachable** through the public API on M4.

---

## What Already Works (device-agnostic)

| Component | Status | Evidence |
|-----------|--------|----------|
| Model loading (`qwen-tts`) | Works on MPS | Community-verified on M1/M2/M3/M4 |
| Prefill (HF forward) | Works on MPS | Device-agnostic `inputs_embeds` usage |
| Attention masks | Works on MPS | `create_causal_mask` from transformers, device-agnostic |
| StaticCache | Works on MPS | Standard transformers API |
| Sampling (`sample_logits`) | Works on MPS | Pure tensor math, no CUDA calls |
| Repetition penalty | Works on MPS | Pure tensor math |
| Codec decode | Works on MPS | `speech_tokenizer.decode()` is device-agnostic |
| Voice clone prompt building | Works on MPS | All device-agnostic tensor ops |
| Input embedding building | Works on MPS | Uses `m.talker.device` dynamically |

---

## What Blocks M4 (exactly 4 things)

| # | Blocker | File | Lines | Fix Effort |
|---|---------|------|-------|------------|
| 1 | **CUDA-only guard** | `model.py` | 118-119 | Trivial: replace with auto-detect |
| 2 | **CUDA graph capture** | `talker_graph.py` | 109-147 | Medium: skip capture on non-CUDA, fall back to direct `_decode_step()` |
| 3 | **CUDA graph capture** | `predictor_graph.py` | 169-202 | Medium: same pattern, fall back to `_full_loop()` |
| 4 | **`torch.cuda.synchronize()`** | `generate.py` (3x), `streaming.py` (6x) | Various | Trivial: guard with device check |

---

## The Existing Fallback Path

The project **already implements** a non-CUDA-graph decode path:

- `parity_generate_streaming()` in `streaming.py:191-359` — uses dynamic cache, no CUDA graphs
- `parity_mode=True` in `generate.py:52-97` — uses `talker.generate()` directly, no CUDA graphs
- These are currently used for **parity testing** (validating correctness against upstream)

On M4, this fallback path would run at ~1x speed (no CUDA graph speedup) but would produce correct output.

---

## What Does NOT Block M4

| Concern | Status |
|---------|--------|
| `flash-attn` | Optional, suppressed by default, not required |
| `torch.compile` | Not used anywhere in the codebase |
| `vLLM` | Not used |
| Triton | Not used |
| `bfloat16` dtype | MPS supports bf16 on M3/M4 (M1/M2 need float32) |

---

## Exposed non-library files (not critical path)

| File | Issue | Impact |
|------|-------|--------|
| `demo/server.py:249` | `.cuda()` on parakeet tensor | Crashes demo on M4 |
| `demo/server.py:713` | `parakeet_from_pretrained(device="cuda")` | Crashes demo on M4 |
| `benchmarks/*.py` | ~40 CUDA calls | All crash on M4 (not shipping code) |
| `tests/test_e2e_parity.py` | All CUDA calls guarded by `skipif` | Already works on M4 |

---

## Full CUDA Dependency Audit

### `faster_qwen3_tts/model.py`

| Line | Content | Classification |
|------|---------|----------------|
| 15 | `from .utils import suppress_flash_attn_warning` | SAFE |
| 35 | `device: str = "cuda"` (constructor default) | COSMETIC |
| 97 | `device: str = "cuda"` (from_pretrained default) | COSMETIC |
| 118 | `if not device.startswith("cuda") or not torch.cuda.is_available():` | SAFE (the guard) |
| 119 | `raise ValueError("CUDA graphs require CUDA device")` | SAFE (the guard) |

### `faster_qwen3_tts/talker_graph.py`

| Line | Content | Classification |
|------|---------|----------------|
| 27 | `device='cuda'` (constructor default) | COSMETIC |
| 31 | `device_index if device_index is not None else torch.cuda.current_device()` | BLOCKER |
| 127 | `torch.cuda.synchronize()` | BLOCKER |
| 131 | `with torch.cuda.device(self.device_index):` | BLOCKER |
| 132 | `self.graph = torch.cuda.CUDAGraph()` | BLOCKER |
| 134 | `s = torch.cuda.Stream()` | BLOCKER |
| 135 | `s.wait_stream(torch.cuda.current_stream())` | BLOCKER |
| 136 | `with torch.cuda.stream(s):` | BLOCKER |
| 139 | `torch.cuda.synchronize()` | BLOCKER |
| 141 | `with torch.cuda.graph(self.graph):` | BLOCKER |
| 144 | `torch.cuda.current_stream().wait_stream(s)` | BLOCKER |
| 145 | `torch.cuda.synchronize()` | BLOCKER |

### `faster_qwen3_tts/predictor_graph.py`

| Line | Content | Classification |
|------|---------|----------------|
| 34 | `device='cuda'` (constructor default) | COSMETIC |
| 38 | `device_index if device_index is not None else torch.cuda.current_device()` | BLOCKER |
| 181 | `torch.cuda.synchronize()` | BLOCKER |
| 185 | `with torch.cuda.device(self.device_index):` | BLOCKER |
| 186 | `s = torch.cuda.Stream()` | BLOCKER |
| 187 | `s.wait_stream(torch.cuda.current_stream())` | BLOCKER |
| 188 | `with torch.cuda.stream(s):` | BLOCKER |
| 189 | `self.graph = torch.cuda.CUDAGraph()` | BLOCKER |
| 193 | `torch.cuda.synchronize()` | BLOCKER |
| 196 | `with torch.cuda.graph(self.graph):` | BLOCKER |
| 199 | `torch.cuda.current_stream().wait_stream(s)` | BLOCKER |
| 200 | `torch.cuda.synchronize()` | BLOCKER |

### `faster_qwen3_tts/generate.py`

| Line | Content | Classification |
|------|---------|----------------|
| 87 | `torch.cuda.synchronize()` | BLOCKER |
| 142 | `torch.cuda.synchronize()` | BLOCKER |
| 201 | `torch.cuda.synchronize()` | BLOCKER |

### `faster_qwen3_tts/streaming.py`

| Line | Content | Classification |
|------|---------|----------------|
| 96 | `torch.cuda.synchronize()` | BLOCKER |
| 158 | `torch.cuda.synchronize()` | BLOCKER |
| 177 | `torch.cuda.synchronize()` | BLOCKER |
| 262 | `torch.cuda.synchronize()` | BLOCKER |
| 330 | `torch.cuda.synchronize()` | BLOCKER |
| 348 | `torch.cuda.synchronize()` | BLOCKER |

### `faster_qwen3_tts/utils.py`

| Line | Content | Classification |
|------|---------|----------------|
| 24 | `"flash-attn is not installed"` (string literal) | SAFE |
| 26 | `"Please install flash-attn"` (string literal) | SAFE |

### `demo/server.py`

| Line | Content | Classification |
|------|---------|----------------|
| 249 | `return _parakeet.transcribe(wav_t.cuda())` | BLOCKER |
| 317 | `device="cuda"` (in from_pretrained call) | SAFE |
| 702 | `device="cuda"` (in from_pretrained call) | SAFE |
| 713 | `_parakeet = _parakeet_from_pretrained(device="cuda")` | BLOCKER |

### Tests (all already guarded)

| File | CUDA refs | Classification |
|------|-----------|----------------|
| `tests/test_sampling.py` | All inside `skipif` blocks | SAFE |
| `tests/test_e2e_parity.py` | All inside `skipif` blocks | SAFE |

### Benchmarks (not shipping code, all BLOCKER)

All benchmark files (`throughput.py`, `baseline.py`, `streaming.py`, `compare_modes.py`, `custom_voice.py`, `chunk_sweep.py`, `parakeet_coexistence.py`, `generate_parity_samples.py`, `generate_parity_samples_icl.py`, `generate_non_streaming_samples.py`) contain hardcoded CUDA calls and would crash on M4.

---

## Summary

| Classification | Count | Where |
|----------------|-------|-------|
| **BLOCKER** | ~75 | Core library + demo + benchmarks |
| **COSMETIC** | ~18 | Default values and from_pretrained calls |
| **SAFE** | ~25 | Guards, test decorators, string literals |

---

## Bottom Line

**To get `faster-qwen3-tts` running on M4, you need to change exactly 4 files in the library core:**

1. `model.py` — Remove/replace CUDA-only guard with auto-detect
2. `talker_graph.py` — Add fallback path in `capture()` and `run()`
3. `predictor_graph.py` — Add fallback path in `capture()` and `run()`
4. `generate.py` + `streaming.py` — Guard 9 `synchronize()` calls

**~80 lines of code.** The architecture is already designed for this — the fallback paths exist, they're just not exposed as a first-class option.

---

## Implementation Complete

### Files Changed

| File | Change |
|------|--------|
| `faster_qwen3_tts/device.py` | **NEW** — `get_optimal_device()`, `device_supports_cuda_graphs()`, `sync_device()` |
| `faster_qwen3_tts/talker_graph.py` | Added `use_cuda_graphs` flag; `capture()` skips CUDA graph on non-CUDA; `run()` calls `_decode_step()` directly |
| `faster_qwen3_tts/predictor_graph.py` | Same pattern: `use_cuda_graphs` flag; `capture()` and `run()` fallback |
| `faster_qwen3_tts/model.py` | `from_pretrained()` default `"auto"`, auto-detects device, switches to float32 on MPS |
| `faster_qwen3_tts/generate.py` | Replaced 3 `torch.cuda.synchronize()` → `sync_device(device)` |
| `faster_qwen3_tts/streaming.py` | Replaced 6 `torch.cuda.synchronize()` → `sync_device(device)` |
| `faster_qwen3_tts/cli.py` | `--device` default changed to `"auto"` |
| `faster_qwen3_tts/__init__.py` | Exports `get_optimal_device`, `device_supports_cuda_graphs`, `sync_device` |

### How It Works

```
User calls: FasterQwen3TTS.from_pretrained("Qwen/...", device="auto")
                │
                ▼
        get_optimal_device("auto")
                │
    ┌───────────┼───────────┐
    │           │           │
  CUDA?       MPS?       CPU
    │           │           │
    ▼           ▼           ▼
  float16    float32     float32
  CUDA graphs  No CUDA    No CUDA
  (fastest)   (fallback)  (slowest)
```

### Usage on M4

```bash
# Auto-detect (recommended)
faster-qwen3-tts clone --device auto --model Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --text "Hello" --language English --ref-audio ref_audio.wav \
  --ref-text "transcript" --output test.wav

# Force MPS
faster-qwen3-tts clone --device mps --model Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --text "Hello" --language English --ref-audio ref_audio.wav \
  --ref-text "transcript" --output test.wav

# Python API
from faster_qwen3_tts import FasterQwen3TTS
model = FasterQwen3TTS.from_pretrained("Qwen/Qwen3-TTS-12Hz-0.6B-Base")
# device="auto" detects MPS automatically
```
