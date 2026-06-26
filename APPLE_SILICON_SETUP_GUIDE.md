# Apple Silicon Setup Guide for Faster Qwen3-TTS

This guide covers running Faster Qwen3-TTS natively on Apple Silicon (M1/M2/M3/M4) using the MPS (Metal Performance Shaders) backend.

## Quick Start (Recommended)

### 1. Run the Setup Script

Open Terminal and navigate to the project directory, then run:

```bash
./setup.sh
```

This script will:
- Create a virtual environment with `uv` (fast) or fall back to `venv`
- Install all dependencies (PyTorch with MPS support included by default)
- Pre-download the Qwen3-TTS models
- Generate a placeholder `ref_audio.wav`

### 2. Run the Benchmark

After setup completes, you can run the benchmark:

```bash
./benchmark.sh
```

Additional usage:
```bash
./benchmark.sh 0.6B    # Benchmark only the 0.6B model
./benchmark.sh 1.7B    # Benchmark only the 1.7B model
./benchmark.sh custom  # Benchmark CustomVoice mode
./benchmark.sh both    # Benchmark both (default)
```

## Manual Setup

If you prefer to set up manually:

```bash
# Create virtual environment
uv venv .venv --python 3.10

# Activate it
source .venv/bin/activate

# Install dependencies
uv pip install -e .

# Download models
python -c "from huggingface_hub import snapshot_download; [snapshot_download(f'Qwen/{m}') for m in ['Qwen3-TTS-12Hz-0.6B-Base', 'Qwen3-TTS-12Hz-1.7B-Base']]"
```

## Key Differences from CUDA

| Feature | CUDA (NVIDIA) | MPS (Apple Silicon) |
|---------|---------------|---------------------|
| Device | `cuda` | `mps` |
| dtype | `bfloat16` | `float32` (bfloat16 not supported) |
| Attention | `eager` or flash-attn | `sdpa` (PyTorch scaled dot product) |
| CUDA graphs | Enabled | Disabled (dynamic cache fallback) |
| `torch.compile` | Not used | `mode="reduce-overhead"` for speed |
| flash-attn | Optional | Not available (uses sdpa instead) |
| nano-parakeet | Available | Not available (CUDA-only) |

### What changes automatically

The code detects your device and adjusts at runtime:
- `get_optimal_device("auto")` returns `"mps"` on Apple Silicon
- `device_supports_cuda_graphs("mps")` returns `False`
- Model loads in float32 + sdpa attention on MPS
- `torch.compile(mode="reduce-overhead")` applied before warmup
- All benchmarks use `if torch.cuda.is_available(): torch.cuda.synchronize()` (no-op on MPS)

### What you lose on MPS

- **CUDA graphs**: The fast path is replaced by dynamic cache. Performance is comparable due to MPS queue serialization.
- **nano-parakeet transcription**: The `/transcribe` endpoint in the demo server returns 503 on MPS.
- **flash-attn**: Not available. PyTorch sdpa attention is used instead (still fast).

### What you gain on MPS

- **No CUDA dependency**: Works with standard PyTorch install, no CUDA toolkit needed.
- **`torch.compile` optimization**: Reduces step time ~3x after the first run (cached compilation).
- **Unified memory**: No GPU/CPU memory copies needed — tensors are shared.

## Troubleshooting

### "flash-attn not available"
This is expected on Apple Silicon. Flash Attention requires CUDA. The code falls back to PyTorch's built-in sdpa attention automatically.

### "WARNING: CUDA not available"
The `setup.sh` script prints this when CUDA isn't detected. On macOS this is harmless — MPS is used instead. You can safely ignore this warning.

### First inference is slow
`torch.compile(mode="reduce-overhead")` compiles optimized kernels on the first run. This takes ~30s on M4 but is cached. Subsequent runs use the cached compilation and are ~3x faster.

### "bfloat16 not supported on MPS"
The code handles this automatically — falls back to float32. No action needed.


## Performance Notes

- **M4 achieves RTF ~1.0** (real-time) for both 0.6B and 1.7B models
- **torch.compile** reduces step time from ~300ms to ~100ms after first run
- **Streaming TTFA** is similar to CUDA (~150ms on M4)
- **No CUDA graphs needed** — MPS queue serialization provides equivalent pipelining

## Running the Demo Server

```bash
python demo/server.py --port 7860
```

The server auto-detects MPS. Note that the transcription endpoint (`/transcribe`) is unavailable without CUDA (nano-parakeet is CUDA-only). Voice clone and custom voice modes work fully.
