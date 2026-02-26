# Windows Setup Guide for Faster Qwen3-TTS

The original `setup.sh` script is designed for Unix/Linux systems. This guide provides instructions for setting up and running Faster Qwen3-TTS natively on Windows.

## Quick Start (Recommended)

### 1. Run the Setup Script
Open Command Prompt and navigate to the project directory, then run:

```cmd
setup_windows.bat
```

This script will:
- Detect if `uv` is installed (recommended for 10x faster setup)
- Create a virtual environment (`.venv`)
- Install all dependencies (including PyTorch with CUDA)
- Pre-download the Qwen3-TTS models
- Generate a placeholder `ref_audio.wav`

### 2. Run the Benchmark
After setup completes, you can run the benchmark:

```cmd
benchmark_windows.bat
```

Additional usage for the benchmark:
```cmd
benchmark_windows.bat 0.6B    # Benchmark only the 0.6B model
benchmark_windows.bat 1.7B    # Benchmark only the 1.7B model
benchmark_windows.bat custom  # Benchmark CustomVoice mode
benchmark_windows.bat both    # Benchmark both (default)
```

## Manual Setup

If you prefer to set up manually:

```cmd
# Create virtual environment
python -m venv .venv

# Activate it
.venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -e .

# Install flash-attn (optional, requires compiler)
pip install flash-attn

# Download models
python -c "from huggingface_hub import snapshot_download; [snapshot_download(f'Qwen/{m}') for m in ['Qwen3-TTS-12Hz-0.6B-Base', 'Qwen3-TTS-12Hz-1.7B-Base']]"
```

## Troubleshooting

### CUDA Issues
If `setup_windows.bat` warns that CUDA is not available:
1. Verify you have an NVIDIA GPU and recent drivers.
2. Run `nvidia-smi` to check CUDA support.
3. You may need to manually install the CUDA version of PyTorch:
   ```cmd
   .venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu124
   ```

### Python Not Found
Ensure Python 3.10 or later is installed and added to your system PATH.

## Performance Notes
- **CUDA Required**: For real-time performance, an NVIDIA GPU is required.
- **uv**: Highly recommended for faster installation. Install it via `pip install uv` or from [astral.sh](https://docs.astral.sh/uv/).
