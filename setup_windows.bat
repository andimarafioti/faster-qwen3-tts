@echo off
setlocal enabledelayedexpansion
REM Setup script for faster-qwen3-tts (Windows)
REM Creates a venv (prefers uv), installs deps, and downloads models

set "DIR=%~dp0"
cd /d "%DIR%"

echo === Faster Qwen3-TTS Setup ===

REM Check for uv
where uv >nul 2>&1
if %errorlevel% equ 0 (
    set "HAS_UV=1"
    echo Found uv, using it for faster setup.
) else (
    set "HAS_UV=0"
    echo uv not found, falling back to standard pip.
)

REM Check Python is available if no uv
if !HAS_UV! equ 0 (
    where python >nul 2>&1
    if %errorlevel% neq 0 (
        echo ERROR: Python not found. install it or uv: https://docs.astral.sh/uv/getting-started/installation/
        pause
        exit /b 1
    )
)

REM Create venv + install deps (skip if venv already exists)
if exist ".venv\Scripts\python.exe" (
    echo Venv already exists, skipping install. Delete .venv to force reinstall.
) else (
    echo Creating venv and installing dependencies...
    if !HAS_UV! equ 1 (
        uv venv .venv --python 3.10
        echo Installing PyTorch with CUDA support...
        uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 --python .venv\Scripts\python.exe
        uv pip install -e . --python .venv\Scripts\python.exe
        
        echo Attempting to install flash-attn ^(optional^)...
        uv pip install flash-attn --python .venv\Scripts\python.exe 2>nul && echo   flash-attn installed || echo   flash-attn not available ^(ok, will use manual attention^)
    ) else (
        python -m venv .venv
        call .venv\Scripts\activate.bat
        python -m pip install --upgrade pip
        echo Installing PyTorch with CUDA support...
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
        pip install -e .
        
        echo Attempting to install flash-attn ^(optional^)...
        pip install flash-attn 2>nul && echo   flash-attn installed || echo   flash-attn not available ^(ok, will use manual attention^)
        call deactivate
    )
)

REM Verify CUDA
echo.
.venv\Scripts\python.exe -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}')" || (
    echo.
    echo WARNING: CUDA not available. You may need to install a CUDA-enabled PyTorch wheel.
    echo   .venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu124
)

REM Pre-download models to HuggingFace cache
echo.
echo Pre-downloading models to HuggingFace cache...
.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; [snapshot_download(f'Qwen/{m}') for m in ['Qwen3-TTS-12Hz-0.6B-Base', 'Qwen3-TTS-12Hz-1.7B-Base']]"

REM Generate ref audio if missing
if not exist "ref_audio.wav" (
    echo.
    echo Generating placeholder reference audio...
    .venv\Scripts\python.exe -c "import numpy as np, soundfile as sf; sr = 16000; t = np.linspace(0, 1.0, sr, dtype=np.float32); audio = 0.3 * np.sin(2 * 3.14159 * 440 * t); sf.write('ref_audio.wav', audio, sr)"
    echo   Generated placeholder ref_audio.wav ^(replace with real speech for best quality^)
)

echo.
echo === Setup complete ===
echo Activate the venv: .venv\Scripts\activate
echo Run benchmark:     .\benchmark_windows.bat
echo.
pause
