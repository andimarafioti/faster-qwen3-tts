@echo off
setlocal enabledelayedexpansion
REM Benchmark Qwen3-TTS with CUDA Graphs (Windows)
REM Usage: .\benchmark_windows.bat [0.6B|1.7B|both|custom]

set "DIR=%~dp0"
cd /d "%DIR%"

set "MODEL=%~1"
if "%MODEL%"=="" set "MODEL=both"

set "PY=.\.venv\Scripts\python.exe"

if not exist "!PY!" (
    echo ERROR: venv not found. Run .\setup_windows.bat first.
    pause
    exit /b 1
)

!PY! -c "import torch; assert torch.cuda.is_available()" 2>nul || (
    echo ERROR: PyTorch with CUDA required. Check your venv.
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('!PY! -c "import torch; print(torch.cuda.get_device_name(0))"') do set "GPU_NAME=%%i"
for /f "tokens=*" %%i in ('!PY! -c "import torch; print(torch.__version__)"') do set "PYTORCH_VER=%%i"
for /f "tokens=*" %%i in ('!PY! -c "import torch; print(torch.version.cuda)"') do set "CUDA_VER=%%i"

echo === Faster Qwen3-TTS Benchmark ===
echo GPU: %GPU_NAME%
echo PyTorch: %PYTORCH_VER%
echo CUDA: %CUDA_VER%
echo.

if /i "%MODEL%"=="0.6B" (
    call :run_model 0.6B
) else if /i "%MODEL%"=="1.7B" (
    call :run_model 1.7B
) else if /i "%MODEL%"=="custom" (
    call :run_custom 0.6B
    call :run_custom 1.7B
) else if /i "%MODEL%"=="both" (
    call :run_model 0.6B
    call :run_model 1.7B
) else (
    echo Usage: .\benchmark_windows.bat [0.6B^|1.7B^|both^|custom]
    pause
    exit /b 1
)

echo Done. Results saved as bench_results_*.json
pause
goto :eof

:run_model
set "size=%~1"
echo --- Benchmarking %size% ---
set "MODEL_SIZE=%size%"
!PY! "benchmarks\throughput.py"
echo.
goto :eof

:run_custom
set "size=%~1"
echo --- Benchmarking %size% (CustomVoice) ---
set "MODEL_SIZE=%size%"
!PY! "benchmarks\custom_voice.py"
echo.
goto :eof
