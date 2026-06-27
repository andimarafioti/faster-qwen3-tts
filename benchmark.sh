#!/bin/bash
# Benchmark Qwen3-TTS with CUDA Graphs
# Usage: ./benchmark.sh [0.6B|1.7B|both|custom]
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

MODEL="${1:-both}"
PY="$DIR/.venv/bin/python"

if [ ! -f "$PY" ]; then
    echo "ERROR: venv not found. Run ./setup.sh first."
    exit 1
fi

# Detect device: CUDA, MPS, or CPU
DEVICE_INFO=""
if $PY -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    GPU_NAME=$($PY -c 'import torch; print(torch.cuda.get_device_name(0))')
    DEVICE_INFO="GPU: $GPU_NAME"
    DEVICE_INFO_LINE="CUDA: $($PY -c 'import torch; print(torch.version.cuda)')"
elif $PY -c "import torch; assert hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()" 2>/dev/null; then
    DEVICE_INFO="Device: Apple Silicon (MPS)"
    DEVICE_INFO_LINE="CUDA graphs: disabled (MPS fallback)"
else
    echo "ERROR: No CUDA or MPS device available."
    exit 1
fi

echo "=== Faster Qwen3-TTS Benchmark ==="
echo "$DEVICE_INFO"
echo "PyTorch: $($PY -c 'import torch; print(torch.__version__)')"
echo "$DEVICE_INFO_LINE"
echo ""

run_model() {
    local size=$1
    echo "--- Benchmarking $size ---"
    MODEL_SIZE="$size" $PY "$DIR/benchmarks/throughput.py"
    echo ""
}

run_custom() {
    local size=$1
    echo "--- Benchmarking $size (CustomVoice) ---"
    MODEL_SIZE="$size" $PY "$DIR/benchmarks/custom_voice.py"
    echo ""
}

case "$MODEL" in
    0.6B) run_model "0.6B" ;;
    1.7B) run_model "1.7B" ;;
    custom)
        run_custom "0.6B"
        run_custom "1.7B"
        ;;
    both)
        run_model "0.6B"
        run_model "1.7B"
        ;;
    *)
        echo "Usage: ./benchmark.sh [0.6B|1.7B|both|custom]"
        exit 1
        ;;
esac

echo "Done. Results saved as bench_results_*.json"
