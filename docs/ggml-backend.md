# qwentts.cpp GGML Backend

This repo keeps `faster-qwen3-tts` as the user-facing package and adds
`qwentts-cpp-python` as an optional native runtime package.

## Package Layout

```text
faster-qwen3-tts
  Existing Torch/CUDA-graph backend
  Optional GGML adapter in faster_qwen3_tts.ggml_backend

qwentts-cpp-python
  Pure Python ctypes wrapper over qwentts.cpp's C ABI
  Platform wheel packaging for libqwen + libggml

qwentts.cpp
  Pascal's C++/GGML implementation, built separately with CMake
```

The main package stays installable without native binaries:

```bash
pip install faster-qwen3-tts
```

The GGML backend is opt-in. By default this installs the PyPI
`qwentts-cpp-python` wheel, currently the CUDA 12.8 build:

```bash
pip install "faster-qwen3-tts[ggml]"
```

Install a backend-specific wrapper wheel first when the PyPI CUDA 12.8 wheel is
not the right runtime for the machine, then install this package as usual. Use
the Hugging Face `+cu128` wheel for Ubuntu 22.04 / older Linux hosts that need
the `manylinux_2_35` CUDA 12.8 build.

```bash
# Ubuntu 22.04 / older Linux with CUDA 12.8
pip install "qwentts-cpp-python==0.3.1+cu128" \
  -f https://huggingface.co/datasets/andito/qwentts-cpp-python-wheels/tree/main/whl/cu128

# CUDA 13 / DGX Spark
pip install "qwentts-cpp-python==0.3.1+cu130" \
  -f https://huggingface.co/datasets/andito/qwentts-cpp-python-wheels/tree/main/whl/cu130

pip install "faster-qwen3-tts[ggml]"
```

The same wheel index also has `0.3.1+cu124`, `0.3.1+cu128`, and `0.3.1+cpu`
variants.

For local wrapper development, clone the wrapper repo beside this checkout and
install it in editable mode:

```bash
git clone https://github.com/andimarafioti/qwentts-cpp-python ../qwentts-cpp-python
pip install -e ../qwentts-cpp-python
```

Development build with local native libraries:

```bash
cd ../qwentts-cpp-python
python scripts/build_native.py --source /path/to/qwentts.cpp --backend cuda --clean
pip install -e .
```

## Python Usage

```python
from faster_qwen3_tts import FasterQwen3TTS

model = FasterQwen3TTS.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    backend="ggml",
    quant="BF16",
)

audio_list, sr = model.generate_voice_design(
    text="Welcome to the show.",
    instruct="Warm, confident narrator with slight British accent",
    language="English",
)
```

Local GGUF paths are supported:

```python
model = FasterQwen3TTS.from_pretrained(
    "unused",
    backend="ggml",
    gguf_talker_path="qwen-talker-1.7b-voicedesign-BF16.gguf",
    gguf_codec_path="qwen-tokenizer-12hz-BF16.gguf",
    qwentts_library_path="/path/to/libqwen.so",
)
```

CLI usage:

```bash
faster-qwen3-tts --backend ggml --quant BF16 design \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --instruct "Warm, confident narrator" \
  --text "Welcome to the show." \
  --language English \
  --output out.wav
```

Raw reference audio is cached automatically after the first base voice-clone
request. The adapter stores qwentts-compatible `.spk` and `.rvq` latents under
`~/.cache/faster-qwen3-tts/qwentts_refs` by default, or under
`FQWEN3TTS_QWENTTS_REF_CACHE_DIR` / `--qwentts-ref-cache-dir` when set.

Precomputed qwentts.cpp references are also supported for base voice cloning:

```python
audio_list, sr = model.generate_voice_clone(
    text="Cached speaker and RVQ latents avoid reference audio encoding.",
    language="English",
    ref_spk="freeman.spk",
    ref_rvq="freeman.rvq",
    ref_text="The transcript for the cached reference audio.",
)
```

Use `ref_spk` by itself for speaker-only conditioning. Use `ref_spk` +
`ref_rvq` + `ref_text` for cached ICL conditioning. Raw `ref_audio` and
explicit cached references are mutually exclusive.

## Current ABI Gaps

The qwentts.cpp C ABI is already enough for buffered and streaming
synthesis, voice cloning, CustomVoice, and VoiceDesign. These gaps remain
before treating the backend as full parity:

- no `non_streaming_mode` switch; requesting
  `non_streaming_mode=False` emits a warning because qwentts.cpp ignores
  that step-by-step text-feed mode and uses its native prompt layout
- base-model `instruct` is rejected by qwentts.cpp
- KV-cache length is fixed in qwentts.cpp

The public GGUF model repo used by the resolver is
`Serveurperso/Qwen3-TTS-GGUF`.

## TTFA Profiling

The GGML adapter attaches a `ggml_profile` snapshot to the first streamed
chunk's `timing` dict. It includes Python-visible boundaries such as ctypes
parameter packing, lock wait, native `qt_synthesize()` entry, first native
audio callback, callback copy/queue cost, and first Python yield.

Run the focused diagnostic with:

```bash
python benchmarks/profile_ggml_ttfa.py \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --mode custom \
  --speaker aiden \
  --quant BF16 \
  --cache-dir .cache/qwentts \
  --local-files-only \
  --chunk-sizes 2,4,8,16
```

With a `libqwen` that includes native profiling markers, the same command also
prints native phase splits for prompt build, first talker prefill, first code
predictor step, first emit entry, and first codec decode. Use `--warmup-runs 0`
to inspect true first-request latency; use the default warmup to inspect
steady-state server latency after CUDA/GGML graphs and prefix caches are hot.

For a direct Torch-vs-GGML comparison, use the benchmark entry point:

```bash
MODEL_SIZE=1.7B MODE=custom QUANT=BF16 ./benchmark.sh backends
MODEL_SIZE=0.6B QUANT=BF16 ./benchmark.sh backend-base
```

The legacy CUDA-graph-only benchmarks still run with `./benchmark.sh`.

## Wheel Distribution

`qwentts-cpp-python==0.3.1` is published on PyPI. The PyPI package is the
default CUDA 12.8 wheel used by `pip install "faster-qwen3-tts[ggml]"`.
Additional local-version wheels are hosted on Hugging Face Hub:

```bash
pip install "qwentts-cpp-python==0.3.1+cpu" \
  -f https://huggingface.co/datasets/andito/qwentts-cpp-python-wheels/tree/main/whl/cpu

pip install "qwentts-cpp-python==0.3.1+cu124" \
  -f https://huggingface.co/datasets/andito/qwentts-cpp-python-wheels/tree/main/whl/cu124

pip install "qwentts-cpp-python==0.3.1+cu128" \
  -f https://huggingface.co/datasets/andito/qwentts-cpp-python-wheels/tree/main/whl/cu128

pip install "qwentts-cpp-python==0.3.1+cu130" \
  -f https://huggingface.co/datasets/andito/qwentts-cpp-python-wheels/tree/main/whl/cu130
```

Hugging Face file hosting is used as a `--find-links` wheelhouse rather than a
PyTorch-style package index. For CUDA 13 / DGX Spark, install the `+cu130`
wheel before installing `faster-qwen3-tts[ggml]`. For Ubuntu 22.04 / older
Linux hosts, install `0.3.1+cu128` from the Hugging Face wheelhouse so pip can
select the `manylinux_2_35` CUDA 12.8 wheel.

For publishing new wrapper wheels, use the manual GitHub Actions workflow:

```text
andimarafioti/qwentts-cpp-python:.github/workflows/publish-hf-wheels.yml
```

The workflow builds Linux x86_64 and Linux aarch64 wheels for CPU, CUDA 12.4,
CUDA 12.8, and CUDA 13.0, then uploads static wheel index pages to the HF
dataset.

Local development builds still use the wrapper build script:

```bash
cd ../qwentts-cpp-python
python scripts/build_native.py \
  --source third_party/qwentts.cpp \
  --backend cuda \
  --clean \
  --cmake-arg=-G \
  --cmake-arg=Ninja \
  --cmake-arg="-DCMAKE_CUDA_ARCHITECTURES=75-virtual;80-real;86-real;90-real;121-real"
python -m build --wheel
```

CUDA-linked `libqwen` depends on CUDA runtime libraries and an NVIDIA driver at
runtime. Choose the wrapper wheel that matches the runtime and GPU target first;
the `faster-qwen3-tts` package only selects the Python adapter.
