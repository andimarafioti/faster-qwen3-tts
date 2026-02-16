# Qwen3-TTS CUDA Graphs

Real-time Qwen3-TTS inference using manual CUDA graph capture. No Flash Attention, no vLLM, no Triton. Just PyTorch.

## Results (Jetson AGX Orin 64GB)

| Model | ms/step | RTF | TTFT | Real-time? |
|---|---|---|---|---|
| 0.6B | 54ms | 1.55 | 77ms | Yes |
| 1.7B | 66ms | 1.24 | 77ms | Yes |

RTF > 1.0 means faster than real-time. Baseline HF `generate()` RTF: 0.175 (5.7x slower than real-time).

## Quick Start

```bash
git clone https://github.com/amarafioti/qwen3-tts-cuda-graphs
cd qwen3-tts-cuda-graphs

# Install deps + download models
./setup.sh

# Run benchmark
./benchmark.sh
```

## Voice Cloning with Precomputed Speaker Embeddings

For production use, you can extract the speaker embedding once and reuse it across all requests. This skips the speaker encoder and audio tokenizer at inference time, and enables accent-free multilingual synthesis using `x_vector_only` mode.

```bash
# 1. Extract speaker embedding from reference audio (one-time, ~10s)
python extract_speaker.py --ref_audio voice.wav --output speaker.pt

# 2. Generate speech with CUDA graphs (real-time)
python generate_xvec.py --speaker speaker.pt --text "Hello!" --language English --output en.wav
python generate_xvec.py --speaker speaker.pt --text "Bonjour!" --language French --output fr.wav
python generate_xvec.py --speaker speaker.pt --text "Hallo!" --language German --output de.wav
```

The speaker embedding is a 4KB file (2048-dim bf16 vector). In x_vector_only mode:
- **No accent bleed**: the model uses its native pronunciation for each language instead of copying the ref audio's accent
- **Shorter prefill**: 10 tokens vs ~80+ in full ICL clone mode, so TTFT is lower
- **No ref audio needed at runtime**: just the 4KB embedding file

## Requirements

- Python 3.10+
- PyTorch 2.1+ with CUDA
- Any NVIDIA GPU (tested: Jetson AGX Orin, more coming)

## How It Works

Qwen3-TTS runs two autoregressive transformers per decode step:
1. **Talker** (28 layers): generates first codebook token
2. **Predictor** (5 layers): generates 15 additional codebook tokens

Each step involves ~500 small CUDA kernel launches with Python overhead between them. CUDA graphs capture the entire step and replay it as a single GPU operation, eliminating all launch overhead.

Key techniques:
- **Static KV cache**: pre-allocated fixed-size tensors (no dynamic allocation during decode)
- **Manual attention**: direct SDPA + RoPE, bypassing HF's DynamicCache
- **Graph capture**: `torch.cuda.CUDAGraph` for both predictor and talker decode steps
- **Padded attention**: talker uses attention mask to handle variable-length KV within fixed buffers

## Files

```
manual_cudagraph_predictor.py   # Predictor graph (261 lines)
manual_cudagraph_talker.py      # Talker graph (341 lines)
fast_generate_v5.py             # Full generation loop (156 lines)
extract_speaker.py              # Extract speaker embedding from ref audio
generate_xvec.py                # End-to-end generation with precomputed speaker
benchmark.sh                    # Portable benchmark script
setup.sh                        # Install deps + download models
bench_v5.py                     # Detailed benchmark (ICL mode)
bench_ttft.py                   # Time-to-first-token benchmark
```

Core implementation: **758 lines** of Python.

## Benchmarking on Other GPUs

```bash
# Benchmark both models
./benchmark.sh

# Benchmark specific model
./benchmark.sh 0.6B
./benchmark.sh 1.7B
```

Results are saved as `bench_results_<GPU_NAME>.json`. Send them back to compare across hardware!

## Blog Post

See the accompanying blog post: *Real-Time Qwen3-TTS on a Jetson in 758 Lines of PyTorch* (link TBD)

## License

MIT

## Acknowledgments

- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) by the Qwen team
- [nano-qwen3tts-vllm](https://github.com/tsdocode/nano-qwen3tts-vllm) for inspiration on CUDA graph usage
- NVIDIA for providing the Jetson AGX Orin board
