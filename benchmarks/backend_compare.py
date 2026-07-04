#!/usr/bin/env python3
"""Compare faster-qwen3-tts torch and GGML backends on a shared workload."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from faster_qwen3_tts import FasterQwen3TTS


DEFAULT_TEXT = (
    "Ladies and gentlemen, I have just been informed that this speech is being "
    "generated faster than I can speak it. Please remain calm."
)
DEFAULT_REF_TEXT = (
    "I'm confused why some people have super short timelines, yet at the same "
    "time are bullish on scaling up reinforcement learning atop LLMs. If we're "
    "actually close to a human-like learner, then this whole approach of training "
    "on verifiable outcomes is doomed."
)


def parse_chunk_sizes(value: str) -> list[int]:
    sizes = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not sizes:
        raise argparse.ArgumentTypeError("expected at least one chunk size")
    if any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("chunk sizes must be positive")
    return sizes


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def model_id(size: str, mode: str) -> str:
    suffix = "CustomVoice" if mode == "custom" else "Base"
    return f"Qwen/Qwen3-TTS-12Hz-{size}-{suffix}"


def local_snapshot_for(model: str) -> str:
    hub_root = Path.home() / ".cache" / "huggingface" / "hub"
    repo = "models--" + model.replace("/", "--")
    snapshots = hub_root / repo / "snapshots"
    if not snapshots.is_dir():
        return model
    dirs = sorted(path for path in snapshots.iterdir() if path.is_dir())
    return str(dirs[-1]) if dirs else model


def summarize_audio(audio: np.ndarray, sr: int, total_s: float) -> dict[str, Any]:
    audio_s = len(audio) / sr if sr else 0.0
    frames = max(1, int(round(audio_s * 12.5)))
    return {
        "audio_s": audio_s,
        "total_s": total_s,
        "frames": frames,
        "ms_per_frame": total_s * 1000 / frames,
        "compute_per_audio": total_s / audio_s if audio_s > 0 else None,
        "audio_per_compute": audio_s / total_s if total_s > 0 else None,
    }


def load_model(args: argparse.Namespace, backend: str, model_name: str):
    if backend == "ggml":
        return FasterQwen3TTS.from_pretrained(
            model_name,
            backend="ggml",
            quant=args.quant,
            cache_dir=args.ggml_cache_dir,
            local_files_only=args.local_files_only,
            qwentts_library_path=args.qwentts_lib,
        )

    torch_model = local_snapshot_for(model_name) if args.local_files_only else model_name
    return FasterQwen3TTS.from_pretrained(
        torch_model,
        device=args.device,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        max_seq_len=args.max_seq_len,
    )


def generation_kwargs(args: argparse.Namespace, mode: str) -> dict[str, Any]:
    common = {
        "text": args.text,
        "language": args.language,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": not args.greedy,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
    }
    if mode == "custom":
        return {
            **common,
            "speaker": args.speaker,
            "instruct": args.instruct or None,
        }
    return {
        **common,
        "ref_audio": args.ref_audio,
        "ref_text": args.ref_text,
        "xvec_only": args.xvec_only,
    }


def generate_buffered(model, mode: str, kwargs: dict[str, Any]):
    if mode == "custom":
        return model.generate_custom_voice(**kwargs)
    return model.generate_voice_clone(**kwargs)


def generate_streaming(model, mode: str, kwargs: dict[str, Any], chunk_size: int):
    stream_kwargs = {**kwargs, "chunk_size": chunk_size}
    if mode == "custom":
        return model.generate_custom_voice_streaming(**stream_kwargs)
    return model.generate_voice_clone_streaming(**stream_kwargs)


def run_backend(args: argparse.Namespace, backend: str) -> dict[str, Any]:
    name = model_id(args.model_size, args.mode)
    print(f"\n=== {backend} {args.model_size} {args.mode} {args.quant} ===", flush=True)
    print(f"model: {name}", flush=True)

    load_start = time.perf_counter()
    model = load_model(args, backend, name)
    sync_cuda()
    load_s = time.perf_counter() - load_start
    print(f"load: {load_s:.3f}s", flush=True)

    kwargs = generation_kwargs(args, args.mode)
    print("warmup...", flush=True)
    warmup_kwargs = {**kwargs, "text": kwargs["text"][:50], "max_new_tokens": min(args.max_new_tokens, 20)}
    warm_start = time.perf_counter()
    generate_buffered(model, args.mode, warmup_kwargs)
    sync_cuda()
    warmup_s = time.perf_counter() - warm_start
    print(f"warmup: {warmup_s:.3f}s", flush=True)

    result: dict[str, Any] = {
        "backend": backend,
        "model": name,
        "mode": args.mode,
        "quant": args.quant,
        "load_s": load_s,
        "warmup_s": warmup_s,
        "max_new_tokens": args.max_new_tokens,
        "ttfa": {},
        "buffered": [],
        "streaming_chunk8": [],
    }

    print("TTFA sweep...", flush=True)
    for chunk_size in args.chunk_sizes:
        values = []
        for run in range(args.ttfa_runs):
            sync_cuda()
            t0 = time.perf_counter()
            stream = generate_streaming(model, args.mode, kwargs, chunk_size)
            try:
                chunk, sr, _timing = next(stream)
            finally:
                stream.close()
            sync_cuda()
            ttfa_ms = (time.perf_counter() - t0) * 1000
            values.append(ttfa_ms)
            print(
                f"  chunk={chunk_size:2d} run={run + 1}: "
                f"{ttfa_ms:.1f}ms, samples={len(chunk)}",
                flush=True,
            )
        result["ttfa"][str(chunk_size)] = {
            "mean_ms": float(np.mean(values)),
            "std_ms": float(np.std(values)),
            "runs_ms": values,
        }

    print("buffered throughput...", flush=True)
    for run in range(args.throughput_runs):
        sync_cuda()
        t0 = time.perf_counter()
        audio_list, sr = generate_buffered(model, args.mode, kwargs)
        sync_cuda()
        item = summarize_audio(audio_list[0], sr, time.perf_counter() - t0)
        result["buffered"].append(item)
        print(
            f"  run={run + 1}: audio={item['audio_s']:.2f}s "
            f"time={item['total_s']:.2f}s ms/frame={item['ms_per_frame']:.1f}",
            flush=True,
        )

    print("streaming throughput (chunk=8)...", flush=True)
    for run in range(args.throughput_runs):
        chunks = []
        sr = None
        sync_cuda()
        t0 = time.perf_counter()
        for audio_chunk, sr, _timing in generate_streaming(model, args.mode, kwargs, 8):
            chunks.append(audio_chunk)
        sync_cuda()
        audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        item = summarize_audio(audio, sr or 24000, time.perf_counter() - t0)
        item["chunks"] = len(chunks)
        result["streaming_chunk8"].append(item)
        print(
            f"  run={run + 1}: chunks={len(chunks)} audio={item['audio_s']:.2f}s "
            f"time={item['total_s']:.2f}s ms/frame={item['ms_per_frame']:.1f}",
            flush=True,
        )

    return result


def print_comparison(results: list[dict[str, Any]]) -> None:
    if len(results) != 2:
        return
    by_backend = {item["backend"]: item for item in results}
    if "torch" not in by_backend or "ggml" not in by_backend:
        return
    torch_result = by_backend["torch"]
    ggml_result = by_backend["ggml"]

    print("\n=== Backend comparison ===")
    print("chunk  torch_ttfa_ms  ggml_ttfa_ms  speedup")
    for chunk_size in torch_result["ttfa"]:
        t = torch_result["ttfa"][chunk_size]["mean_ms"]
        g = ggml_result["ttfa"][chunk_size]["mean_ms"]
        print(f"{chunk_size:>5}  {t:13.1f}  {g:12.1f}  {t / g:7.2f}x")

    for section in ("buffered", "streaming_chunk8"):
        t = float(np.mean([item["ms_per_frame"] for item in torch_result[section]]))
        g = float(np.mean([item["ms_per_frame"] for item in ggml_result[section]]))
        print(f"{section}: torch={t:.1f} ms/frame ggml={g:.1f} ms/frame speedup={t / g:.2f}x")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["torch", "ggml", "both"], default="both")
    parser.add_argument("--mode", choices=["custom", "base"], default="custom")
    parser.add_argument("--model-size", choices=["0.6B", "1.7B"], default="1.7B")
    parser.add_argument("--quant", default="BF16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--ttfa-runs", type=int, default=3)
    parser.add_argument("--throughput-runs", type=int, default=2)
    parser.add_argument("--chunk-sizes", type=parse_chunk_sizes, default=parse_chunk_sizes("2,4,8,16"))
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--language", default="English")
    parser.add_argument("--speaker", default="aiden")
    parser.add_argument("--instruct", default="")
    parser.add_argument("--ref-audio", default="ref_audio.wav")
    parser.add_argument("--ref-text", default=DEFAULT_REF_TEXT)
    parser.add_argument("--xvec-only", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--ggml-cache-dir", default=".cache/qwentts")
    parser.add_argument("--qwentts-lib")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--json-output", default="bench_results_backends.json")
    args = parser.parse_args()

    if args.mode == "custom" and args.model_size == "0.6B" and args.quant == "BF16":
        print("Note: GGUF cache may only contain 0.6B CustomVoice Q4_K_M; use --quant Q4_K_M if needed.")

    backends = ["torch", "ggml"] if args.backend == "both" else [args.backend]
    results = [run_backend(args, backend) for backend in backends]
    print_comparison(results)

    path = Path(args.json_output)
    payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "results": results,
    }
    path.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
