#!/usr/bin/env python3
"""Isolated-process benchmark of the public voice-clone APIs.

Run from the repository with the faster-qwen3-tts Python environment:
    python benchmarks/batched_throughput.py --model /path/to/1.7B-Base \
        --batch-sizes 1 4 8 16 32 --num-texts 32 --repeats 3 --output results.json

Add --upstream-python /path/to/original-qwen-env/python for an original qwen-tts
baseline. The default eager baseline uses the SAME qwen-tts/Transformers as the
graphs; the original distribution is a separate cross-environment comparison.
Each engine/size runs in a fresh process. All use the exact same precomputed ICL
prompt, texts, BF16, SDPA and explicit sampling settings. Timing includes prompt
building and waveform decoding, but excludes loading, reference extraction and
warmup/graph capture. No empty_cache calls occur inside the timed workload.
Sampling and BF16 batch numerics can change output length; report both clips/s
and audio seconds/s, rather than assuming every engine produces identical audio.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

REF_TEXT = (
    "I'm confused why some people have super short timelines, yet at the same time are bullish on scaling up "
    "reinforcement learning atop LLMs. If we're actually close to a human-like learner, then this whole approach "
    "of training on verifiable outcomes is doomed."
)
SENTENCES = [
    "Hello there.",
    "The meeting has been moved to Thursday afternoon.",
    "Please remember to bring your laptop and the printed slides.",
    "It rained all night, and the puddles on the street reflected the dim yellow lamps.",
    "He opened the door slowly, but the room behind it was completely dark.",
    "Ladies and gentlemen, I have just been informed that this speech is being generated faster than I can speak it.",
    "When the train finally arrived, nobody on the platform moved, as if they had all forgotten where they were going.",
    "Thank you for calling. All of our agents are currently busy, so please stay on the line.",
]
SAMPLING = dict(do_sample=True, top_k=50, top_p=1.0, temperature=0.9,
                repetition_penalty=1.05, subtalker_dosample=True,
                subtalker_top_k=50, subtalker_top_p=1.0, subtalker_temperature=0.9)
ENGINES = ("eager", "batch", "single", "original")
ENGINE_LABELS = {
    "original": "Original Qwen3-TTS — standard PyTorch",
    "eager": "qwen-tts-hf — standard PyTorch",
    "single": "faster-qwen3-tts — existing single-text graphs",
    "batch": "faster-qwen3-tts — this PR's batch graphs",
}
ENGINE_GUIDE = """The four engines are:

- **Original Qwen3-TTS — standard PyTorch:** the original `qwen-tts` distribution, using Hugging Face generation without CUDA graphs, in the separate original-Qwen environment.
- **qwen-tts-hf — standard PyTorch:** the compatibility fork that faster-qwen3-tts depends on, using Hugging Face generation without CUDA graphs. It runs in the same environment as both faster-qwen3-tts paths, providing the controlled baseline.
- **faster-qwen3-tts — existing single-text graphs:** the existing accelerated inference path, generating one text per call with CUDA graphs.
- **faster-qwen3-tts — this PR's batch graphs:** the proposed accelerated inference path, generating multiple texts per call with batched CUDA graphs.
"""


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 8, 16, 32])
    p.add_argument("--num-texts", type=int, default=32)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmup-tokens", type=int, default=32)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--allocator-conf", help="optional PyTorch allocator settings, applied to every worker")
    p.add_argument("--ref-audio", type=Path, default=Path("ref_audio.wav"))
    p.add_argument("--ref-text", default=REF_TEXT)
    p.add_argument("--texts-file", type=Path, help="UTF-8 file, one text per nonempty line; replaces built-in texts")
    p.add_argument("--language", default="English")
    p.add_argument("--engines", nargs="+", choices=ENGINES, default=["eager", "batch", "single"])
    p.add_argument("--upstream-python", type=Path)
    p.add_argument("--package-root", type=Path, default=Path(__file__).resolve().parents[1],
                   help="root containing the proposed faster_qwen3_tts package")
    p.add_argument("--output", type=Path, default=Path("batched-throughput.json"))
    p.add_argument("--worker", choices=["prepare", *ENGINES], help=argparse.SUPPRESS)
    p.add_argument("--size", type=int, help=argparse.SUPPRESS)
    p.add_argument("--prompt-file", type=Path, help=argparse.SUPPRESS)
    return p


def versions():
    result = {"python": platform.python_version(), "executable": sys.executable}
    for name in ("torch", "transformers", "qwen-tts", "qwen-tts-hf", "numpy", "soundfile"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result


def get_texts(args):
    if args.texts_file:
        texts = [t.strip() for t in args.texts_file.read_text(encoding="utf-8").splitlines() if t.strip()]
    else:
        texts = [SENTENCES[i % len(SENTENCES)] for i in range(args.num_texts)]
    if not texts:
        raise ValueError("No texts to benchmark")
    return sorted(texts, key=len, reverse=True)


def worker(args):
    import torch
    from qwen_tts import Qwen3TTSModel, VoiceClonePromptItem

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    if args.worker == "prepare":
        base = Qwen3TTSModel.from_pretrained(args.model, device_map="cuda",
                                            torch_dtype=torch.bfloat16, attn_implementation="sdpa")
        item = base.create_voice_clone_prompt(ref_audio=str(args.ref_audio), ref_text=args.ref_text)[0]
        payload = {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in vars(item).items()}
        torch.save(payload, args.prompt_file)
        print("Saved shared ICL prompt (no appended silence).", flush=True)
        return

    if args.worker in ("eager", "original"):
        model = Qwen3TTSModel.from_pretrained(args.model, device_map="cuda",
                                             torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    else:
        sys.path.insert(0, str(args.package_root.resolve()))
        from faster_qwen3_tts import FasterQwen3TTS
        model = FasterQwen3TTS.from_pretrained(args.model, device="cuda", dtype=torch.bfloat16,
                                              attn_implementation="sdpa")
    payload = torch.load(args.prompt_file, map_location="cuda", weights_only=True)
    base = model.model if args.worker in ("batch", "single") else model
    for key, value in SAMPLING.items():
        if key.startswith("subtalker_") and base.generate_defaults.get(key, value) != value:
            raise ValueError(f"Checkpoint predictor default {key} differs from benchmark settings")
    prompt = VoiceClonePromptItem(**payload)
    texts = get_texts(args)
    size = 1 if args.worker == "single" else args.size
    batches = [texts[i:i + size] for i in range(0, len(texts), size)]

    def clone(batch, token_limit):
        # The existing single path's predictor is captured with these defaults.
        kwargs = SAMPLING | dict(max_new_tokens=token_limit, non_streaming_mode=True)
        if args.worker in ("batch", "single"):
            # These are fixed predictor defaults on the faster public API.
            kwargs = {k: v for k, v in kwargs.items() if not k.startswith("subtalker_")}
        if args.worker == "single":
            return model.generate_voice_clone(text=batch[0], language=args.language,
                                               voice_clone_prompt=[prompt], **kwargs)
        return model.generate_voice_clone(text=batch, language=[args.language] * len(batch),
                                           voice_clone_prompt=[prompt] * len(batch), **kwargs)

    # Warm up each actual row count with its longest inputs; exclude capture/setup.
    for n in sorted({len(b) for b in batches}, reverse=True):
        clone(next(b for b in batches if len(b) == n), args.warmup_tokens)
    torch.cuda.synchronize()
    trials = []
    for rep in range(args.repeats):
        torch.manual_seed(args.seed + rep)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        durations = []
        for batch in batches:
            wavs, sr = clone(batch, args.max_new_tokens)
            if len(wavs) != len(batch):
                raise RuntimeError("Engine returned an incorrect number of clips")
            durations.extend(len(w) / sr for w in wavs)
        torch.cuda.synchronize()
        wall = time.perf_counter() - start
        # Conservative indicator: these clips may have hit the generation limit.
        near_limit = sum(d >= (args.max_new_tokens - 2) / 12 for d in durations)
        trial = dict(seed=args.seed + rep, wall_s=wall, audio_s=sum(durations),
                     clips_per_s=len(texts) / wall, audio_per_wall=sum(durations) / wall,
                     peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                     peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                     durations_s=durations, possibly_truncated_clips=near_limit)
        trials.append(trial)
        print(f"{args.worker} B={size} repeat {rep + 1}: {wall:.2f}s, "
              f"{sum(durations):.2f}s audio, {trial['audio_per_wall']:.2f}x, "
              f"{trial['peak_allocated_gib']:.2f} GiB allocated, limit flags={near_limit}", flush=True)
    result = dict(engine=args.worker, batch_size=size, num_texts=len(texts), trials=trials,
                  versions=versions(), gpu=torch.cuda.get_device_name(), cuda=torch.version.cuda,
                  allocator_conf=os.environ.get("PYTORCH_ALLOC_CONF", os.environ.get("PYTORCH_CUDA_ALLOC_CONF")),
                  package_root=str(args.package_root.resolve()) if args.worker in ("batch", "single") else None)
    if args.worker in ("batch", "single"):
        result["source_sha256"] = {
            name: hashlib.sha256((args.package_root / "faster_qwen3_tts" / name).read_bytes()).hexdigest()
            for name in ("model.py", "batched.py", "predictor_graph.py", "talker_graph.py", "sampling.py")
        }
    if args.worker == "batch":
        result["resident_graph_buckets"] = list(model._batched.resident)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


def markdown(results):
    rows = [ENGINE_GUIDE.strip(), "",
            "| Engine / inference path | Batch | Wall median (range), s | Audio median, s | Clips/s | Audio/wall | Allocated / reserved peak, GiB |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        t = r["trials"]
        med = lambda k: statistics.median(x[k] for x in t)
        walls = [x["wall_s"] for x in t]
        rows.append(f"| {ENGINE_LABELS[r['engine']]} | {r['batch_size']} | {med('wall_s'):.2f} "
                    f"({min(walls):.2f}–{max(walls):.2f}) | {med('audio_s'):.2f} | "
                    f"{med('clips_per_s'):.2f} | {med('audio_per_wall'):.2f}× | "
                    f"{max(x['peak_allocated_gib'] for x in t):.2f} / "
                    f"{max(x['peak_reserved_gib'] for x in t):.2f} |")
    return "\n".join(rows) + "\n"


def main():
    p = parser()
    args = p.parse_args()
    if min(args.num_texts, args.repeats, args.threads, *args.batch_sizes) < 1:
        p.error("Counts and batch sizes must be positive")
    if args.warmup_tokens < 2 or args.max_new_tokens < 2:
        p.error("Token limits must be at least 2")
    if args.worker:
        worker(args)
        return
    if "original" in args.engines and not args.upstream_python:
        p.error("original requires --upstream-python")
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    artifacts = args.output.parent / (args.output.stem + "-workers")
    artifacts.mkdir(exist_ok=True)
    prompt_file = artifacts / "prompt.pt"
    # Forward arguments with argv, never shell interpolation.
    forwarded = sys.argv[1:]
    script = str(Path(__file__).resolve())
    env = os.environ.copy()
    env["NUMBA_CACHE_DIR"] = str(artifacts / "numba-cache")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if args.allocator_conf:
        # Both names keep this option compatible with older and newer PyTorch.
        env["PYTORCH_ALLOC_CONF"] = args.allocator_conf
        env["PYTORCH_CUDA_ALLOC_CONF"] = args.allocator_conf

    def launch(python, engine, size, output):
        command = [str(python), "-u", script, *forwarded, "--worker", engine,
                   "--size", str(size), "--prompt-file", str(prompt_file), "--output", str(output)]
        subprocess.run(command, check=True, env=env)

    launch(sys.executable, "prepare", 1, args.output)
    texts = get_texts(args)
    settings = dict(model=args.model, ref_audio=str(args.ref_audio.resolve()), ref_text=args.ref_text,
                    reference_sha256=hashlib.sha256(args.ref_audio.read_bytes()).hexdigest(),
                    prompt_sha256=hashlib.sha256(prompt_file.read_bytes()).hexdigest(), texts=texts,
                    language=args.language, dtype="bfloat16", attention="sdpa", sampling=SAMPLING,
                    max_new_tokens=args.max_new_tokens, warmup_tokens=args.warmup_tokens,
                    repeats=args.repeats, threads=args.threads, seed=args.seed,
                    allocator_conf=env.get("PYTORCH_ALLOC_CONF", env.get("PYTORCH_CUDA_ALLOC_CONF")),
                    timing="Public API incl. prompt building and waveform decode; excludes model loading, "
                           "reference extraction and warmup; isolated process per engine/size",
                    platform=platform.platform())
    results = []
    engines = list(dict.fromkeys(args.engines))
    if args.upstream_python and "original" not in engines:
        engines.append("original")
    for engine in engines:
        for size in ([1] if engine == "single" else sorted(set(args.batch_sizes))):
            output = artifacts / f"{engine}-{size}.json"
            python = args.upstream_python if engine == "original" else sys.executable
            print(f"Starting isolated {engine}, batch {size}", flush=True)
            launch(python, engine, size, output)
            results.append(json.loads(output.read_text(encoding="utf-8")))
            # Keep completed measurements if a later worker fails.
            args.output.write_text(json.dumps(dict(settings=settings, results=results), indent=2), encoding="utf-8")
            args.output.with_suffix(".md").write_text(markdown(results), encoding="utf-8")
    print(markdown(results), flush=True)
    print(f"Raw measurements: {args.output}", flush=True)
    if any(t["possibly_truncated_clips"] for r in results for t in r["trials"]):
        print("Some clips may have reached the token cap. Raise --max-new-tokens and check before quoting throughput.")


if __name__ == "__main__":
    main()
