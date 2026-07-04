#!/usr/bin/env python3
"""Profile Python-visible and native GGML streaming TTFA phases.

This diagnostic uses the timers exposed by the qwentts.cpp ctypes wrapper
and parses native `[Profile]` log markers when the loaded `libqwen` contains
the local timing patch. It separates Python setup/callback plumbing from
native prompt build, first-frame AR generation, and first codec decode.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Iterable

from faster_qwen3_tts import FasterQwen3TTS


DEFAULT_TEXT = (
    "Ladies and gentlemen, I have just been informed that this speech is being "
    "generated faster than I can speak it. Please remain calm."
)
DEFAULT_REF_TEXT = (
    "I'm confused why some people have super short timelines, yet at the same "
    "time are bullish on scaling up reinforcement learning atop LLMs."
)


def parse_chunk_sizes(value: str) -> list[int]:
    sizes = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not sizes:
        raise argparse.ArgumentTypeError("expected at least one chunk size")
    if any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("chunk sizes must be positive")
    return sizes


def mean(records: Iterable[dict[str, Any]], key: str) -> float | None:
    values = [float(record[key]) for record in records if record.get(key) is not None]
    if not values:
        return None
    return statistics.mean(values)


def ms(value: float | None) -> str:
    if value is None:
        return "   n/a"
    return f"{value:6.1f}"


def parse_native_profile_event(message: str) -> dict[str, Any] | None:
    if not message.startswith("[Profile] "):
        return None
    event: dict[str, Any] = {}
    for part in message[len("[Profile] ") :].split():
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        try:
            if "." in value:
                event[key] = float(value)
            else:
                event[key] = int(value)
        except ValueError:
            event[key] = value
    return event


def first_chunk(
    model: FasterQwen3TTS,
    args: argparse.Namespace,
    *,
    chunk_size: int,
) -> dict[str, Any]:
    kwargs = {
        "text": args.text,
        "language": args.language,
        "max_new_tokens": max(args.max_new_tokens, chunk_size),
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "do_sample": not args.greedy,
        "repetition_penalty": args.repetition_penalty,
        "chunk_size": chunk_size,
    }

    if args.mode == "custom":
        stream = model.generate_custom_voice_streaming(
            speaker=args.speaker,
            instruct=args.instruct or None,
            **kwargs,
        )
    elif args.mode == "design":
        stream = model.generate_voice_design_streaming(
            instruct=args.instruct,
            **kwargs,
        )
    else:
        stream = model.generate_voice_clone_streaming(
            ref_audio=args.ref_audio,
            ref_text=args.ref_text,
            xvec_only=args.xvec_only,
            **kwargs,
        )

    if hasattr(args, "_native_logs"):
        args._native_logs.clear()

    t0 = time.perf_counter()
    try:
        chunk, sr, timing = next(stream)
        ttfa_ms = (time.perf_counter() - t0) * 1000
    finally:
        stream.close()

    profile = timing.get("ggml_profile") or {}
    native_logs = list(getattr(args, "_native_logs", []))
    native_events = [
        event for event in (parse_native_profile_event(item["message"]) for item in native_logs) if event is not None
    ]
    native_by_phase = {
        str(event["phase"]): event
        for event in native_events
        if "phase" in event
    }
    return {
        "chunk_size": chunk_size,
        "ttfa_ms": ttfa_ms,
        "audio_samples": int(len(chunk)),
        "audio_s": float(len(chunk)) / int(sr),
        "adapter_prepare_ms": timing.get("adapter_prepare_ms"),
        "make_params_ms": profile.get("make_params_ms"),
        "lock_wait_ms": profile.get("lock_wait_ms"),
        "native_enter_ms": profile.get("native_enter_ms"),
        "first_callback_enter_ms": profile.get("first_callback_enter_ms"),
        "first_callback_to_yield_ms": profile.get("first_callback_to_yield_ms"),
        "first_callback_copy_ms": profile.get("first_callback_copy_ms"),
        "first_callback_queue_ms": profile.get("first_callback_queue_ms"),
        "first_callback_n_samples": profile.get("first_callback_n_samples"),
        "callback_count_at_yield": profile.get("callback_count"),
        "native_profile_events": native_events,
        "native_prompt_build_done_ms": native_by_phase.get("prompt_build_done", {}).get("total_ms"),
        "native_step0_talker_prefill_done_ms": native_by_phase.get("step0_talker_prefill_done", {}).get("total_ms"),
        "native_step0_code_predictor_done_ms": native_by_phase.get("step0_code_predictor_done", {}).get("total_ms"),
        "native_step0_frame_done_ms": native_by_phase.get("step0_frame_done", {}).get("total_ms"),
        "native_first_emit_push_enter_ms": native_by_phase.get("first_emit_push_enter", {}).get("total_ms"),
        "native_first_emit_done_ms": native_by_phase.get("first_emit_done", {}).get("total_ms"),
        "native_first_emit_codec_decode_ms": native_by_phase.get("first_emit_done", {}).get("codec_decode_ms"),
        "native_logs": native_logs,
        "raw_timing": timing,
    }


def print_summary(records: list[dict[str, Any]]) -> None:
    by_chunk: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        by_chunk.setdefault(int(record["chunk_size"]), []).append(record)

    print()
    print(
        "chunk  runs  ttfa  native_enter  first_callback  cb_to_yield  "
        "make_params  lock_wait  cb_copy"
    )
    print("-" * 88)
    for chunk_size, chunk_records in sorted(by_chunk.items()):
        print(
            f"{chunk_size:5d}  {len(chunk_records):4d}  "
            f"{ms(mean(chunk_records, 'ttfa_ms'))}  "
            f"{ms(mean(chunk_records, 'native_enter_ms'))}  "
            f"{ms(mean(chunk_records, 'first_callback_enter_ms'))}  "
            f"{ms(mean(chunk_records, 'first_callback_to_yield_ms'))}  "
            f"{ms(mean(chunk_records, 'make_params_ms'))}  "
            f"{ms(mean(chunk_records, 'lock_wait_ms'))}  "
            f"{ms(mean(chunk_records, 'first_callback_copy_ms'))}"
        )

    if any(record.get("native_prompt_build_done_ms") is not None for record in records):
        print()
        print(
            "chunk  prompt  step0_prefill  step0_cp  step0_done  "
            "emit_enter  emit_done  codec_decode"
        )
        print("-" * 84)
        for chunk_size, chunk_records in sorted(by_chunk.items()):
            print(
                f"{chunk_size:5d}  "
                f"{ms(mean(chunk_records, 'native_prompt_build_done_ms'))}  "
                f"{ms(mean(chunk_records, 'native_step0_talker_prefill_done_ms'))}  "
                f"{ms(mean(chunk_records, 'native_step0_code_predictor_done_ms'))}  "
                f"{ms(mean(chunk_records, 'native_step0_frame_done_ms'))}  "
                f"{ms(mean(chunk_records, 'native_first_emit_push_enter_ms'))}  "
                f"{ms(mean(chunk_records, 'native_first_emit_done_ms'))}  "
                f"{ms(mean(chunk_records, 'native_first_emit_codec_decode_ms'))}"
            )

    prompt_phases = [
        ("prompt_bpe_main_done", "bpe_main"),
        ("prompt_text_projection_load_done", "proj_load"),
        ("prompt_special_text_embeds_done", "special_text"),
        ("prompt_special_codec_embeds_done", "special_codec"),
        ("prompt_bpe_instruct_done", "bpe_instruct"),
        ("prompt_instruct_rows_done", "instruct_rows"),
        ("prompt_role_rows_done", "role_rows"),
        ("prompt_codec_prefix_done", "codec_prefix"),
        ("prompt_standard_rows_done", "text_rows"),
        ("prompt_done", "prompt_done"),
    ]
    prompt_values: dict[str, list[float]] = {phase: [] for phase, _ in prompt_phases}
    for record in records:
        for event in record.get("native_profile_events", []):
            phase = event.get("phase")
            if phase in prompt_values and event.get("total_ms") is not None:
                prompt_values[phase].append(float(event["total_ms"]))
    if any(prompt_values.values()):
        print()
        print("prompt phase              cumulative_ms")
        print("-" * 37)
        for phase, label in prompt_phases:
            values = prompt_values[phase]
            if values:
                print(f"{label:24s} {statistics.mean(values):8.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
    parser.add_argument("--mode", choices=["custom", "design", "clone"], default="custom")
    parser.add_argument("--quant", default="BF16")
    parser.add_argument("--cache-dir", default=".cache/qwentts")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--qwentts-lib")
    parser.add_argument("--chunk-sizes", type=parse_chunk_sizes, default=parse_chunk_sizes("2,4,8,16"))
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--language", default="English")
    parser.add_argument("--speaker", default="aiden")
    parser.add_argument(
        "--instruct",
        default="Warm, confident narrator with clear diction and a steady pace.",
    )
    parser.add_argument("--ref-audio", default="ref_audio.wav")
    parser.add_argument("--ref-text", default=DEFAULT_REF_TEXT)
    parser.add_argument("--xvec-only", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--show-native-log", action="store_true")
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    if args.runs <= 0:
        raise SystemExit("--runs must be positive")
    if args.warmup_runs < 0:
        raise SystemExit("--warmup-runs cannot be negative")

    model = FasterQwen3TTS.from_pretrained(
        args.model,
        backend="ggml",
        quant=args.quant,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        qwentts_library_path=args.qwentts_lib,
    )
    args._native_logs = []
    if hasattr(model.runtime, "set_log_callback"):
        def log_callback(level: int, message: str) -> None:
            args._native_logs.append({"level": level, "message": message})
            if args.show_native_log:
                print(message)

        model.runtime.set_log_callback(log_callback)

    warmup_chunk = args.chunk_sizes[0]
    for _ in range(args.warmup_runs):
        first_chunk(model, args, chunk_size=warmup_chunk)

    records: list[dict[str, Any]] = []
    for chunk_size in args.chunk_sizes:
        for run in range(args.runs):
            record = first_chunk(model, args, chunk_size=chunk_size)
            record["run"] = run + 1
            records.append(record)
            print(
                f"chunk={chunk_size:2d} run={run + 1:2d} "
                f"ttfa={record['ttfa_ms']:.1f}ms "
                f"native_enter={ms(record.get('native_enter_ms')).strip()}ms "
                f"first_callback={ms(record.get('first_callback_enter_ms')).strip()}ms "
                f"cb_to_yield={ms(record.get('first_callback_to_yield_ms')).strip()}ms"
            )

    print_summary(records)

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(records, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
