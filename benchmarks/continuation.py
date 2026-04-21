#!/usr/bin/env python3
"""Benchmark fresh vs continuation follow-up turns."""
import argparse
import os
import time

import numpy as np
import torch

from faster_qwen3_tts import FasterQwen3TTS

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_FIRST = (
    "The first sentence should establish a natural speaking rhythm and a clear vocal posture."
)
DEFAULT_SECOND = (
    "The second sentence should sound like the same speaker continuing the same thought without a hard reset."
)
DEFAULT_REF_TEXT = (
    "I'm confused why some people have super short timelines, yet at the same time are bullish on scaling up reinforcement learning atop LLMs."
)


def _measure_stream_ttfa(fn, runs: int):
    values = []
    for _ in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        gen = fn()
        try:
            next(gen)
        finally:
            gen.close()
        torch.cuda.synchronize()
        values.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(values)), float(np.std(values))


def _measure_nonstream(fn, runs: int):
    latencies = []
    rtfs = []
    for _ in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = fn()
        torch.cuda.synchronize()
        total = time.perf_counter() - t0
        if len(result) == 3:
            audio_list, sr, _info = result
        else:
            audio_list, sr = result
        dur = len(audio_list[0]) / sr
        latencies.append(total * 1000)
        rtfs.append(dur / total if total > 0 else 0.0)
    return (
        float(np.mean(latencies)),
        float(np.std(latencies)),
        float(np.mean(rtfs)),
        float(np.std(rtfs)),
    )


def main():
    parser = argparse.ArgumentParser(description="Benchmark continuation-state follow-up turns")
    parser.add_argument("--mode", choices=["voice_clone", "custom_voice", "voice_design"], default="voice_clone")
    parser.add_argument("--model", default=None, help="Override model id")
    parser.add_argument("--language", default=os.environ.get("LANGUAGE", "English"))
    parser.add_argument("--first-text", default=os.environ.get("FIRST_TEXT", DEFAULT_FIRST))
    parser.add_argument("--second-text", default=os.environ.get("SECOND_TEXT", DEFAULT_SECOND))
    parser.add_argument("--chunk-size", type=int, default=int(os.environ.get("CHUNK_SIZE", "8")))
    parser.add_argument("--runs", type=int, default=int(os.environ.get("RUNS", "5")))
    parser.add_argument("--xvec-only", action="store_true")
    parser.add_argument("--speaker", default=os.environ.get("SPEAKER"))
    parser.add_argument("--instruct", default=os.environ.get("INSTRUCT", "Warm, conversational, medium pace."))
    parser.add_argument("--ref-audio", default=os.environ.get("REF_AUDIO", os.path.join(PROJECT_DIR, "ref_audio.wav")))
    parser.add_argument("--ref-text", default=os.environ.get("REF_TEXT", DEFAULT_REF_TEXT))
    parser.add_argument("--max-new-tokens", type=int, default=int(os.environ.get("MAX_NEW_TOKENS", "256")))
    args = parser.parse_args()

    if args.model is None:
        if args.mode == "voice_clone":
            args.model = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
        elif args.mode == "custom_voice":
            args.model = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
        else:
            args.model = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"

    print(f"Loading {args.mode}: {args.model}")
    model = FasterQwen3TTS.from_pretrained(
        args.model,
        device="cuda",
        dtype=torch.bfloat16,
        attn_implementation="eager",
        max_seq_len=2048,
    )

    common_kwargs = {
        "language": args.language,
    }

    if args.mode == "voice_clone":
        first_call = lambda **kw: model.generate_voice_clone(
            text=args.first_text,
            ref_audio=args.ref_audio,
            ref_text=args.ref_text,
            xvec_only=args.xvec_only,
            return_continuation_state="full",
            max_new_tokens=kw.pop("max_new_tokens", args.max_new_tokens),
            **common_kwargs,
            **kw,
        )
        fresh_nonstream = lambda: model.generate_voice_clone(
            text=args.second_text,
            ref_audio=args.ref_audio,
            ref_text=args.ref_text,
            xvec_only=args.xvec_only,
            return_continuation_state="delta",
            max_new_tokens=args.max_new_tokens,
            **common_kwargs,
        )
        fresh_stream = lambda: model.generate_voice_clone_streaming(
            text=args.second_text,
            ref_audio=args.ref_audio,
            ref_text=args.ref_text,
            xvec_only=args.xvec_only,
            chunk_size=args.chunk_size,
            return_continuation_state="delta",
            max_new_tokens=args.max_new_tokens,
            **common_kwargs,
        )
    elif args.mode == "custom_voice":
        speakers = model.model.get_supported_speakers() or []
        if not speakers and not args.speaker:
            raise RuntimeError("No speakers reported by model; pass --speaker explicitly")
        speaker = args.speaker or speakers[0]
        print(f"Using speaker: {speaker}")
        first_call = lambda **kw: model.generate_custom_voice(
            text=args.first_text,
            speaker=speaker,
            instruct=args.instruct,
            return_continuation_state="full",
            max_new_tokens=kw.pop("max_new_tokens", args.max_new_tokens),
            **common_kwargs,
            **kw,
        )
        fresh_nonstream = lambda: model.generate_custom_voice(
            text=args.second_text,
            speaker=speaker,
            instruct=args.instruct,
            return_continuation_state="delta",
            max_new_tokens=args.max_new_tokens,
            **common_kwargs,
        )
        fresh_stream = lambda: model.generate_custom_voice_streaming(
            text=args.second_text,
            speaker=speaker,
            instruct=args.instruct,
            chunk_size=args.chunk_size,
            return_continuation_state="delta",
            max_new_tokens=args.max_new_tokens,
            **common_kwargs,
        )
    else:
        first_call = lambda **kw: model.generate_voice_design(
            text=args.first_text,
            instruct=args.instruct,
            return_continuation_state="full",
            max_new_tokens=kw.pop("max_new_tokens", args.max_new_tokens),
            **common_kwargs,
            **kw,
        )
        fresh_nonstream = lambda: model.generate_voice_design(
            text=args.second_text,
            instruct=args.instruct,
            return_continuation_state="delta",
            max_new_tokens=args.max_new_tokens,
            **common_kwargs,
        )
        fresh_stream = lambda: model.generate_voice_design_streaming(
            text=args.second_text,
            instruct=args.instruct,
            chunk_size=args.chunk_size,
            return_continuation_state="delta",
            max_new_tokens=args.max_new_tokens,
            **common_kwargs,
        )

    print("Warmup run (includes graph capture)...")
    _ = first_call(max_new_tokens=20)

    print("Preparing continuation state from the first sentence...")
    first_result = first_call()
    state = first_result[2]["continuation_state"]

    if args.mode == "voice_clone":
        cont_nonstream = lambda: model.generate_voice_clone(
            text=args.second_text,
            ref_audio=args.ref_audio,
            ref_text=args.ref_text,
            xvec_only=args.xvec_only,
            continuation_state=state,
            return_continuation_state="delta",
            max_new_tokens=args.max_new_tokens,
            **common_kwargs,
        )
        cont_stream = lambda: model.generate_voice_clone_streaming(
            text=args.second_text,
            ref_audio=args.ref_audio,
            ref_text=args.ref_text,
            xvec_only=args.xvec_only,
            chunk_size=args.chunk_size,
            continuation_state=state,
            return_continuation_state="delta",
            max_new_tokens=args.max_new_tokens,
            **common_kwargs,
        )
    elif args.mode == "custom_voice":
        cont_nonstream = lambda: model.generate_custom_voice(
            text=args.second_text,
            speaker=speaker,
            instruct=args.instruct,
            continuation_state=state,
            return_continuation_state="delta",
            max_new_tokens=args.max_new_tokens,
            **common_kwargs,
        )
        cont_stream = lambda: model.generate_custom_voice_streaming(
            text=args.second_text,
            speaker=speaker,
            instruct=args.instruct,
            chunk_size=args.chunk_size,
            continuation_state=state,
            return_continuation_state="delta",
            max_new_tokens=args.max_new_tokens,
            **common_kwargs,
        )
    else:
        cont_nonstream = lambda: model.generate_voice_design(
            text=args.second_text,
            instruct=args.instruct,
            continuation_state=state,
            return_continuation_state="delta",
            max_new_tokens=args.max_new_tokens,
            **common_kwargs,
        )
        cont_stream = lambda: model.generate_voice_design_streaming(
            text=args.second_text,
            instruct=args.instruct,
            chunk_size=args.chunk_size,
            continuation_state=state,
            return_continuation_state="delta",
            max_new_tokens=args.max_new_tokens,
            **common_kwargs,
        )

    print("\n=== Second sentence only ===")
    ttfa_fresh = _measure_stream_ttfa(fresh_stream, args.runs)
    ttfa_cont = _measure_stream_ttfa(cont_stream, args.runs)
    ns_fresh = _measure_nonstream(fresh_nonstream, args.runs)
    ns_cont = _measure_nonstream(cont_nonstream, args.runs)

    print(
        f"fresh        | TTFA={ttfa_fresh[0]:7.1f}ms ± {ttfa_fresh[1]:5.1f} | "
        f"latency={ns_fresh[0]:7.1f}ms ± {ns_fresh[1]:5.1f} | "
        f"RTF={ns_fresh[2]:.3f} ± {ns_fresh[3]:.3f}"
    )
    print(
        f"continuation | TTFA={ttfa_cont[0]:7.1f}ms ± {ttfa_cont[1]:5.1f} | "
        f"latency={ns_cont[0]:7.1f}ms ± {ns_cont[1]:5.1f} | "
        f"RTF={ns_cont[2]:.3f} ± {ns_cont[3]:.3f}"
    )


if __name__ == "__main__":
    main()
