#!/usr/bin/env python3
"""
Create a Hugging Face Inference Endpoint for faster-qwen3-tts.

This helper assumes:
- the model repository selected for the endpoint is mounted at /repository
- the custom container image was built from Dockerfile.hf
- autoscaling replica bounds are configured via the public huggingface_hub API
"""

import argparse
import json

from huggingface_hub import HfApi


def _parse_kv(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"Expected KEY=VALUE, got: {value}")
    key, raw_value = value.split("=", 1)
    key = key.strip()
    if not key:
        raise argparse.ArgumentTypeError(f"Missing key in: {value}")
    return key, raw_value


def _build_env(args: argparse.Namespace) -> dict[str, str]:
    env = {
        "HF_MODEL_DIR": "/repository",
        "QWEN_TTS_MODEL": "/repository",
        "QWEN_TTS_MAX_PENDING": str(args.max_pending),
        "QWEN_TTS_CHUNK_SIZE": str(args.chunk_size),
        "QWEN_TTS_DEVICE": args.device,
    }
    if args.voices:
        env["QWEN_TTS_VOICES"] = args.voices
    if args.ref_audio:
        env["QWEN_TTS_REF_AUDIO"] = args.ref_audio
    if args.ref_text:
        env["QWEN_TTS_REF_TEXT"] = args.ref_text
    if args.language:
        env["QWEN_TTS_LANGUAGE"] = args.language
    for key, value in args.env:
        env[key] = value
    return env


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a Hugging Face Inference Endpoint")
    parser.add_argument("--name", required=True, help="Endpoint name")
    parser.add_argument("--repository", required=True, help="Model repository mounted at /repository")
    parser.add_argument("--image", required=True, help="Published custom image URL, for example ghcr.io/<org>/<image>:tag")
    parser.add_argument("--vendor", default="aws", help="Cloud vendor")
    parser.add_argument("--region", default="us-east-1", help="Cloud region")
    parser.add_argument("--accelerator", default="gpu", help="Endpoint accelerator")
    parser.add_argument("--instance-size", default="x1", help="Instance size")
    parser.add_argument("--instance-type", default="nvidia-a10g", help="Instance type")
    parser.add_argument("--type", default="protected", help="Endpoint exposure type")
    parser.add_argument("--namespace", help="HF namespace; defaults to current user/org")
    parser.add_argument("--revision", help="Optional model revision")
    parser.add_argument("--min-replica", type=int, default=1, help="Minimum replica count")
    parser.add_argument("--max-replica", type=int, default=6, help="Maximum replica count")
    parser.add_argument(
        "--scale-to-zero-timeout",
        type=int,
        help="Idle minutes before scaling to zero; omit to keep one warm replica",
    )
    parser.add_argument("--max-pending", type=int, default=8, help="Server-side pending request cap")
    parser.add_argument("--chunk-size", type=int, default=12, help="Streaming chunk size")
    parser.add_argument("--device", default="cuda", help="Torch device inside the container")
    parser.add_argument("--voices", help="Path inside /repository to a voices.json file")
    parser.add_argument("--ref-audio", help="Path inside /repository or container to a reference WAV")
    parser.add_argument("--ref-text", default="", help="Reference transcript for --ref-audio")
    parser.add_argument("--language", default="Auto", help="Default language when --ref-audio is used")
    parser.add_argument("--env", type=_parse_kv, action="append", default=[], help="Additional env var, repeated")
    parser.add_argument("--wait", action="store_true", help="Wait for the endpoint deployment to finish")
    parser.add_argument("--timeout", type=int, default=1800, help="Wait timeout in seconds")
    return parser


def main() -> None:
    args = _parser().parse_args()

    if not args.voices and not args.ref_audio:
        raise SystemExit("Provide either --voices or --ref-audio so the server has at least one voice")

    api = HfApi()
    endpoint = api.create_inference_endpoint(
        name=args.name,
        repository=args.repository,
        framework="custom",
        accelerator=args.accelerator,
        instance_size=args.instance_size,
        instance_type=args.instance_type,
        region=args.region,
        vendor=args.vendor,
        namespace=args.namespace,
        revision=args.revision,
        min_replica=args.min_replica,
        max_replica=args.max_replica,
        scale_to_zero_timeout=args.scale_to_zero_timeout,
        type=args.type,
        custom_image={
            "url": args.image,
            "health_route": "/health",
        },
        env=_build_env(args),
    )

    if args.wait:
        endpoint = endpoint.wait(timeout=args.timeout)

    print(
        json.dumps(
            {
                "name": endpoint.name,
                "status": endpoint.status,
                "url": endpoint.url,
                "repository": args.repository,
                "image": args.image,
                "min_replica": args.min_replica,
                "max_replica": args.max_replica,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
