#!/usr/bin/env python3
"""
Detect boundary clicks by comparing streaming vs non-streaming decode.
"""
import os
import numpy as np
import torch

from faster_qwen3_tts import FasterQwen3TTS
from faster_qwen3_tts.streaming import fast_generate_streaming
from faster_qwen3_tts.model import _stream_decode_chunks


TEXT = os.environ.get(
    "TEXT",
    "Ladies and gentlemen, I have just been informed that this speech is being generated faster than I can speak it.",
)
LANG = os.environ.get("LANGUAGE", "English")
REF_AUDIO = os.environ.get("REF_AUDIO", "ref_audio.wav")
REF_TEXT = os.environ.get(
    "REF_TEXT",
    "I'm confused why some people have super short timelines, yet at the same time are bullish on scaling up reinforcement learning atop LLMs.",
)
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "512"))
CONTEXT_FRAMES = int(os.environ.get("CONTEXT_FRAMES", "25"))
SMOOTH_MS = float(os.environ.get("SMOOTH_MS", "5.0"))
DETERMINISTIC = os.environ.get("DETERMINISTIC", "0")
MODE = os.environ.get("MODE", "full")
LOOKAHEAD_FRAMES = int(os.environ.get("LOOKAHEAD_FRAMES", "0"))
SAVE_WAV = os.environ.get("SAVE_WAV", "0") == "1"


def main():
    model_id = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    model = FasterQwen3TTS.from_pretrained(model_id, device="cuda", dtype=torch.bfloat16)

    if DETERMINISTIC == "1":
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    elif DETERMINISTIC == "2":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    m, talker, config, tie, tam, tth, tpe = model._prepare_generation(
        TEXT, REF_AUDIO, REF_TEXT, language=LANG, xvec_only=True
    )

    codec_iter = fast_generate_streaming(
        talker=talker,
        talker_input_embeds=tie,
        attention_mask=tam,
        trailing_text_hiddens=tth,
        tts_pad_embed=tpe,
        config=config,
        predictor_graph=model.predictor_graph,
        talker_graph=model.talker_graph,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=1.0,
        top_k=0,
        do_sample=False,
        repetition_penalty=1.05,
        chunk_size=CHUNK_SIZE,
    )

    chunks = []
    for codec_chunk, timing in codec_iter:
        chunks.append((codec_chunk, timing))

    if not chunks:
        print("No streaming codes produced.")
        return

    def chunk_iter():
        for c, t in chunks:
            yield c, t

    stream_audio = []
    boundaries = []
    total_len = 0
    for audio_chunk, sr, timing in _stream_decode_chunks(
        m.speech_tokenizer,
        chunk_iter(),
        context_frames=CONTEXT_FRAMES,
        lookahead_frames=LOOKAHEAD_FRAMES,
        smooth_ms=SMOOTH_MS,
        mode=MODE,
    ):
        stream_audio.append(audio_chunk)
        total_len += len(audio_chunk)
        boundaries.append(total_len)

    if not stream_audio:
        print("No streaming audio produced.")
        return

    stream_audio = np.concatenate(stream_audio)
    spf = int(m.speech_tokenizer.get_decode_upsample_rate())

    all_codes = torch.cat([c for c, _t in chunks], dim=0)
    full_audio_list, sr_full = m.speech_tokenizer.decode({"audio_codes": all_codes.unsqueeze(0)})
    full_audio = full_audio_list[0]
    if hasattr(full_audio, "cpu"):
        full_audio = full_audio.flatten().cpu().numpy()
    else:
        full_audio = full_audio.flatten() if hasattr(full_audio, "flatten") else full_audio

    # Compute boundary discontinuities in streaming output
    boundary = []
    for idx in boundaries[:-1]:
        if 0 < idx < len(stream_audio):
            boundary.append(abs(stream_audio[idx] - stream_audio[idx - 1]))

    boundary = np.array(boundary)
    rand_idx = np.random.default_rng(0).integers(1, len(stream_audio) - 1, size=min(1000, len(stream_audio) - 2))
    random_deltas = np.abs(stream_audio[rand_idx] - stream_audio[rand_idx - 1])

    print(f"Stream audio length: {len(stream_audio)} samples, sr={sr}")
    print(f"Full audio length:   {len(full_audio)} samples, sr={sr_full}")
    print(f"Boundary delta mean: {boundary.mean():.6f}, median: {np.median(boundary):.6f}")
    print(f"Random delta mean:   {random_deltas.mean():.6f}, median: {np.median(random_deltas):.6f}")
    print(f"Boundary/Random median ratio: {np.median(boundary) / (np.median(random_deltas) + 1e-8):.2f}x")
    
    # Compare streaming to full decode prefix
    min_len = min(len(stream_audio), len(full_audio))
    diff = np.abs(stream_audio[:min_len] - full_audio[:min_len])
    max_diff = float(diff.max()) if diff.size else 0.0
    p99 = float(np.percentile(diff, 99)) if diff.size else 0.0
    print(f"Streaming vs full prefix max diff: {max_diff:.6f}, p99: {p99:.6f}")

    if SAVE_WAV:
        import soundfile as sf
        sf.write("streaming_output.wav", stream_audio, sr)
        sf.write("full_output.wav", full_audio, sr_full)
        print("Wrote streaming_output.wav and full_output.wav")


if __name__ == "__main__":
    main()
