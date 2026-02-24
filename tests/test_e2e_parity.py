import os
import random

import numpy as np
import pytest
import torch

from qwen_tts import Qwen3TTSModel

from faster_qwen3_tts import FasterQwen3TTS


MODEL_ID = os.environ.get("QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-Base")


def _seed_all(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@pytest.fixture(scope="module")
def parity_fixture():
    _seed_all(0)

    device = "cuda"
    dtype = torch.bfloat16

    ref_audio = "ref_audio.wav"
    text = "Short parity test."

    base = Qwen3TTSModel.from_pretrained(
        MODEL_ID, device_map=device, dtype=dtype, attn_implementation="eager"
    )
    fast = FasterQwen3TTS.from_pretrained(
        MODEL_ID, device=device, dtype=dtype, attn_implementation="eager"
    )

    prompt_items = base.create_voice_clone_prompt(ref_audio=ref_audio, ref_text="", x_vector_only_mode=True)
    vcp = base._prompt_items_to_voice_clone_prompt(prompt_items)

    input_ids_base = base._tokenize_texts([base._build_assistant_text(text)])
    input_ids_fast = fast.model._tokenize_texts([fast.model._build_assistant_text(text)])

    tie, tam, tth, tpe = fast._build_talker_inputs_local(
        m=fast.model.model,
        input_ids=input_ids_fast,
        ref_ids=[None],
        voice_clone_prompt=vcp,
        languages=["English"],
        speakers=None,
        non_streaming_mode=False,
    )

    if not fast._warmed_up:
        # Force deterministic predictor sampling for parity.
        fast.predictor_graph.do_sample = False
        fast.predictor_graph.top_k = 0
        fast.predictor_graph.top_p = 1.0
        fast.predictor_graph.temperature = 1.0
        fast._warmup(tie.shape[1])

    return dict(
        base=base,
        fast=fast,
        vcp=vcp,
        input_ids_base=input_ids_base,
        input_ids_fast=input_ids_fast,
        tie=tie,
        tam=tam,
        tth=tth,
        tpe=tpe,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for parity test.")
def test_voice_clone_token_parity_xvec_only(parity_fixture):
    base = parity_fixture["base"]
    fast = parity_fixture["fast"]
    vcp = parity_fixture["vcp"]
    input_ids_base = parity_fixture["input_ids_base"]
    input_ids_fast = parity_fixture["input_ids_fast"]
    tie = parity_fixture["tie"]
    tam = parity_fixture["tam"]
    tth = parity_fixture["tth"]
    tpe = parity_fixture["tpe"]

    assert torch.equal(input_ids_base[0].cpu(), input_ids_fast[0].cpu())

    # Run fast generation (no silence append involved in xvec-only)
    from faster_qwen3_tts.generate import fast_generate

    fast_codes, _ = fast_generate(
        talker=fast.model.model.talker,
        talker_input_embeds=tie,
        attention_mask=tam,
        trailing_text_hiddens=tth,
        tts_pad_embed=tpe,
        config=fast.model.model.config.talker_config,
        predictor_graph=fast.predictor_graph,
        talker_graph=fast.talker_graph,
        max_new_tokens=64,
        min_new_tokens=0,
        do_sample=False,
        top_k=0,
        top_p=1.0,
        temperature=1.0,
        repetition_penalty=1.0,
    )

    # Run upstream generation to get talker codes.
    gen_kwargs = dict(
        do_sample=False,
        top_k=0,
        top_p=1.0,
        temperature=1.0,
        repetition_penalty=1.0,
        max_new_tokens=64,
        min_new_tokens=0,
        subtalker_dosample=False,
        subtalker_top_k=0,
        subtalker_top_p=1.0,
        subtalker_temperature=1.0,
    )
    talker_codes_list, _ = base.model.generate(
        input_ids=input_ids_base,
        ref_ids=[None],
        voice_clone_prompt=vcp,
        languages=["English"],
        non_streaming_mode=False,
        **gen_kwargs,
    )

    upstream_codes = talker_codes_list[0].detach().cpu()
    fast_codes_cpu = fast_codes.detach().cpu()

    min_len = min(upstream_codes.shape[0], fast_codes_cpu.shape[0])
    mismatch = None
    for i in range(min_len):
        if not torch.equal(upstream_codes[i], fast_codes_cpu[i]):
            mismatch = i
            break
    # Allow tiny drift in long runs due to static cache + CUDA graph numerics,
    # but enforce exact parity for the initial chunk where artifacts are most audible.
    if mismatch is not None:
        assert mismatch >= 32

    if mismatch is None and upstream_codes.shape[0] != fast_codes_cpu.shape[0]:
        eos_id = base.model.config.talker_config.codec_eos_token_id
        if upstream_codes.shape[0] > fast_codes_cpu.shape[0]:
            assert upstream_codes[min_len, 0].item() == eos_id
        else:
            assert fast_codes_cpu[min_len, 0].item() == eos_id


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for streaming parity test.")
def test_streaming_matches_non_streaming_prefix(parity_fixture):
    from faster_qwen3_tts.generate import fast_generate
    from faster_qwen3_tts.streaming import fast_generate_streaming

    fast = parity_fixture["fast"]
    tie = parity_fixture["tie"]
    tam = parity_fixture["tam"]
    tth = parity_fixture["tth"]
    tpe = parity_fixture["tpe"]

    full_codes, _ = fast_generate(
        talker=fast.model.model.talker,
        talker_input_embeds=tie,
        attention_mask=tam,
        trailing_text_hiddens=tth,
        tts_pad_embed=tpe,
        config=fast.model.model.config.talker_config,
        predictor_graph=fast.predictor_graph,
        talker_graph=fast.talker_graph,
        max_new_tokens=48,
        min_new_tokens=0,
        do_sample=False,
        top_k=0,
        top_p=1.0,
        temperature=1.0,
        repetition_penalty=1.0,
    )

    chunks = []
    for chunk, _ in fast_generate_streaming(
        talker=fast.model.model.talker,
        talker_input_embeds=tie,
        attention_mask=tam,
        trailing_text_hiddens=tth,
        tts_pad_embed=tpe,
        config=fast.model.model.config.talker_config,
        predictor_graph=fast.predictor_graph,
        talker_graph=fast.talker_graph,
        max_new_tokens=48,
        min_new_tokens=0,
        do_sample=False,
        top_k=0,
        top_p=1.0,
        temperature=1.0,
        repetition_penalty=1.0,
        chunk_size=8,
    ):
        chunks.append(chunk)

    stream_codes = torch.cat(chunks, dim=0)
    min_len = min(full_codes.shape[0], stream_codes.shape[0])
    assert torch.equal(full_codes[:min_len].cpu(), stream_codes[:min_len].cpu())
