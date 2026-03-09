import inspect
import types

import pytest
import torch

from faster_qwen3_tts.model import FasterQwen3TTS


def _dummy_graph():
    return object()


def _build_dummy_model():
    base = types.SimpleNamespace()
    base.model = types.SimpleNamespace(
        talker=types.SimpleNamespace(rope_deltas=None),
        config=types.SimpleNamespace(talker_config=types.SimpleNamespace()),
    )
    base._build_assistant_text = lambda text: text
    base._build_ref_text = lambda text: text
    base._tokenize_texts = lambda _texts: [torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)]

    def _fail(*_args, **_kwargs):
        raise AssertionError("create_voice_clone_prompt should not be called when voice_clone_prompt is provided")

    base.create_voice_clone_prompt = _fail
    base._prompt_items_to_voice_clone_prompt = _fail

    model = FasterQwen3TTS(base, _dummy_graph(), _dummy_graph(), device="cpu", dtype=torch.float32)
    model._build_talker_inputs_local = lambda **_kwargs: (
        torch.zeros(1, 10, 4, dtype=torch.float32),
        torch.ones(1, 10, dtype=torch.long),
        torch.zeros(1, 1, 4, dtype=torch.float32),
        torch.zeros(1, 1, 4, dtype=torch.float32),
    )
    model._warmup = lambda _prefill_len: setattr(model, "_warmed_up", True)
    return model


def _xvec_prompt():
    return {
        "ref_spk_embedding": [torch.zeros(1, 4, dtype=torch.float32)],
    }


def test_public_api_exposes_voice_clone_prompt_parameter():
    sig_clone = inspect.signature(FasterQwen3TTS.generate_voice_clone)
    sig_stream = inspect.signature(FasterQwen3TTS.generate_voice_clone_streaming)
    assert "voice_clone_prompt" in sig_clone.parameters
    assert "voice_clone_prompt" in sig_stream.parameters


def test_prepare_generation_uses_precomputed_xvec_prompt_without_prompt_extraction():
    model = _build_dummy_model()

    _m, _talker, _config, _tie, _tam, _tth, _tpe, ref_codes = model._prepare_generation(
        text="hello",
        ref_audio="ignored.wav",
        ref_text="",
        language="English",
        xvec_only=False,  # ignored when voice_clone_prompt is provided
        voice_clone_prompt=_xvec_prompt(),
    )
    assert ref_codes is None


def test_prepare_generation_rejects_missing_voice_clone_prompt_keys():
    model = _build_dummy_model()
    bad_prompt = {}

    with pytest.raises(ValueError, match="missing required keys"):
        model._prepare_generation(
            text="hello",
            ref_audio="ignored.wav",
            ref_text="",
            language="English",
            voice_clone_prompt=bad_prompt,
        )


def test_prepare_generation_rejects_icl_mode_in_precomputed_prompt():
    model = _build_dummy_model()
    icl_prompt = _xvec_prompt()
    icl_prompt["icl_mode"] = [True]

    with pytest.raises(ValueError, match="x_vector_only prompts only"):
        model._prepare_generation(
            text="hello",
            ref_audio="ignored.wav",
            ref_text="",
            language="English",
            voice_clone_prompt=icl_prompt,
        )


def test_prepare_generation_ignores_ref_text_with_precomputed_prompt():
    model = _build_dummy_model()
    _m, _talker, _config, _tie, _tam, _tth, _tpe, ref_codes = model._prepare_generation(
        text="hello",
        ref_audio="ignored.wav",
        ref_text="this should be ignored",
        language="English",
        voice_clone_prompt=_xvec_prompt(),
    )
    assert ref_codes is None
