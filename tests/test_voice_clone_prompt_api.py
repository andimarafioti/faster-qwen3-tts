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
    base._build_instruct_text = lambda text: text
    base._tokenize_texts = lambda _texts: [torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)]

    def _fail(*_args, **_kwargs):
        raise AssertionError("create_voice_clone_prompt should not be called when voice_clone_prompt is provided")

    base.create_voice_clone_prompt = _fail
    base._prompt_items_to_voice_clone_prompt = lambda prompt_items: {
        "ref_code": [item.ref_code for item in prompt_items],
        "ref_spk_embedding": [item.ref_spk_embedding for item in prompt_items],
        "x_vector_only_mode": [item.x_vector_only_mode for item in prompt_items],
        "icl_mode": [item.icl_mode for item in prompt_items],
    }

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
    assert list(sig_clone.parameters).index("max_new_tokens") == 5
    assert list(sig_stream.parameters).index("max_new_tokens") == 5
    assert list(sig_clone.parameters)[-1] == "voice_clone_prompt"
    assert list(sig_stream.parameters)[-1] == "voice_clone_prompt"
    assert sig_clone.parameters["xvec_only"].default is False
    assert sig_clone.parameters["non_streaming_mode"].default is False
    assert sig_stream.parameters["xvec_only"].default is False
    assert sig_stream.parameters["non_streaming_mode"].default is False


def test_prepare_generation_uses_precomputed_xvec_prompt_without_prompt_extraction():
    model = _build_dummy_model()

    _m, _talker, _config, _tie, _tam, _tth, _tpe, ref_codes = model._prepare_generation(
        text="hello",
        ref_audio=None,
        ref_text="",
        language="English",
        xvec_only=False,  # ignored when voice_clone_prompt is provided
        voice_clone_prompt=_xvec_prompt(),
    )
    assert ref_codes is None


def test_prepare_generation_warns_for_instruct_with_xvec_only(caplog):
    model = _build_dummy_model()

    with caplog.at_level("WARNING"):
        model._prepare_generation(
            text="hello",
            ref_audio=None,
            ref_text="",
            language="English",
            voice_clone_prompt=_xvec_prompt(),
            instruct="Please speak very fast.",
        )

    assert "experimental" in caplog.text
    assert "x-vector-only" in caplog.text


def test_prepare_generation_rejects_missing_voice_clone_prompt_keys():
    model = _build_dummy_model()
    bad_prompt = {}

    with pytest.raises(ValueError, match="missing required keys"):
        model._prepare_generation(
            text="hello",
            ref_audio=None,
            ref_text="",
            language="English",
            voice_clone_prompt=bad_prompt,
        )


def test_prepare_generation_accepts_icl_prompt_with_ref_text():
    model = _build_dummy_model()
    icl_prompt = {
        "ref_spk_embedding": [torch.zeros(1, 4, dtype=torch.float32)],
        "x_vector_only_mode": [False],
        "icl_mode": [True],
        "ref_code": [torch.zeros(2, 16, dtype=torch.long)],
    }

    _m, _talker, _config, _tie, _tam, _tth, _tpe, ref_codes = model._prepare_generation(
        text="hello",
        ref_audio=None,
        ref_text="reference text",
        language="English",
        voice_clone_prompt=icl_prompt,
    )
    assert ref_codes is not None


def test_prepare_generation_accepts_upstream_prompt_items():
    model = _build_dummy_model()
    prompt_items = [
        types.SimpleNamespace(
            ref_code=torch.zeros(2, 16, dtype=torch.long),
            ref_spk_embedding=torch.zeros(1, 4, dtype=torch.float32),
            x_vector_only_mode=False,
            icl_mode=True,
            ref_text="reference text from prompt item",
        )
    ]

    _m, _talker, _config, _tie, _tam, _tth, _tpe, ref_codes = model._prepare_generation(
        text="hello",
        ref_audio=None,
        ref_text="",
        language="English",
        voice_clone_prompt=prompt_items,
    )

    assert ref_codes is not None


def test_prepare_generation_ignores_ref_text_with_precomputed_prompt():
    model = _build_dummy_model()
    _m, _talker, _config, _tie, _tam, _tth, _tpe, ref_codes = model._prepare_generation(
        text="hello",
        ref_audio=None,
        ref_text="this should be ignored",
        language="English",
        voice_clone_prompt=_xvec_prompt(),
    )
    assert ref_codes is None


def test_prepare_generation_icl_prompt_requires_ref_text():
    model = _build_dummy_model()
    icl_prompt = {
        "ref_spk_embedding": [torch.zeros(1, 4, dtype=torch.float32)],
        "x_vector_only_mode": [False],
        "icl_mode": [True],
        "ref_code": [torch.zeros(2, 16, dtype=torch.long)],
    }

    with pytest.raises(ValueError, match="ref_text is required"):
        model._prepare_generation(
            text="hello",
            ref_audio=None,
            ref_text="",
            language="English",
            voice_clone_prompt=icl_prompt,
        )


def test_prepare_generation_requires_ref_audio_without_precomputed_prompt():
    model = _build_dummy_model()
    with pytest.raises(ValueError, match="ref_audio is required"):
        model._prepare_generation(
            text="hello",
            ref_audio=None,
            ref_text="",
            language="English",
            voice_clone_prompt=None,
        )
