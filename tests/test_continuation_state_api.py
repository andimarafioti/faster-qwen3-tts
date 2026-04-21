import types

import pytest
import torch

from faster_qwen3_tts.continuation import (
    apply_continuation_state_delta,
    build_continuation_state_status,
)
from faster_qwen3_tts.model import FasterQwen3TTS
from faster_qwen3_tts.streaming import parity_generate_streaming


def _dummy_graph():
    return types.SimpleNamespace(num_layers=2, max_seq_len=64, hidden_size=4)


def _build_wrapper():
    base = types.SimpleNamespace()
    base.model = types.SimpleNamespace(
        talker=types.SimpleNamespace(rope_deltas=None),
        config=types.SimpleNamespace(talker_config=types.SimpleNamespace()),
        tts_model_size="1b7",
        tts_model_type="base",
    )
    base._validate_languages = lambda _languages: None
    base._validate_speakers = lambda _speakers: None
    model = FasterQwen3TTS(base, _dummy_graph(), _dummy_graph(), device="cpu", dtype=torch.float32)
    model._warmup = lambda _prefill_len: setattr(model, "_warmed_up", True)
    return model


def _delta(*, base_seq_len, added_seq_len, value):
    key = torch.full((1, 1, added_seq_len, 2), value, dtype=torch.float32)
    val = torch.full((1, 1, added_seq_len, 2), value + 10, dtype=torch.float32)
    return {
        "version": 1,
        "state_kind": "delta",
        "mode": "custom_voice",
        "non_streaming_mode": False,
        "model_signature": {
            "num_layers": 2,
            "max_seq_len": 64,
            "hidden_size": 4,
        },
        "base_seq_len": base_seq_len,
        "seq_len": base_seq_len + added_seq_len,
        "added_seq_len": added_seq_len,
        "cache_delta": [
            {"key": key.clone(), "value": val.clone()},
            {"key": key.clone() + 1, "value": val.clone() + 1},
        ],
        "rope_deltas": torch.zeros(1, 1, dtype=torch.float32),
        "first_codebook_history_delta": torch.tensor([value, value + 1], dtype=torch.long),
        "codec_ids_delta": torch.full((2, 16), value, dtype=torch.long),
    }


def test_apply_continuation_state_delta_builds_and_extends_state():
    state = apply_continuation_state_delta(None, _delta(base_seq_len=0, added_seq_len=3, value=5))
    assert state["state_kind"] == "full"
    assert state["seq_len"] == 3
    assert state["cache"][0]["key"].shape == (1, 1, 3, 2)
    assert state["first_codebook_history"].tolist() == [5, 6]
    assert state["decoder_context_codes"].shape == (2, 16)

    state = apply_continuation_state_delta(state, _delta(base_seq_len=3, added_seq_len=2, value=9))
    assert state["seq_len"] == 5
    assert state["cache"][0]["key"].shape == (1, 1, 5, 2)
    assert state["first_codebook_history"].tolist() == [5, 6, 9, 10]
    assert torch.all(state["cache"][0]["key"][:, :, :3, :] == 5)
    assert torch.all(state["cache"][0]["key"][:, :, 3:, :] == 9)


def test_apply_continuation_state_delta_rejects_nonzero_base_without_state():
    with pytest.raises(ValueError, match="base_seq_len is non-zero"):
        apply_continuation_state_delta(None, _delta(base_seq_len=3, added_seq_len=2, value=5))


def test_apply_continuation_state_delta_rejects_mode_mismatch():
    state = apply_continuation_state_delta(None, _delta(base_seq_len=0, added_seq_len=2, value=5))
    delta = _delta(base_seq_len=2, added_seq_len=1, value=7)
    delta["mode"] = "voice_clone"

    with pytest.raises(ValueError, match="mode mismatch"):
        apply_continuation_state_delta(state, delta)


def test_apply_continuation_state_delta_rejects_version_mismatch():
    delta = _delta(base_seq_len=0, added_seq_len=2, value=5)
    delta["version"] = 999

    with pytest.raises(ValueError, match="Unsupported continuation delta version"):
        apply_continuation_state_delta(None, delta)


def test_apply_continuation_state_delta_rejects_seq_len_mismatch():
    state = apply_continuation_state_delta(None, _delta(base_seq_len=0, added_seq_len=2, value=5))

    with pytest.raises(ValueError, match="base_seq_len"):
        apply_continuation_state_delta(state, _delta(base_seq_len=3, added_seq_len=1, value=7))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for mixed-device merge test")
def test_apply_continuation_state_delta_moves_delta_to_state_device():
    state = apply_continuation_state_delta(None, _delta(base_seq_len=0, added_seq_len=2, value=5))
    delta = _delta(base_seq_len=2, added_seq_len=1, value=7)
    delta["rope_deltas"] = delta["rope_deltas"].to("cuda")
    delta["first_codebook_history_delta"] = delta["first_codebook_history_delta"].to("cuda")
    delta["codec_ids_delta"] = delta["codec_ids_delta"].to("cuda")
    for layer in delta["cache_delta"]:
        layer["key"] = layer["key"].to("cuda")
        layer["value"] = layer["value"].to("cuda")

    merged = apply_continuation_state_delta(state, delta)

    assert merged["rope_deltas"].device.type == "cpu"
    assert merged["first_codebook_history"].device.type == "cpu"
    assert merged["decoder_context_codes"].device.type == "cpu"


def test_build_continuation_state_status_warns_near_limit():
    status = build_continuation_state_status(seq_len=1800, max_seq_len=2048)
    assert status["remaining_tokens"] == 247
    assert status["should_reset"] is True
    assert "warning" in status


def test_generate_voice_clone_returns_info_when_continuation_requested(monkeypatch):
    model = _build_wrapper()
    timing = {
        "prefill_ms": 10.0,
        "decode_s": 0.02,
        "steps": 3,
        "ms_per_step": 1.0,
        "steps_per_s": 1000.0,
        "continuation_state_delta": {"delta": True},
        "continuation_state_status": {"remaining_tokens": 123},
    }

    monkeypatch.setattr(
        model,
        "_prepare_generation",
        lambda **_kwargs: (
            types.SimpleNamespace(
                speech_tokenizer=types.SimpleNamespace(
                    decode=lambda _payload: ([torch.arange(12, dtype=torch.float32)], 24000)
                )
            ),
            object(),
            object(),
            torch.zeros(1, 1, 1),
            torch.ones(1, 1, dtype=torch.long),
            torch.zeros(1, 1, 1),
            torch.zeros(1, 1, 1),
            None,
        ),
    )

    monkeypatch.setattr(
        "faster_qwen3_tts.generate.fast_generate",
        lambda **_kwargs: (torch.zeros(3, 16, dtype=torch.long), timing),
    )

    audio, sr, info = model.generate_voice_clone(
        text="hello",
        language="English",
        voice_clone_prompt={"ref_spk_embedding": [torch.zeros(1, 4)]},
        return_continuation_state=True,
    )

    assert sr == 24000
    assert len(audio) == 1
    assert info["continuation_state_delta"] == {"delta": True}
    assert info["continuation_state_status"] == {"remaining_tokens": 123}


def test_generate_custom_voice_streaming_keeps_continuation_delta_in_timing(monkeypatch):
    model = _build_wrapper()
    model.model.model.tts_model_type = "custom_voice"

    monkeypatch.setattr(
        model,
        "_prepare_generation_custom",
        lambda **_kwargs: (
            types.SimpleNamespace(
                speech_tokenizer=types.SimpleNamespace(
                    decode=lambda _payload: ([torch.arange(16, dtype=torch.float32)], 24000)
                )
            ),
            object(),
            object(),
            torch.zeros(1, 1, 1),
            torch.ones(1, 1, dtype=torch.long),
            torch.zeros(1, 1, 1),
            torch.zeros(1, 1, 1),
        ),
    )

    def _fake_stream(**_kwargs):
        yield torch.zeros(2, 16, dtype=torch.long), {
            "chunk_index": 0,
            "chunk_steps": 2,
            "prefill_ms": 0.0,
            "decode_ms": 1.0,
            "total_steps_so_far": 2,
            "is_final": True,
            "continuation_state_delta": {"delta": True},
            "continuation_state_status": {"remaining_tokens": 55},
        }

    monkeypatch.setattr("faster_qwen3_tts.streaming.fast_generate_streaming", _fake_stream)

    chunk, sr, timing = next(
        model.generate_custom_voice_streaming(
            text="hello",
            speaker="speaker_a",
            language="English",
            return_continuation_state=True,
        )
    )

    assert sr == 24000
    assert chunk.ndim == 1
    assert timing["continuation_state_delta"] == {"delta": True}
    assert timing["continuation_state_status"] == {"remaining_tokens": 55}


def test_parity_generate_streaming_smoke_with_continuation(monkeypatch):
    delta = _delta(base_seq_len=0, added_seq_len=2, value=5)
    delta["mode"] = "voice_clone"
    state = apply_continuation_state_delta(None, delta)

    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(
        "faster_qwen3_tts.streaming.prefill_with_continuation",
        lambda **_kwargs: (
            types.SimpleNamespace(
                past_key_values=types.SimpleNamespace(layers=[]),
                past_hidden=torch.zeros(1, 1, 4),
                generation_step=0,
                logits=torch.zeros(1, 1, 32),
            ),
            torch.ones(1, state["seq_len"] + 1, dtype=torch.long),
            state["seq_len"],
        ),
    )

    talker = types.SimpleNamespace(
        config=types.SimpleNamespace(num_hidden_layers=2, hidden_size=4),
        rope_deltas=torch.zeros(1, 1),
    )
    config = types.SimpleNamespace(codec_eos_token_id=31, vocab_size=32)

    result = list(
        parity_generate_streaming(
            talker=talker,
            talker_input_embeds=torch.zeros(1, 1, 4),
            attention_mask=torch.ones(1, 1, dtype=torch.long),
            trailing_text_hiddens=torch.zeros(1, 1, 4),
            tts_pad_embed=torch.zeros(1, 1, 4),
            config=config,
            max_new_tokens=0,
            do_sample=False,
            continuation_state=state,
            continuation_mode="voice_clone",
            continuation_non_streaming_mode=False,
            continuation_max_seq_len=64,
        )
    )

    assert result == []
