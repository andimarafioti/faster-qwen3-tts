import sys
import types
import warnings
from pathlib import Path

import numpy as np
import pytest

from faster_qwen3_tts import FasterQwen3TTS
from faster_qwen3_tts.cli import build_parser
import faster_qwen3_tts.ggml_backend as ggml_backend
from faster_qwen3_tts.ggml_backend import GGMLQwen3TTS


class _FakeVoiceRef:
    def __init__(self, spk=None, codes=None):
        self.ref_spk_emb = np.array([4.0, 5.0, 6.0], dtype=np.float32) if spk is None else spk
        self.ref_codes = (
            np.arange(8, dtype=np.int32).reshape(4, 2)
            if codes is None
            else codes
        )

    def save(self, spk_path, rvq_path):
        Path(spk_path).write_bytes(b"spk")
        Path(rvq_path).write_bytes(b"rvq")


class _FakeRuntime:
    def __init__(self):
        self.calls = []
        self.last_extract_voice_ref_profile = None

    def synthesize(self, **kwargs):
        self.calls.append(("synthesize", kwargs))
        return np.array([0.0, 0.25], dtype=np.float32), 24000

    def stream(self, *, codec_chunk_sec, **kwargs):
        self.calls.append(("stream", {"codec_chunk_sec": codec_chunk_sec, **kwargs}))
        yield np.array([0.0, 0.25], dtype=np.float32), 24000

    def load_rvq_codes(self, path):
        self.calls.append(("load_rvq_codes", {"path": path}))
        return np.arange(8, dtype=np.int32).reshape(4, 2)

    def extract_voice_ref(self, ref_audio_24k):
        self.calls.append(("extract_voice_ref", {"n_samples": int(ref_audio_24k.shape[0])}))
        self.last_extract_voice_ref_profile = {"native_extract_ms": 12.5}
        return _FakeVoiceRef()

    def load_voice_ref(self, spk_path, rvq_path):
        self.calls.append(("load_voice_ref", {"spk_path": spk_path, "rvq_path": rvq_path}))
        assert Path(spk_path).is_file()
        assert Path(rvq_path).is_file()
        return _FakeVoiceRef(
            spk=np.array([9.0, 10.0, 11.0], dtype=np.float32),
            codes=np.arange(8, 16, dtype=np.int32).reshape(4, 2),
        )

    def version(self):
        return "fake-native"

    def speaker_names(self):
        return ["aiden", "vivian"]


@pytest.fixture
def qwentts_cpp_stub(monkeypatch):
    module = types.SimpleNamespace(
        QwenTTS=object,
        load_speaker_embedding=lambda _path: np.array([1.0, 2.0, 3.0], dtype=np.float32),
    )
    monkeypatch.setitem(sys.modules, "qwentts_cpp", module)
    return module


def test_cached_speaker_only_forwards_spk_embedding(qwentts_cpp_stub):
    runtime = _FakeRuntime()
    model = GGMLQwen3TTS(runtime)

    audio_list, sr = model.generate_voice_clone(
        text="hello",
        language="English",
        ref_spk="speaker.spk",
        xvec_only=True,
    )

    assert sr == 24000
    np.testing.assert_array_equal(audio_list[0], np.array([0.0, 0.25], dtype=np.float32))
    _name, kwargs = runtime.calls[0]
    np.testing.assert_array_equal(
        kwargs["ref_spk_emb"],
        np.array([1.0, 2.0, 3.0], dtype=np.float32),
    )
    assert kwargs["ref_text"] is None
    assert "ref_audio_24k" not in kwargs
    assert "ref_codes" not in kwargs


def test_cached_icl_forwards_spk_rvq_and_ref_text(qwentts_cpp_stub):
    runtime = _FakeRuntime()
    model = GGMLQwen3TTS(runtime)

    model.generate_voice_clone(
        text="hello",
        language="English",
        ref_spk="speaker.spk",
        ref_rvq="reference.rvq",
        ref_text="reference transcript",
    )

    assert runtime.calls[0] == ("load_rvq_codes", {"path": "reference.rvq"})
    _name, kwargs = runtime.calls[1]
    np.testing.assert_array_equal(kwargs["ref_codes"], np.arange(8, dtype=np.int32).reshape(4, 2))
    assert kwargs["ref_text"] == "reference transcript"


def test_cached_streaming_forwards_adapter_timing(qwentts_cpp_stub):
    runtime = _FakeRuntime()
    model = GGMLQwen3TTS(runtime)

    chunk, sr, timing = next(
        model.generate_voice_clone_streaming(
            text="hello",
            language="English",
            ref_spk_emb=np.ones(3, dtype=np.float32),
            chunk_size=4,
        )
    )

    assert sr == 24000
    np.testing.assert_array_equal(chunk, np.array([0.0, 0.25], dtype=np.float32))
    assert timing["adapter_prepare_ms"] >= 0.0
    _name, kwargs = runtime.calls[0]
    assert kwargs["codec_chunk_sec"] == pytest.approx(4 / 12.5)
    np.testing.assert_array_equal(kwargs["ref_spk_emb"], np.ones(3, dtype=np.float32))


@pytest.mark.parametrize(
    ("method_name", "kwargs", "streaming"),
    [
        (
            "generate_voice_clone",
            {"text": "hello", "language": "English", "ref_spk_emb": np.ones(3, dtype=np.float32)},
            False,
        ),
        (
            "generate_voice_clone_streaming",
            {"text": "hello", "language": "English", "ref_spk_emb": np.ones(3, dtype=np.float32)},
            True,
        ),
        (
            "generate_custom_voice",
            {"text": "hello", "speaker": "aiden", "language": "English"},
            False,
        ),
        (
            "generate_custom_voice_streaming",
            {"text": "hello", "speaker": "aiden", "language": "English"},
            True,
        ),
        (
            "generate_voice_design",
            {"text": "hello", "instruct": "warm voice", "language": "English"},
            False,
        ),
        (
            "generate_voice_design_streaming",
            {"text": "hello", "instruct": "warm voice", "language": "English"},
            True,
        ),
    ],
)
def test_ggml_warns_when_non_prefill_text_feed_is_requested(
    qwentts_cpp_stub,
    method_name,
    kwargs,
    streaming,
):
    model = GGMLQwen3TTS(_FakeRuntime())

    with pytest.warns(RuntimeWarning, match="step-by-step text feeding"):
        result = getattr(model, method_name)(non_streaming_mode=False, **kwargs)
        if streaming:
            next(result)


def test_ggml_does_not_warn_for_prefill_text_feed(qwentts_cpp_stub):
    model = GGMLQwen3TTS(_FakeRuntime())

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.generate_custom_voice(
            text="hello",
            speaker="aiden",
            language="English",
            non_streaming_mode=True,
        )

    assert not [
        warning for warning in caught
        if "step-by-step text feeding" in str(warning.message)
    ]


def test_raw_ref_audio_uses_memory_voice_ref_cache(monkeypatch, tmp_path):
    ref_audio_24k = np.array([0.0, 0.25, -0.25], dtype=np.float32)
    monkeypatch.setattr(
        ggml_backend,
        "_load_ref_audio_24k",
        lambda *_args, **_kwargs: ref_audio_24k,
    )
    runtime = _FakeRuntime()
    model = GGMLQwen3TTS(runtime, model_identity="fake-model", voice_ref_cache_dir=tmp_path)

    first = next(
        model.generate_voice_clone_streaming(
            text="hello",
            language="English",
            ref_audio="reference.wav",
            ref_text="reference transcript",
            chunk_size=4,
        )
    )
    second = next(
        model.generate_voice_clone_streaming(
            text="hello again",
            language="English",
            ref_audio="reference.wav",
            ref_text="reference transcript",
            chunk_size=4,
        )
    )

    assert [name for name, _ in runtime.calls].count("extract_voice_ref") == 1
    assert first[2]["adapter_profile"]["voice_ref_cache"] == "miss"
    assert second[2]["adapter_profile"]["voice_ref_cache"] == "memory"
    stream_calls = [kwargs for name, kwargs in runtime.calls if name == "stream"]
    np.testing.assert_array_equal(stream_calls[0]["ref_spk_emb"], np.array([4.0, 5.0, 6.0], dtype=np.float32))
    np.testing.assert_array_equal(stream_calls[0]["ref_codes"], np.arange(8, dtype=np.int32).reshape(4, 2))


def test_raw_ref_audio_reuses_disk_voice_ref_cache(monkeypatch, tmp_path):
    ref_audio_24k = np.array([0.0, 0.25, -0.25], dtype=np.float32)
    monkeypatch.setattr(
        ggml_backend,
        "_load_ref_audio_24k",
        lambda *_args, **_kwargs: ref_audio_24k,
    )

    first_runtime = _FakeRuntime()
    first_model = GGMLQwen3TTS(first_runtime, model_identity="fake-model", voice_ref_cache_dir=tmp_path)
    first_model.generate_voice_clone(
        text="hello",
        language="English",
        ref_audio="reference.wav",
        ref_text="reference transcript",
    )

    second_runtime = _FakeRuntime()
    second_model = GGMLQwen3TTS(second_runtime, model_identity="fake-model", voice_ref_cache_dir=tmp_path)
    second_model.generate_voice_clone(
        text="hello",
        language="English",
        ref_audio="reference.wav",
        ref_text="reference transcript",
    )

    assert [name for name, _ in second_runtime.calls].count("extract_voice_ref") == 0
    assert [name for name, _ in second_runtime.calls].count("load_voice_ref") == 1
    assert second_model.last_adapter_profile["voice_ref_cache"] == "disk"
    _name, kwargs = second_runtime.calls[-1]
    np.testing.assert_array_equal(kwargs["ref_spk_emb"], np.array([9.0, 10.0, 11.0], dtype=np.float32))
    np.testing.assert_array_equal(kwargs["ref_codes"], np.arange(8, 16, dtype=np.int32).reshape(4, 2))


def test_cached_references_reject_raw_audio_mix(qwentts_cpp_stub):
    model = GGMLQwen3TTS(_FakeRuntime())

    with pytest.raises(ValueError, match="mutually exclusive"):
        model.generate_voice_clone(
            text="hello",
            language="English",
            ref_audio="reference.wav",
            ref_spk="speaker.spk",
        )


def test_cached_rvq_requires_ref_text(qwentts_cpp_stub):
    model = GGMLQwen3TTS(_FakeRuntime())

    with pytest.raises(ValueError, match="ref_text is required"):
        model.generate_voice_clone(
            text="hello",
            language="English",
            ref_spk="speaker.spk",
            ref_rvq="reference.rvq",
        )


def test_ggml_speaker_listing_uses_runtime():
    model = GGMLQwen3TTS(_FakeRuntime())

    assert model.get_supported_speakers() == ["aiden", "vivian"]


def test_ggml_public_warmup_is_a_no_op():
    runtime = _FakeRuntime()
    model = GGMLQwen3TTS(runtime)

    assert model.warmup(prefill_len=42) is None
    assert runtime.calls == []


def test_adapter_from_pretrained_forwards_qwentts_runtime_flags(monkeypatch, qwentts_cpp_stub):
    captured = {}

    class FakeQwenTTS:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return _FakeRuntime()

    qwentts_cpp_stub.QwenTTS = FakeQwenTTS

    model = GGMLQwen3TTS.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        quant="Q4_K_M",
        cache_dir=".cache/qwentts",
        local_files_only=True,
        library_path="libqwen.so",
        use_fa=False,
        clamp_fp16=True,
    )

    assert isinstance(model, GGMLQwen3TTS)
    assert captured["args"] == ("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",)
    assert captured["kwargs"] == {
        "quant": "Q4_K_M",
        "cache_dir": ".cache/qwentts",
        "local_files_only": True,
        "library_path": "libqwen.so",
        "use_fa": False,
        "clamp_fp16": True,
    }


def test_public_from_pretrained_forwards_qwentts_runtime_flags(monkeypatch):
    captured = {}
    sentinel = object()

    def fake_from_pretrained(cls, *args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return sentinel

    monkeypatch.setattr(
        GGMLQwen3TTS,
        "from_pretrained",
        classmethod(fake_from_pretrained),
    )

    result = FasterQwen3TTS.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        backend="ggml",
        quant="Q8_0",
        cache_dir=".cache/qwentts",
        local_files_only=True,
        qwentts_library_path="libqwen.so",
        qwentts_use_fa=False,
        qwentts_clamp_fp16=True,
        qwentts_ref_cache_dir=".cache/refs",
    )

    assert result is sentinel
    assert captured["args"] == ("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",)
    assert captured["kwargs"] == {
        "quant": "Q8_0",
        "cache_dir": ".cache/qwentts",
        "local_files_only": True,
        "library_path": "libqwen.so",
        "use_fa": False,
        "clamp_fp16": True,
        "voice_ref_cache_dir": ".cache/refs",
    }


def test_public_from_gguf_forwards_qwentts_runtime_flags(monkeypatch):
    captured = {}
    sentinel = object()

    def fake_from_gguf(cls, *args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return sentinel

    monkeypatch.setattr(
        GGMLQwen3TTS,
        "from_gguf",
        classmethod(fake_from_gguf),
    )

    result = FasterQwen3TTS.from_pretrained(
        "unused",
        backend="qwentts",
        gguf_talker_path="talker.gguf",
        gguf_codec_path="codec.gguf",
        qwentts_library_path="libqwen.so",
        qwentts_use_fa=False,
        qwentts_clamp_fp16=True,
        qwentts_ref_cache_dir=".cache/refs",
    )

    assert result is sentinel
    assert captured["args"] == ("talker.gguf", "codec.gguf")
    assert captured["kwargs"] == {
        "library_path": "libqwen.so",
        "use_fa": False,
        "clamp_fp16": True,
        "voice_ref_cache_dir": ".cache/refs",
    }


def test_cli_parses_qwentts_runtime_flags():
    parser = build_parser()

    args = parser.parse_args(
        [
            "--backend",
            "ggml",
            "--qwentts-no-fa",
            "--qwentts-clamp-fp16",
            "--qwentts-ref-cache-dir",
            ".cache/refs",
            "design",
            "--model",
            "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
            "--instruct",
            "warm voice",
            "--text",
            "hello",
            "--output",
            "out.wav",
        ]
    )

    assert args.qwentts_use_fa is False
    assert args.qwentts_clamp_fp16 is True
    assert args.qwentts_ref_cache_dir == ".cache/refs"
