from examples.ray_dual_worker_server import (
    AppState,
    CapsWriterSpeakRequest,
    SpeechRequest,
    PIPELINE_VERSION,
    _hit_token_cap,
    _next_retry_tokens,
    _result_quality_issues,
    _stable_language_for_tts,
    _status_payload,
    _estimate_max_new_tokens,
    _resolve_max_new_tokens,
    _synthesize_with_quality_retry,
    _trim_trailing_silence,
    _to_wav_bytes,
)
import asyncio
import numpy as np
import pytest


def test_resolve_max_new_tokens_uses_estimate_when_missing():
    text = "这是一个用于测试的短句。"

    assert _resolve_max_new_tokens(text, None, 512) == _estimate_max_new_tokens(text, 512)
    assert _resolve_max_new_tokens(text, None, 32) == 32


def test_resolve_max_new_tokens_clamps_client_value():
    assert _resolve_max_new_tokens("短句。", 1, 512) == 64
    assert _resolve_max_new_tokens("短句。", 999, 512) == 512
    assert _resolve_max_new_tokens("短句。", 999, 32) == 32


def test_short_chunk_default_stays_below_global_cap():
    text = "这是一个六十字以内的长文分段，用来避免默认直接放到五百一十二个 token 导致异常长音频。"

    assert _resolve_max_new_tokens(text, None, 512) < 512


def test_hit_token_cap_uses_actual_token_audio_frame_duration():
    assert _hit_token_cap(11.76, 147) is True
    assert _hit_token_cap(17.60, 220) is True
    assert _hit_token_cap(10.0, 147) is False


def test_quality_issues_detect_token_cap_and_long_silence():
    silence = np.zeros(24000 * 3, dtype=np.float32)

    issues = _result_quality_issues(
        {
            "bytes": _to_wav_bytes(silence, 24000),
            "hit_token_cap": True,
            "suspicious_duration": False,
        }
    )

    assert "hit_token_cap" in issues
    assert "empty_audio" in issues
    assert "long_silence" in issues


def test_quality_issues_detect_suspicious_short_duration():
    sample_rate = 24000
    t = np.arange(sample_rate * 3, dtype=np.float32) / sample_rate
    tone = 0.08 * np.sin(2 * np.pi * 220 * t)

    issues = _result_quality_issues(
        {
            "bytes": _to_wav_bytes(tone, sample_rate),
            "hit_token_cap": False,
            "suspicious_duration": False,
            "estimated_audio_s": 12.0,
        }
    )

    assert "suspicious_short_duration" in issues


def test_quality_issues_detect_mid_length_early_stop():
    sample_rate = 24000
    t = np.arange(int(sample_rate * 1.55), dtype=np.float32) / sample_rate
    tone = 0.08 * np.sin(2 * np.pi * 220 * t)

    issues = _result_quality_issues(
        {
            "bytes": _to_wav_bytes(tone, sample_rate),
            "hit_token_cap": False,
            "suspicious_duration": False,
            "estimated_audio_s": 4.72,
        }
    )

    assert "suspicious_short_duration" in issues


def test_next_retry_tokens_grows_with_hard_cap():
    assert _next_retry_tokens(147, 512) == 221
    assert _next_retry_tokens(400, 512) == 512


def test_stable_language_defaults_chinese_for_chinese_text(monkeypatch):
    monkeypatch.delenv("QWEN_TTS_STABLE_CHINESE_DEFAULT", raising=False)

    assert _stable_language_for_tts("请读 A I。", None) == "Chinese"
    assert _stable_language_for_tts("Please read AI.", None) is None
    assert _stable_language_for_tts("请读 A I。", "Auto") == "Auto"


def test_trim_trailing_silence_keeps_voice_and_short_pause():
    sample_rate = 24000
    t = np.arange(sample_rate, dtype=np.float32) / sample_rate
    tone = 0.1 * np.sin(2 * np.pi * 440 * t)
    silence = np.zeros(sample_rate * 3, dtype=np.float32)
    audio = np.concatenate([tone, silence])

    trimmed, trimmed_s = _trim_trailing_silence(audio, sample_rate, keep_s=0.5)

    assert trimmed_s >= 2.0
    assert len(trimmed) <= int(sample_rate * 1.6)


def test_speech_requests_accept_optional_max_new_tokens():
    capswriter_req = CapsWriterSpeakRequest(text="你好。", max_new_tokens=96)
    openai_req = SpeechRequest(input="你好。", max_new_tokens=96)

    assert capswriter_req.max_new_tokens == 96
    assert openai_req.max_new_tokens == 96


def test_speech_requests_accept_optional_trace_id():
    capswriter_req = CapsWriterSpeakRequest(text="你好。", trace_id="trace-a")
    openai_req = SpeechRequest(input="你好。", trace_id="trace-b")

    assert capswriter_req.trace_id == "trace-a"
    assert openai_req.trace_id == "trace-b"


def test_speech_requests_accept_optional_affinity_key():
    capswriter_req = CapsWriterSpeakRequest(text="你好。", affinity_key="announcement:1")
    openai_req = SpeechRequest(input="你好。", affinity_key="announcement:2")

    assert capswriter_req.affinity_key == "announcement:1"
    assert openai_req.affinity_key == "announcement:2"


def test_app_state_uses_stable_worker_for_affinity_key():
    workers = [object(), object(), object()]
    app_state = AppState(workers)

    first = app_state.pick_worker("announcement:stable")
    assert app_state.pick_worker("announcement:stable") is first
    assert app_state.pick_worker("announcement:stable") is first


def test_status_payload_adds_v1_capabilities_without_removing_existing_fields():
    rows = [
        {
            "worker_id": 0,
            "status": "ready",
            "gpu_ids": ["0"],
            "default_speaker": "serena",
            "speakers": ["serena", "aiden"],
        },
        {
            "worker_id": 1,
            "status": "ready",
            "gpu_ids": ["1"],
            "default_speaker": "serena",
            "speakers": ["serena", "dylan"],
        },
    ]

    payload = _status_payload(rows, "ray_dual_worker")

    assert payload["success"] is True
    assert payload["tts_backend"] == "ray_dual_worker"
    assert payload["workers_ready"] == 2
    assert payload["workers"] == rows
    assert payload["api_version"] == "tts-http-v1"
    assert payload["speakers"] == ["serena", "aiden", "dylan"]
    assert payload["capabilities"]["supports_plan"] is True
    assert payload["capabilities"]["supports_trace_id"] is True
    assert payload["capabilities"]["supports_affinity_key"] is True
    assert payload["pipeline_version"] == PIPELINE_VERSION
    assert payload["generation"]["do_sample"] is False
    assert payload["client_defaults"]["recommended_speak_concurrency"] == 2
    assert payload["audio"]["content_type"] == "audio/wav"


def test_quality_retry_raises_when_all_attempts_and_fallback_fail(monkeypatch):
    import examples.ray_dual_worker_server as server

    class FakeSynth:
        def remote(self, _text, _speaker, _language, _instruction, max_new_tokens):
            return max_new_tokens

    class FakeWorker:
        synthesize = FakeSynth()

    async def fake_ray_get(max_new_tokens):
        silence = np.zeros(24000 * 3, dtype=np.float32)
        return {
            "worker_id": 0,
            "gpu_ids": ["0"],
            "audio_s": 3.0,
            "elapsed_s": 0.5,
            "ttfa_s": 0.1,
            "rtf": 6.0,
            "max_new_tokens": max_new_tokens,
            "estimated_audio_s": 8.0,
            "hit_token_cap": False,
            "suspicious_duration": False,
            "bytes": _to_wav_bytes(silence, 24000),
        }

    async def no_split_fallback(**_kwargs):
        return None

    monkeypatch.setattr(server, "state", AppState([FakeWorker(), FakeWorker(), FakeWorker()]))
    monkeypatch.setattr(server, "_ray_get", fake_ray_get)
    monkeypatch.setattr(server, "_synthesize_split_fallback", no_split_fallback)

    with pytest.raises(RuntimeError, match="TTS quality check failed"):
        asyncio.run(
            _synthesize_with_quality_retry(
                text="风险是，若配置错误，可能引发并发混乱。",
                speaker="serena",
                language="Chinese",
                instruction=None,
                max_new_tokens=96,
                trace_id="test-quality-fail",
            )
        )
