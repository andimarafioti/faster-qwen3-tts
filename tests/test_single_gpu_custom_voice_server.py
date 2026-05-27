from examples.single_gpu_custom_voice_server import (
    CapsWriterSpeakRequest,
    SpeechRequest,
    _estimate_max_new_tokens,
    _hit_token_cap,
    _next_retry_tokens,
    _result_quality_issues,
    _resolve_max_new_tokens,
    _split_tts_text_into_chunks,
    _status_payload,
    _to_wav_bytes,
)
import numpy as np


def test_single_gpu_token_default_matches_estimate():
    text = "这是一个用于测试的短句。"

    assert _resolve_max_new_tokens(text, None, 512) == _estimate_max_new_tokens(text, 512)


def test_single_gpu_token_clamps_to_hard_cap():
    assert _resolve_max_new_tokens("短句。", 1, 512) == 64
    assert _resolve_max_new_tokens("短句。", 999, 512) == 512
    assert _resolve_max_new_tokens("短句。", 999, 32) == 32


def test_single_gpu_short_chunk_default_stays_below_global_cap():
    text = "这是一个六十字以内的长文分段，用来避免默认直接放到五百一十二个 token 导致异常长音频。"

    assert _resolve_max_new_tokens(text, None, 512) < 512


def test_single_gpu_hit_token_cap_uses_actual_token_audio_frame_duration():
    assert _hit_token_cap(11.76, 147) is True
    assert _hit_token_cap(17.60, 220) is True
    assert _hit_token_cap(10.0, 147) is False


def test_single_gpu_quality_issues_detect_token_cap_and_long_silence():
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


def test_single_gpu_next_retry_tokens_grows_with_hard_cap():
    assert _next_retry_tokens(147, 512) == 221
    assert _next_retry_tokens(400, 512) == 512


def test_single_gpu_requests_accept_optional_max_new_tokens():
    capswriter_req = CapsWriterSpeakRequest(text="你好。", max_new_tokens=96)
    openai_req = SpeechRequest(input="你好。", max_new_tokens=96)

    assert capswriter_req.max_new_tokens == 96
    assert openai_req.max_new_tokens == 96


def test_single_gpu_requests_accept_optional_trace_id():
    capswriter_req = CapsWriterSpeakRequest(text="你好。", trace_id="trace-a")
    openai_req = SpeechRequest(input="你好。", trace_id="trace-b")

    assert capswriter_req.trace_id == "trace-a"
    assert openai_req.trace_id == "trace-b"


def test_single_gpu_plan_splits_long_text_without_truncating():
    text = "第一句用于测试。第二句继续测试。第三句确保可以切分。"

    chunks, truncated = _split_tts_text_into_chunks(text, max_chars=20, min_chars=8, max_segments=10)

    assert chunks
    assert truncated is False
    assert all(len(chunk) <= 20 for chunk in chunks)


def test_single_gpu_status_payload_recommends_single_concurrency():
    rows = [
        {
            "worker_id": 0,
            "status": "ready",
            "gpu_ids": ["0"],
            "default_speaker": "serena",
            "speakers": ["serena", "aiden"],
        }
    ]

    payload = _status_payload(rows, "single_gpu")

    assert payload["success"] is True
    assert payload["tts_backend"] == "single_gpu"
    assert payload["workers_ready"] == 1
    assert payload["api_version"] == "tts-http-v1"
    assert payload["capabilities"]["supports_trace_id"] is True
    assert payload["client_defaults"]["recommended_speak_concurrency"] == 1
    assert payload["client_defaults"]["recommended_prefetch_chunks"] == 1
