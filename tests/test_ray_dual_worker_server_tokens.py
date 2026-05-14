from examples.ray_dual_worker_server import (
    CapsWriterSpeakRequest,
    SpeechRequest,
    _estimate_max_new_tokens,
    _resolve_max_new_tokens,
)


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


def test_speech_requests_accept_optional_max_new_tokens():
    capswriter_req = CapsWriterSpeakRequest(text="你好。", max_new_tokens=96)
    openai_req = SpeechRequest(input="你好。", max_new_tokens=96)

    assert capswriter_req.max_new_tokens == 96
    assert openai_req.max_new_tokens == 96
