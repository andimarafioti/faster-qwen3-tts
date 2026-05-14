from examples.single_gpu_custom_voice_server import (
    CapsWriterSpeakRequest,
    SpeechRequest,
    _estimate_max_new_tokens,
    _resolve_max_new_tokens,
    _split_tts_text_into_chunks,
)


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


def test_single_gpu_requests_accept_optional_max_new_tokens():
    capswriter_req = CapsWriterSpeakRequest(text="你好。", max_new_tokens=96)
    openai_req = SpeechRequest(input="你好。", max_new_tokens=96)

    assert capswriter_req.max_new_tokens == 96
    assert openai_req.max_new_tokens == 96


def test_single_gpu_plan_splits_long_text_without_truncating():
    text = "第一句用于测试。第二句继续测试。第三句确保可以切分。"

    chunks, truncated = _split_tts_text_into_chunks(text, max_chars=20, min_chars=8, max_segments=10)

    assert chunks
    assert truncated is False
    assert all(len(chunk) <= 20 for chunk in chunks)
