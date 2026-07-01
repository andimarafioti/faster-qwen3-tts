from examples.ray_dual_worker_server import _split_tts_text_into_chunks
from examples.tts_text_normalizer import has_readable_text, normalize_for_tts
import json
import os
import pytest


@pytest.fixture(autouse=True)
def isolate_user_pronunciation_terms(monkeypatch, tmp_path):
    monkeypatch.setenv("QWEN_TTS_PRONUNCIATION_TERMS", str(tmp_path / "pronunciation_terms.json"))


def test_normalizer_removes_markdown_separator_without_dropping_tail(monkeypatch):
    monkeypatch.setenv("QWEN_TTS_NORMALIZER", "basic")

    result = normalize_for_tts(
        "好像发了一个 hello，也发了一个 n i h a o，好像报错。  --- 感觉还是有不能连贯输出的情况?"
    )

    assert "---" not in result.text
    assert "hello" in result.text
    assert "感觉还是有不能连贯输出的情况" in result.text
    assert result.changed is True


def test_normalizer_keeps_english_tail_after_please(monkeypatch):
    monkeypatch.setenv("QWEN_TTS_NORMALIZER", "basic")

    result = normalize_for_tts("Please check the summary and TTS function normally, please continue.")

    assert result.text.endswith("please continue.")
    assert has_readable_text(result.text)


def test_normalizer_drops_pure_symbol_text(monkeypatch):
    monkeypatch.setenv("QWEN_TTS_NORMALIZER", "basic")

    result = normalize_for_tts("--- *** ___")

    assert result.text == ""
    assert has_readable_text(result.text) is False


def test_cleaned_text_keeps_tail_when_chunked(monkeypatch):
    monkeypatch.setenv("QWEN_TTS_NORMALIZER", "basic")
    text = "好像发了一个 hello，也发了一个 n i h a o，好像报错。  --- 感觉还是有不能连贯输出的情况?"

    chunks, truncated = _split_tts_text_into_chunks(text, max_chars=90, min_chars=28, max_segments=10)

    assert truncated is False
    assert chunks
    assert "---" not in "".join(chunks)
    assert "感觉还是有不能连贯输出的情况" in "".join(chunks)


def test_multi_sentence_diagnostic_text_splits_below_max_chars(monkeypatch):
    monkeypatch.setenv("QWEN_TTS_NORMALIZER", "basic")
    text = "诊断确认: T T S 服务端生成坏音频仍返回二零零,需修复质量校验. 风险是播报失败被误判为成功,下一步是增强服务端失败拦截与重试逻辑."

    chunks, truncated = _split_tts_text_into_chunks(text, max_chars=90, min_chars=28, max_segments=10)

    assert truncated is False
    assert len(chunks) == 2
    assert "仍返回二零零" in chunks[0]
    assert "需修复质量校验" in chunks[0]
    assert "风险是播报失败" in chunks[1]
    assert "下一步是增强服务端" in chunks[1]


def test_plain_multi_sentence_summary_stays_one_chunk_below_max_chars(monkeypatch):
    monkeypatch.setenv("QWEN_TTS_NORMALIZER", "basic")
    text = "设备 ivan，目录 UGREEN。当前流程持续传音，耗流且资源浪费。建议本地加轻量VAD，有声才发，尾音保留后交云端处理，降本节能。"

    chunks, truncated = _split_tts_text_into_chunks(text, max_chars=90, min_chars=28, max_segments=10)

    assert truncated is False
    assert len(chunks) == 1
    assert "当前流程持续传音" in chunks[0]
    assert "建议本地加轻量 V A D" in chunks[0]


def test_full_user_report_keeps_chinese_content():
    text = (
        '我说错了，它不是音色反转，它它是发音的语气会有变化，比如说之前那一句。'
        '"[Image #1] 好像发了一个 hello，也发了一个 n i h a o，好像报错。  --- '
        '感觉还是有不能连贯输出的情况?"  --- '
        "这句话的前面部分，一直到好像报错，是一种，嗯，语气风格，它音色是相同的，"
        "但是后面感觉还是不能连贯输出的情况，它就变成另外一种语气风格。 ----- "
        "上面这整段好像还是处理不好。中间会有跳过很多文字，而且是中文的。"
    )

    result = normalize_for_tts(text)
    chunks, truncated = _split_tts_text_into_chunks(text, max_chars=90, min_chars=28, max_segments=20)

    assert truncated is False
    assert "[Image" not in result.text
    assert "这句话的前面部分" in result.text
    assert "它就变成另外一种语气风格" in result.text
    assert "上面这整段好像还是处理不好" in result.text
    assert "中间会有跳过很多文字" in result.text
    assert all(len(chunk) > 12 for chunk in chunks)


def test_pinyin_like_unknown_token_uses_chinese_syllable_fallback(monkeypatch):
    monkeypatch.setenv("QWEN_TTS_NORMALIZER", "basic")

    result = normalize_for_tts("siyuan resume 019e63c7")

    assert "斯 元" in result.text
    assert any(item["kind"] == "pinyin_fallback" and item["source"] == "siyuan" for item in result.normalization_trace)


def test_uppercase_ambiguous_token_defaults_to_letter_spelling(monkeypatch):
    monkeypatch.setenv("QWEN_TTS_NORMALIZER", "basic")

    result = normalize_for_tts("设备 MI，API 和 GPU 正常，读AI和AI助手。")

    assert "M I" in result.text
    assert "A P I" in result.text
    assert "G P U" in result.text
    assert "读 A I 和 A I 助手" in result.text
    assert "米" not in result.text


def test_technical_identifiers_are_verbalized_by_shape(monkeypatch):
    monkeypatch.setenv("QWEN_TTS_NORMALIZER", "basic")

    result = normalize_for_tts("README、WebUI、ESP32-screen、QWEN_TTS_PORT=8092、--max-new-tokens。")

    assert "R E A D M E" in result.text
    assert "Web U I" in result.text
    assert "E S P 三 二 screen" in result.text
    assert "环境变量 Q W E N T T S P O R T 等于 八 零 九 二" in result.text
    assert "参数 max new tokens" in result.text
    assert {item["kind"] for item in result.normalization_trace} >= {
        "acronym",
        "mixed_identifier",
        "hyphen",
        "env_assignment",
        "cli_flag",
    }


def test_lowercase_pronunciation_terms_are_verbalized_without_splitting_words(monkeypatch):
    monkeypatch.setenv("QWEN_TTS_NORMALIZER", "basic")

    result = normalize_for_tts("设备 ai 正常，video analyzer 和 open 保持英文。")

    assert "设备 A I 正常" in result.text
    assert "video analyzer" in result.text
    assert "open" in result.text


def test_mixed_tokens_split_uppercase_runs(monkeypatch):
    monkeypatch.setenv("QWEN_TTS_NORMALIZER", "basic")

    result = normalize_for_tts("OpenAI WebUI QwenTTS")

    assert "Open A I" in result.text
    assert "Web U I" in result.text
    assert "Qwen T T S" in result.text


def test_pronunciation_terms_reload_when_file_changes(monkeypatch, tmp_path):
    monkeypatch.setenv("QWEN_TTS_NORMALIZER", "basic")
    monkeypatch.setenv("QWEN_TTS_PRONUNCIATION_TERMS", str(tmp_path / "terms.json"))
    path = tmp_path / "terms.json"

    path.write_text(json.dumps({"qwen": "Q wen"}), encoding="utf-8")
    first = normalize_for_tts("qwen 服务正常。")
    assert "Q wen 服务正常" in first.text

    path.write_text(json.dumps({"qwen": "Q w e n", "faster-qwen3-tts": "faster Q wen three T T S"}), encoding="utf-8")
    os.utime(path, None)
    second = normalize_for_tts("qwen 和 faster-qwen3-tts 服务正常。")

    assert "Q w e n 和 faster Q wen three T T S 服务正常" in second.text


def test_code_symbols_paths_and_colon_none_are_verbalized(monkeypatch):
    monkeypatch.setenv("QWEN_TTS_NORMALIZER", "basic")

    result = normalize_for_tts("风险:无，a != b，/v1/audio/speech。")

    assert "风险是，无" in result.text
    assert "不等于" in result.text
    assert "路径 v 一 audio speech" in result.text
    assert "风险五" not in result.text


def test_discourse_labels_are_verbalized_for_speech(monkeypatch):
    monkeypatch.setenv("QWEN_TTS_NORMALIZER", "basic")

    result = normalize_for_tts("风险:若配置错误,可能引发并发混乱。下一步:启用 traceid 哈希分配。")

    assert result.text.startswith("风险是，若配置错误")
    assert "。下一步是，启用 traceid 哈希分配" in result.text


def test_discourse_label_verbalizer_does_not_touch_urls_or_assignments(monkeypatch):
    monkeypatch.setenv("QWEN_TTS_NORMALIZER", "basic")

    result = normalize_for_tts("地址 http://example.test/a:b，QWEN_TTS_PORT=8092，路径 /v1/audio:speech。")

    assert "地址是" not in result.text
    assert "风险是" not in result.text
    assert "下一步是" not in result.text
    assert "环境变量 Q W E N T T S P O R T 等于 八 零 九 二" in result.text
    assert "路径 v 一 audio" in result.text


def test_normalizer_can_disable_technical_verbalizer(monkeypatch):
    monkeypatch.setenv("QWEN_TTS_NORMALIZER", "basic")
    monkeypatch.setenv("QWEN_TTS_TECH_NORMALIZER", "0")

    result = normalize_for_tts("siyuan MI --max-new-tokens")

    assert result.text == "siyuan MI --max-new-tokens"
    assert result.normalization_trace == ()
