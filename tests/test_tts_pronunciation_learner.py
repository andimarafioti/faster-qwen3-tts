import json

from examples import tts_pronunciation_learner as learner


def test_learner_accepts_high_confidence_default_term(monkeypatch, tmp_path):
    monkeypatch.setenv("QWEN_TTS_CONFIG_DIR", str(tmp_path))
    candidate_path = tmp_path / "pronunciation_candidates.jsonl"
    rows = [
        {
            "token": "ai",
            "present_in_asr": False,
            "issues": ["text_mismatch"],
            "expected_text": "设备 ai 服务异常。",
            "asr_text": "设备 A 服务异常。",
        }
    ]
    candidate_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")

    result = learner.learn_terms(threshold=0.85, dry_run=False)

    terms = json.loads((tmp_path / "pronunciation_terms.json").read_text(encoding="utf-8"))
    assert result["accepted"] == {"ai": "A I"}
    assert terms["ai"] == "A I"


def test_learner_does_not_auto_accept_common_words(monkeypatch, tmp_path):
    monkeypatch.setenv("QWEN_TTS_CONFIG_DIR", str(tmp_path))
    candidate_path = tmp_path / "pronunciation_candidates.jsonl"
    rows = [
        {
            "token": "open",
            "present_in_asr": False,
            "issues": ["text_mismatch"],
            "expected_text": "open 服务异常。",
            "asr_text": "服务异常。",
        }
    ]
    candidate_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")

    result = learner.learn_terms(threshold=0.85, dry_run=False)

    assert result["accepted"] == {}
    assert not (tmp_path / "pronunciation_terms.json").exists()
