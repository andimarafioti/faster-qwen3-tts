import json

from examples import tts_pronunciation_advisor as advisor


class _FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps(self.payload, ensure_ascii=False).encode("utf-8")


def _job():
    return advisor.AdviceJob(
        validation_id="val-test",
        expected_text="设备 qwen 服务异常。",
        asr_text="设备会服务异常。",
        issues=("text_mismatch",),
        similarity=0.4,
    )


def test_extract_candidate_terms_handles_technical_shapes():
    terms = advisor.extract_candidate_terms("faster-qwen3-tts 调用 WebUI, redyuan43 和 youtube watch。")

    assert "faster-qwen3-tts" in terms
    assert "qwen3" in terms
    assert "WebUI" in terms
    assert "redyuan43" in terms
    assert "youtube" in terms
    assert "watch" not in terms


def test_parse_advice_accepts_json_object():
    item = advisor._parse_advice(
        "qwen",
        '{"term":"qwen","pronunciation":"Q wen","confidence":0.98,"reason":"verified","action":"accept"}',
    )

    assert item is not None
    assert item.term == "qwen"
    assert item.pronunciation == "Q wen"
    assert item.action == "accept"


def test_parse_advice_normalizes_spelled_digits():
    item = advisor._parse_advice(
        "redyuan43",
        '{"term":"redyuan43","pronunciation":"R e d Y u a n 4 3","confidence":0.98,"reason":"ok","action":"accept"}',
    )

    assert item is not None
    assert item.pronunciation == "R e d Y u a n four three"


def test_request_advice_calls_agx_ollama(monkeypatch):
    def fake_urlopen(req, timeout):
        assert req.full_url == "http://agx.taild500c8.ts.net:11434/api/generate"
        body = json.loads(req.data.decode("utf-8"))
        assert body["model"] == "caps-voice-edit-qwen3-4b:latest"
        assert "qwen" in body["prompt"]
        return _FakeResponse(
            {
                "response": '{"term":"qwen","pronunciation":"Q wen","confidence":0.98,"reason":"ok","action":"accept"}'
            }
        )

    monkeypatch.setattr(advisor.request, "urlopen", fake_urlopen)

    item = advisor.request_advice("qwen", _job())

    assert item is not None
    assert item.pronunciation == "Q wen"
    assert item.confidence == 0.98


def test_apply_advice_writes_terms_and_audit(monkeypatch, tmp_path):
    monkeypatch.setenv("QWEN_TTS_CONFIG_DIR", str(tmp_path))
    item = advisor.PronunciationAdvice(
        term="qwen",
        pronunciation="Q wen",
        confidence=0.98,
        reason="verified",
        action="accept",
    )

    advisor.apply_advice(item, _job())

    terms = json.loads((tmp_path / "pronunciation_terms.json").read_text(encoding="utf-8"))
    audit = (tmp_path / "pronunciation_advice.audit.jsonl").read_text(encoding="utf-8")
    assert terms["qwen"] == "Q wen"
    assert '"validation_id": "val-test"' in audit


def test_apply_low_confidence_advice_stays_pending(monkeypatch, tmp_path):
    monkeypatch.setenv("QWEN_TTS_CONFIG_DIR", str(tmp_path))
    item = advisor.PronunciationAdvice(
        term="youtube",
        pronunciation="YouTube",
        confidence=0.5,
        reason="uncertain",
        action="pending",
    )

    advisor.apply_advice(item, _job())

    pending = json.loads((tmp_path / "pronunciation_terms.pending.json").read_text(encoding="utf-8"))
    assert pending["youtube"]["replacement"] == "YouTube"
    assert not (tmp_path / "pronunciation_terms.json").exists()


def test_should_request_advice_for_low_similarity_technical_record():
    record = {
        "validation_id": "val-test",
        "expected_text": "设备 qwen 服务异常。",
        "asr_text": "设备会服务异常。",
        "issues": [],
        "similarity": 0.8,
    }

    assert advisor.should_request_advice(record) is True


def test_cli_prints_advice_without_applying_by_default(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("QWEN_TTS_CONFIG_DIR", str(tmp_path))

    def fake_request_advice(term, job):
        assert term == "qwen"
        assert job.expected_text == "设备 qwen 服务异常。"
        return advisor.PronunciationAdvice(
            term="qwen",
            pronunciation="Q wen",
            confidence=0.98,
            reason="verified",
            action="accept",
        )

    monkeypatch.setattr(advisor, "request_advice", fake_request_advice)

    rc = advisor.main(["qwen", "--expected-text", "设备 qwen 服务异常。"])

    assert rc == 0
    assert '"pronunciation": "Q wen"' in capsys.readouterr().out
    assert not (tmp_path / "pronunciation_terms.json").exists()
