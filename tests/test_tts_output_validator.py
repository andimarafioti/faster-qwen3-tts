import math
import struct
import json

import numpy as np
import pytest

from examples import tts_output_validator as validator


@pytest.fixture(autouse=True)
def isolate_validation_records(monkeypatch, tmp_path):
    monkeypatch.setenv("QWEN_TTS_VALIDATION_RECORDS", str(tmp_path / "tts_validation_records.jsonl"))


def _wav_bytes(samples: np.ndarray, sample_rate: int = 24000) -> bytes:
    samples = np.clip(samples.astype(np.float32), -1.0, 1.0)
    pcm = (samples * 32767.0).astype("<i2").tobytes()
    byte_rate = sample_rate * 2
    header = b"".join(
        [
            b"RIFF",
            struct.pack("<I", 36 + len(pcm)),
            b"WAVE",
            b"fmt ",
            struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, byte_rate, 2, 16),
            b"data",
            struct.pack("<I", len(pcm)),
        ]
    )
    return header + pcm


def test_validation_enabled_defaults_to_true(monkeypatch):
    monkeypatch.delenv("QWEN_TTS_VALIDATION_ENABLED", raising=False)

    assert validator.validation_enabled() is True


def test_validation_can_be_disabled(monkeypatch):
    monkeypatch.setenv("QWEN_TTS_VALIDATION_ENABLED", "0")

    assert validator.validation_enabled() is False
    assert validator.enqueue_validation(expected_text="你好", wav_bytes=_wav_bytes(np.zeros(2400))) is None


def test_audio_analysis_flags_silence():
    audio = validator.analyze_wav_bytes(_wav_bytes(np.zeros(24000)))
    issues = validator._audio_issues(audio)

    assert audio["duration_s"] == 1.0
    assert "empty_audio" in issues


def test_audio_analysis_accepts_tone():
    t = np.arange(24000, dtype=np.float32) / 24000.0
    tone = 0.1 * np.sin(2 * math.pi * 440 * t)

    audio = validator.analyze_wav_bytes(_wav_bytes(tone))
    issues = validator._audio_issues(audio)

    assert audio["duration_s"] == 1.0
    assert audio["rms"] > 0.01
    assert "empty_audio" not in issues


def test_audio_issues_detect_suspicious_short_duration():
    t = np.arange(24000 * 3, dtype=np.float32) / 24000.0
    tone = 0.1 * np.sin(2 * math.pi * 440 * t)

    audio = validator.analyze_wav_bytes(_wav_bytes(tone))
    issues = validator._audio_issues(audio, expected_audio_s=12.0)

    assert "suspicious_short_duration" in issues


def test_validate_wav_rejects_mid_length_early_stop_even_when_asr_matches(monkeypatch):
    def fake_asr(_wav_bytes: bytes, filename: str = "tts-validation.wav"):
        return {"success": True, "text": "风险是，测试短音频提前停止。"}

    monkeypatch.setattr(validator, "call_asr_transcribe", fake_asr)
    sample_rate = 24000
    t = np.arange(int(sample_rate * 1.55), dtype=np.float32) / sample_rate
    tone = 0.08 * np.sin(2 * math.pi * 220 * t)

    result = validator.validate_wav_bytes(
        expected_text="风险是，测试短音频提前停止。",
        wav_bytes=_wav_bytes(tone, sample_rate),
        metadata={"estimated_audio_s": 4.72},
    )

    assert result["passed"] is False
    assert result["verdict"] == "warning"
    assert "suspicious_short_duration" in result["issues"]


def test_text_similarity_ignores_punctuation_and_case():
    assert validator.text_similarity("Hello，世界！", "hello 世界") > 0.95


def test_validate_job_skips_when_asr_unavailable(monkeypatch):
    def raise_asr(_wav_bytes: bytes, filename: str = "tts-validation.wav"):
        raise RuntimeError("offline")

    monkeypatch.setattr(validator, "call_asr_transcribe", raise_asr)
    job = validator.ValidationJob(
        validation_id="val-test",
        expected_text="你好",
        wav_bytes=_wav_bytes(np.zeros(24000)),
    )

    result = validator._validate_job(job)

    assert result["verdict"] == "skipped"
    assert "asr_unavailable" in result["issues"]
    assert result["audio"]["duration_s"] == 1.0


def test_validation_records_lowercase_pronunciation_candidates(monkeypatch, tmp_path):
    monkeypatch.setenv("QWEN_TTS_CONFIG_DIR", str(tmp_path))

    job = validator.ValidationJob(
        validation_id="val-test",
        expected_text="设备 ai 需要调用 tts。",
        wav_bytes=_wav_bytes(np.ones(24000, dtype=np.float32) * 0.02),
    )
    record = {
        "issues": ["text_mismatch"],
        "asr_text": "设备 A 需要调用。",
        "similarity": 0.5,
    }

    validator._record_pronunciation_candidates(job, record)

    content = (tmp_path / "pronunciation_candidates.jsonl").read_text(encoding="utf-8")
    assert '"token": "ai"' in content
    assert '"token": "tts"' in content


def test_validate_wav_persists_record_to_jsonl(monkeypatch, tmp_path):
    records_path = tmp_path / "records" / "tts_validation_records.jsonl"
    monkeypatch.setenv("QWEN_TTS_VALIDATION_RECORDS", str(records_path))

    def fake_asr(_wav_bytes: bytes, filename: str = "tts-validation.wav"):
        return {"success": True, "text": "你好"}

    monkeypatch.setattr(validator, "call_asr_transcribe", fake_asr)

    result = validator.validate_wav_bytes(
        expected_text="你好",
        wav_bytes=_wav_bytes(np.ones(24000, dtype=np.float32) * 0.02),
        trace_id="persist-test",
        endpoint="/api/tts/speak",
    )

    rows = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["validation_id"] == result["validation_id"]
    assert rows[0]["trace_id"] == "persist-test"
    assert rows[0]["expected_text"] == "你好"
    assert rows[0]["endpoint"] == "/api/tts/speak"
    assert rows[0]["verdict"] == "passed"


def test_persisted_validation_results_returns_newest_first(monkeypatch, tmp_path):
    records_path = tmp_path / "tts_validation_records.jsonl"
    monkeypatch.setenv("QWEN_TTS_VALIDATION_RECORDS", str(records_path))

    validator._append_validation_record({"validation_id": "old", "created_at": 1})
    validator._append_validation_record({"validation_id": "new", "created_at": 2})

    rows = validator.persisted_validation_results(limit=2)

    assert [row["validation_id"] for row in rows] == ["new", "old"]
