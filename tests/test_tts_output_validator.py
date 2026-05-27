import math
import struct

import numpy as np

from examples import tts_output_validator as validator


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
