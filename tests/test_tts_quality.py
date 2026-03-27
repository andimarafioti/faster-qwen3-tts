import os       
import subprocess
import tempfile
import requests
import pytest

TTS_URL = os.getenv("TTS_URL", "http://localhost:8000")
PARAKEET_BIN = os.getenv("PARAKEET_BIN", os.path.expanduser("~/work/parakeet-rs/target/release/examples/raw"))
PARAKEET_MODEL_DIR = os.getenv("PARAKEET_MODEL_DIR", os.path.expanduser("~/work/parakeet-rs"))

def tts_and_recognize(text, language="en", voice_id="english-male"):
    """Send text to TTS, convert audio, run ASR, return recognized text."""
    # 1. TTS
    r = requests.post(
        f"{TTS_URL}/tts",
        json={"text": text, "language": language, "voice_id": voice_id},
        timeout=120,
    )
    assert r.status_code == 200, f"TTS failed: {r.text}"

    with tempfile.TemporaryDirectory() as tmp:
        mp3_path = os.path.join(tmp, "tts.mp3")
        wav_path = os.path.join(tmp, "tts.wav")

        # 2. Save MP3
        with open(mp3_path, "wb") as f:
            f.write(r.content)

        # 3. Convert to 16kHz mono WAV
        subprocess.run(
            ["ffmpeg", "-y", "-i", mp3_path, "-ar", "16000", "-ac", "1", wav_path],
            check=True,
            capture_output=True,
        )

        # 4. Run parakeet-rs
        result = subprocess.run(
            [PARAKEET_BIN, wav_path, "tdt"],
            capture_output=True,
            text=True,
            cwd=PARAKEET_MODEL_DIR,
        )
        assert result.returncode == 0, f"Parakeet failed: {result.stderr}"

    return parse_parakeet_output(result.stdout)


def parse_parakeet_output(output):
    """Extract transcription text from parakeet-rs output."""
    lines = output.strip().split("\n")
    for i, line in enumerate(lines):
        if line.startswith("Loading"):
            if i + 1 < len(lines):
                return lines[i + 1].strip()
    return ""


def normalize(text):
    """Normalize text for comparison: lowercase, replace hyphens with spaces, remove punctuation."""
    text = text.lower().replace("-", " ")
    return " ".join("".join(c for c in text if c.isalnum() or c == " ").split())


def word_overlap(original, recognized):
    """Calculate percentage of original words found in recognized text."""
    orig_words = set(normalize(original).split())
    rec_words = set(normalize(recognized).split())
    if not orig_words:
        return 0.0
    return len(orig_words & rec_words) / len(orig_words)


class TestTTSQuality:
    @pytest.mark.parametrize("voice_id, language, text", [
        ("english-male", "en", "Hello, this is a test of the text to speech system."),
        ("english-female", "en", "Hello, this is a test of the text to speech system."),
        ("french-male", "fr", "Bonjour, ceci est un test du système de synthèse vocale."),
        ("french-female", "fr", "Bonjour, ceci est un test du système de synthèse vocale."),
    ])
    def test_voice(self, voice_id, language, text):
        recognized = tts_and_recognize(text, language=language, voice_id=voice_id)
        overlap = word_overlap(text, recognized)
        print(f"\nVoice:      {voice_id}")
        print(f"Original:   {text}")
        print(f"Recognized: {recognized.strip()}")
        print(f"Overlap:    {overlap:.0%}")
        assert overlap >= 0.8

    def test_english_numbers(self):
        text = "The meeting is at three thirty in room two hundred."
        recognized = tts_and_recognize(text)
        overlap = word_overlap(text, recognized)
        print(f"\nOriginal:   {text}")
        print(f"Recognized: {recognized.strip()}")
        print(f"Overlap:    {overlap:.0%}")
        assert overlap >= 0.8

    def test_english_long(self):
        text = "Artificial intelligence is transforming the way we interact with technology in our daily lives."
        recognized = tts_and_recognize(text)
        overlap = word_overlap(text, recognized)
        print(f"\nOriginal:   {text}")
        print(f"Recognized: {recognized.strip()}")
        print(f"Overlap:    {overlap:.0%}")
        assert overlap >= 0.8