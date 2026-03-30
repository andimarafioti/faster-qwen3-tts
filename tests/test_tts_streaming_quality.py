import asyncio
import json
import os
import subprocess
import tempfile
import wave

import numpy as np
import pytest
import websockets

WS_URL = os.getenv("TTS_URL", "ws://localhost:8000").replace("http://", "ws://").replace("https://", "wss://")
PARAKEET_BIN = os.getenv("PARAKEET_BIN", os.path.expanduser("~/work/parakeet-rs/target/release/examples/raw"))
PARAKEET_MODEL_DIR = os.getenv("PARAKEET_MODEL_DIR", os.path.expanduser("~/work/parakeet-rs"))


async def _ws_tts(text, language, voice):
    """Connect to /tts/ws, collect all audio chunks, return (samples, sample_rate)."""
    uri = f"{WS_URL}/tts/ws"
    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps({"text": text, "language": language, "voice": voice}))
        samples = []
        sample_rate = 24000
        while True:
            msg = await ws.recv()
            data = json.loads(msg)
            if data["type"] == "audio":
                samples.extend(data["data"])
                sample_rate = data["sample_rate"]
            elif data["type"] == "end":
                break
            elif data["type"] == "error":
                raise RuntimeError(f"TTS error: {data['message']}")
    return np.array(samples, dtype=np.float32), sample_rate


def tts_and_recognize(text, language="English", voice="english-male"):
    """Stream audio via WebSocket, resample, transcribe with parakeet-rs."""
    audio, sample_rate = asyncio.run(_ws_tts(text, language, voice))

    with tempfile.TemporaryDirectory() as tmp:
        wav_path = os.path.join(tmp, "tts.wav")

        # Resample to 16kHz mono for parakeet (TTS outputs 24kHz)
        if sample_rate != 16000:
            import torch
            import torchaudio
            tensor = torch.from_numpy(audio).unsqueeze(0)
            tensor = torchaudio.functional.resample(tensor, sample_rate, 16000)
            audio = tensor.squeeze(0).numpy()

        # Save as 16kHz mono 16-bit WAV
        audio_int16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)
        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(audio_int16.tobytes())

        # Run parakeet-rs
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


class TestTTSStreamingQuality:
    @pytest.mark.parametrize("voice, language, text", [
        ("english-male", "English", "Hello, this is a test of the text to speech system."),
        ("english-female", "English", "Hello, this is a test of the text to speech system."),
        ("french-male", "French", "Bonjour, ceci est un test du système de synthèse vocale."),
        ("french-female", "French", "Bonjour, ceci est un test du système de synthèse vocale."),
    ])
    def test_voice(self, voice, language, text):
        recognized = tts_and_recognize(text, language=language, voice=voice)
        overlap = word_overlap(text, recognized)
        print(f"\nVoice:      {voice}")
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
