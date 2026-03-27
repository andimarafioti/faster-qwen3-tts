import io
from pathlib import Path

import numpy as np
import soundfile as sf


def make_wav_bytes(duration_seconds: float = 4.0, sample_rate: int = 24000, channels: int = 1) -> bytes:
    """Generate valid WAV file bytes containing a 440Hz sine wave."""
    num_frames = int(duration_seconds * sample_rate)
    t = np.linspace(0, duration_seconds, num_frames, dtype=np.float32)
    waveform = np.sin(2 * np.pi * 440 * t)
    if channels > 1:
        waveform = np.column_stack([waveform] * channels)
    buf = io.BytesIO()
    sf.write(buf, waveform, sample_rate, format="WAV")
    buf.seek(0)
    return buf.read()


def make_wav_file(path: Path, duration_seconds: float = 4.0, sample_rate: int = 24000) -> Path:
    """Write a valid WAV file to the given path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(make_wav_bytes(duration_seconds=duration_seconds, sample_rate=sample_rate))
    return path
