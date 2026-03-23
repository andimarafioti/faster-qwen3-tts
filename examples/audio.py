"""Audio helpers used by the local streaming examples."""

from __future__ import annotations

import queue
import threading
from typing import Optional

import numpy as np


class StreamPlayer:
    """Play streaming audio chunks through one persistent output stream."""

    def __init__(self, *, channels: int = 1, dtype: str = "float32", max_queue_chunks: int = 0):
        self.channels = channels
        self.dtype = dtype
        self.max_queue_chunks = max_queue_chunks

        self._queue: queue.Queue[Optional[np.ndarray]] = queue.Queue(maxsize=max_queue_chunks)
        self._pending = np.zeros((0, channels), dtype=np.float32)
        self._stream = None
        self._sample_rate: Optional[int] = None
        self._closed = False
        self._drained = threading.Event()

    def _load_sounddevice(self):
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise ImportError(
                "examples.audio.StreamPlayer requires the optional 'sounddevice' package. "
                "Install it with: pip install sounddevice"
            ) from exc
        return sd

    def _reshape_chunk(self, audio_chunk: np.ndarray) -> np.ndarray:
        arr = np.asarray(audio_chunk, dtype=np.float32)
        if arr.ndim == 1:
            if self.channels != 1:
                raise ValueError(f"Expected {self.channels} channels, got mono audio")
            return arr.reshape(-1, 1)
        if arr.ndim == 2:
            if arr.shape[1] != self.channels:
                raise ValueError(f"Expected {self.channels} channels, got {arr.shape[1]}")
            return arr
        raise ValueError(f"Expected 1D or 2D audio chunk, got shape {arr.shape}")

    def _callback(self, outdata, frames, _time, status):
        if status:
            pass

        written = 0
        while written < frames:
            if self._pending.shape[0] == 0:
                try:
                    next_chunk = self._queue.get_nowait()
                except queue.Empty:
                    outdata[written:] = 0
                    return

                if next_chunk is None:
                    outdata[written:] = 0
                    self._drained.set()
                    sd = self._load_sounddevice()
                    raise sd.CallbackStop()

                self._pending = next_chunk

            take = min(frames - written, self._pending.shape[0])
            outdata[written:written + take] = self._pending[:take]
            self._pending = self._pending[take:]
            written += take

    def _ensure_stream(self, sample_rate: int):
        if self._stream is not None:
            if sample_rate != self._sample_rate:
                raise ValueError(
                    f"StreamPlayer sample rate changed from {self._sample_rate} to {sample_rate}"
                )
            return

        sd = self._load_sounddevice()
        self._sample_rate = sample_rate
        self._stream = sd.OutputStream(
            samplerate=sample_rate,
            channels=self.channels,
            dtype=self.dtype,
            callback=self._callback,
        )
        self._stream.start()

    def __call__(self, audio_chunk: np.ndarray, sample_rate: int):
        if self._closed:
            raise RuntimeError("StreamPlayer is already closed")
        self._ensure_stream(sample_rate)
        self._queue.put(self._reshape_chunk(audio_chunk))

    def close(self, *, wait: bool = True, timeout: Optional[float] = None):
        if self._closed:
            return
        self._closed = True

        if self._stream is None:
            return

        self._queue.put(None)
        if wait:
            self._drained.wait(timeout=timeout)

        self._stream.close()
        self._stream = None
