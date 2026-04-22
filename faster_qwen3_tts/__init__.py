"""
faster-qwen3-tts: Real-time Qwen3-TTS inference using CUDA graphs
"""
from .continuation import apply_continuation_state_delta
from .model import FasterQwen3TTS

__version__ = "0.3.0"
__all__ = ["FasterQwen3TTS", "apply_continuation_state_delta"]
