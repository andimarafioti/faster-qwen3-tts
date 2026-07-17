"""
faster-qwen3-tts: Real-time Qwen3-TTS inference using CUDA graphs
"""
from .model import FasterQwen3TTS
from .ggml_backend import GGMLQwen3TTS

__version__ = "0.3.2"
__all__ = ["FasterQwen3TTS", "GGMLQwen3TTS"]
