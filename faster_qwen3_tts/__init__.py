"""
faster-qwen3-tts: Real-time Qwen3-TTS inference using CUDA graphs
"""
from .model import FasterQwen3TTS
from .ggml_backend import GGMLQwen3TTS
from .utils import get_optimal_device, device_supports_cuda_graphs

__version__ = "0.3.0"
__all__ = ["FasterQwen3TTS", "GGMLQwen3TTS", "get_optimal_device", "device_supports_cuda_graphs"]
