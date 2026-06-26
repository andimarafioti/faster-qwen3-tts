"""
faster-qwen3-tts: Real-time Qwen3-TTS inference using CUDA graphs
"""
from .model import FasterQwen3TTS
from .utils import get_optimal_device, device_supports_cuda_graphs, sync_device

__version__ = "0.2.6"
__all__ = ["FasterQwen3TTS", "get_optimal_device", "device_supports_cuda_graphs", "sync_device"]
