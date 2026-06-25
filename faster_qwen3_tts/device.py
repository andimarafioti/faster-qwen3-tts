"""
Device detection and management for CUDA, MPS (Apple Silicon), and CPU.
"""
import logging

import torch

logger = logging.getLogger(__name__)


def get_optimal_device(requested: str = "auto") -> str:
    """Resolve the best available device.

    Args:
        requested: "auto", "cuda", "mps", or "cpu"

    Returns:
        Resolved device string.
    """
    if requested != "auto":
        return requested

    if torch.cuda.is_available():
        return "cuda"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def device_supports_cuda_graphs(device: str) -> bool:
    """Check if the device supports CUDA graph capture."""
    return device.startswith("cuda") and torch.cuda.is_available()


def sync_device(device) -> None:
    """Synchronize the given device (no-op for CPU).

    Args:
        device: str ("cuda", "mps", "cpu") or torch.device
    """
    device_str = str(device) if not isinstance(device, str) else device
    if device_str.startswith("cuda"):
        torch.cuda.synchronize()
    elif device_str == "mps" and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()
