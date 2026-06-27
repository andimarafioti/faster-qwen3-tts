import contextlib
import sys
import logging
import torch

logger = logging.getLogger(__name__)

class _FilteredStdout:
    def __init__(self, stream, suppress_substrings):
        self._stream = stream
        self._suppress = suppress_substrings

    def write(self, data):
        if any(s in data for s in self._suppress):
            return len(data)
        return self._stream.write(data)

    def flush(self):
        return self._stream.flush()


@contextlib.contextmanager
def suppress_flash_attn_warning():
    filtered = _FilteredStdout(
        sys.stdout,
        suppress_substrings=(
            "flash-attn is not installed",
            "manual PyTorch version",
            "Please install flash-attn",
        ),
    )
    with contextlib.redirect_stdout(filtered):
        yield


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

