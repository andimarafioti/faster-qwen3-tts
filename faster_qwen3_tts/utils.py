import contextlib
import inspect
import sys


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


def patch_sdpa_enable_gqa():
    """Patch torch SDPA to ignore enable_gqa on builds that don't support it."""
    try:
        import torch
    except Exception:
        return
    sig = None
    try:
        sig = inspect.signature(torch.nn.functional.scaled_dot_product_attention)
    except (TypeError, ValueError):
        sig = None
    if sig is not None and "enable_gqa" in sig.parameters:
        return
    orig = torch.nn.functional.scaled_dot_product_attention

    def _wrapped(*args, **kwargs):
        kwargs.pop("enable_gqa", None)
        return orig(*args, **kwargs)

    torch.nn.functional.scaled_dot_product_attention = _wrapped
