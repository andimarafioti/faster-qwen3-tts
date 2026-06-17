"""Tests for the git voice source in init_voices (codeload tarball extraction).

init_voices is mocked in conftest (sys.modules), so the real module is loaded
fresh from disk under a separate name to exercise the actual code."""

import importlib.util
import io
import tarfile
import urllib.request
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_init_voices():
    spec = importlib.util.spec_from_file_location(
        "init_voices_real", _REPO_ROOT / "init_voices.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


iv = _load_init_voices()


def _make_tarball(files: dict[str, bytes], root: str = "live-assets-abc123") -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for path, data in files.items():
            info = tarfile.TarInfo(name=f"{root}/{path}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


class _FakeResp:
    def __init__(self, data):
        self._data = data

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@pytest.fixture
def fake_codeload(monkeypatch):
    def _install(tar_bytes):
        monkeypatch.setattr(
            urllib.request, "urlopen", lambda *a, **k: _FakeResp(tar_bytes)
        )
    return _install


def test_qwen3_prefix_extracts_flat(fake_codeload, tmp_path):
    fake_codeload(_make_tarball({
        "voices/qwen3/english-male.wav": b"WAV",
        "voices/qwen3/english-male.txt": b"txt",
        "voices/qwen3/english-male.pt": b"PT",
        "voices/voxcpm/vx-ref/reference.wav": b"OTHER",
        "README.md": b"docs",
    }))
    n = iv._download_voices_from_git(
        "QarlAI/live-assets", "develop", "", "voices/qwen3", str(tmp_path)
    )
    assert n == 3
    assert (tmp_path / "english-male.wav").read_bytes() == b"WAV"
    # voxcpm + top-level noise are excluded by the qwen3 prefix.
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "english-male.pt", "english-male.txt", "english-male.wav",
    ]


def test_voices_prefix_preserves_subtrees(fake_codeload, tmp_path):
    fake_codeload(_make_tarball({
        "voices/english-male.wav": b"FLAT",
        "voices/qwen3/english-male.wav": b"Q",
        "voices/voxcpm/vx-ref/reference.wav": b"V",
    }))
    n = iv._download_voices_from_git(
        "QarlAI/live-assets", "develop", "", "voices", str(tmp_path)
    )
    assert n == 3
    assert (tmp_path / "english-male.wav").read_bytes() == b"FLAT"
    assert (tmp_path / "qwen3" / "english-male.wav").read_bytes() == b"Q"
    assert (tmp_path / "voxcpm" / "vx-ref" / "reference.wav").read_bytes() == b"V"


def test_path_traversal_member_skipped(fake_codeload, tmp_path):
    fake_codeload(_make_tarball({
        "voices/qwen3/english-male.wav": b"OK",
        "voices/qwen3/../../evil.txt": b"PWNED",
    }))
    n = iv._download_voices_from_git(
        "QarlAI/live-assets", "develop", "", "voices/qwen3", str(tmp_path)
    )
    assert n == 1
    assert (tmp_path / "english-male.wav").exists()
    assert not (tmp_path.parent / "evil.txt").exists()


def test_no_matching_prefix_returns_zero(fake_codeload, tmp_path):
    fake_codeload(_make_tarball({"backgrounds/office.png": b"IMG"}))
    n = iv._download_voices_from_git(
        "QarlAI/live-assets", "develop", "", "voices/qwen3", str(tmp_path)
    )
    assert n == 0
