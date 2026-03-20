import asyncio
import importlib.util
import json
from pathlib import Path

import pytest


_MODULE_PATH = Path(__file__).resolve().parents[1] / "examples" / "openai_server.py"
_SPEC = importlib.util.spec_from_file_location("openai_server_test_module", _MODULE_PATH)
openai_server = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(openai_server)


def test_inference_queue_is_fifo_and_bounded():
    async def scenario():
        queue = openai_server.InferenceQueue(max_pending=2)
        first = await queue.acquire()

        acquired = []

        async def second_request():
            lease = await queue.acquire()
            acquired.append("second")
            await lease.release()

        waiter = asyncio.create_task(second_request())
        await asyncio.sleep(0)

        assert queue.depth == 2
        assert queue.waiting == 1

        with pytest.raises(openai_server.QueueFullError):
            await queue.acquire()

        await first.release()
        await waiter

        assert acquired == ["second"]
        assert queue.depth == 0
        assert queue.waiting == 0

    asyncio.run(scenario())


def test_load_voices_file_resolves_audio_relative_to_config(tmp_path):
    voices_path = tmp_path / "voices.json"
    ref_audio = tmp_path / "voice.wav"
    ref_audio.write_bytes(b"RIFF")
    voices_path.write_text(
        json.dumps(
            {
                "alloy": {
                    "ref_audio": "voice.wav",
                    "ref_text": "reference text",
                    "language": "English",
                }
            }
        ),
        encoding="utf-8",
    )

    voices, default_voice = openai_server._load_voices_file(str(voices_path))

    assert default_voice == "alloy"
    assert voices["alloy"]["ref_audio"] == str(ref_audio.resolve())
    assert voices["alloy"]["chunk_size"] == openai_server.DEFAULT_CHUNK_SIZE


def test_resolve_path_falls_back_to_hf_model_dir(tmp_path):
    original_model_dir = openai_server.HF_MODEL_DIR
    model_dir = tmp_path / "repository"
    model_dir.mkdir()
    audio_path = model_dir / "hf_only_ref_audio.wav"
    audio_path.write_bytes(b"RIFF")
    openai_server.HF_MODEL_DIR = model_dir
    try:
        resolved = openai_server._resolve_path("hf_only_ref_audio.wav")
    finally:
        openai_server.HF_MODEL_DIR = original_model_dir

    assert resolved == str(audio_path.resolve())
