import importlib.util
from pathlib import Path
from types import SimpleNamespace


_MODULE_PATH = Path(__file__).resolve().parents[1] / "examples" / "create_hf_endpoint.py"
_SPEC = importlib.util.spec_from_file_location("create_hf_endpoint_test_module", _MODULE_PATH)
create_hf_endpoint = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(create_hf_endpoint)


def test_build_env_includes_expected_server_settings():
    parser = create_hf_endpoint._parser()
    args = parser.parse_args(
        [
            "--name",
            "demo-endpoint",
            "--repository",
            "org/model-repo",
            "--image",
            "ghcr.io/org/faster-qwen3-tts:hf",
            "--voices",
            "voices.json",
            "--max-pending",
            "12",
            "--chunk-size",
            "16",
            "--env",
            "FOO=bar",
        ]
    )

    env = create_hf_endpoint._build_env(args)

    assert env["HF_MODEL_DIR"] == "/repository"
    assert env["QWEN_TTS_MODEL"] == "/repository"
    assert env["QWEN_TTS_MAX_PENDING"] == "12"
    assert env["QWEN_TTS_CHUNK_SIZE"] == "16"
    assert env["QWEN_TTS_VOICES"] == "voices.json"
    assert env["FOO"] == "bar"


def test_main_calls_supported_hf_api_kwargs(monkeypatch):
    captured = {}

    class DummyEndpoint:
        name = "demo-endpoint"
        status = "pending"
        url = "https://example.invalid"

        def wait(self, timeout):
            captured["wait_timeout"] = timeout
            return self

    class DummyApi:
        def create_inference_endpoint(self, name, **kwargs):
            captured["name"] = name
            captured["kwargs"] = kwargs
            return DummyEndpoint()

    monkeypatch.setattr(create_hf_endpoint, "HfApi", lambda: DummyApi())
    args = SimpleNamespace(
        name="demo-endpoint",
        repository="org/model-repo",
        image="ghcr.io/org/faster-qwen3-tts:hf",
        vendor="aws",
        region="us-east-1",
        accelerator="gpu",
        instance_size="x1",
        instance_type="nvidia-a10g",
        type="protected",
        namespace=None,
        revision=None,
        min_replica=1,
        max_replica=6,
        scale_to_zero_timeout=None,
        max_pending=8,
        chunk_size=12,
        device="cuda",
        voices="voices.json",
        ref_audio=None,
        ref_text="",
        language="Auto",
        env=[],
        wait=True,
        timeout=1800,
    )
    monkeypatch.setattr(create_hf_endpoint, "_parser", lambda: SimpleNamespace(parse_args=lambda: args))

    create_hf_endpoint.main()

    assert captured["name"] == "demo-endpoint"
    assert captured["wait_timeout"] == 1800
    assert captured["kwargs"]["repository"] == "org/model-repo"
    assert captured["kwargs"]["framework"] == "custom"
    assert captured["kwargs"]["min_replica"] == 1
    assert captured["kwargs"]["max_replica"] == 6
    assert captured["kwargs"]["env"]["QWEN_TTS_VOICES"] == "voices.json"
    assert captured["kwargs"]["custom_image"]["health_route"] == "/health"
    assert "scaling_metric" not in captured["kwargs"]
    assert "scaling_threshold" not in captured["kwargs"]
