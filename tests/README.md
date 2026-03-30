# Tests

Automated test suite for the Faster Qwen3 TTS API. Uses **pytest** with a mocked FastAPI environment — no GPU or real model weights required for any test except `test_e2e_parity.py`.

## Running the tests

```bash
# All tests (excludes e2e parity)
.venv/bin/pytest tests/ -v

# A single file
.venv/bin/pytest tests/test_api.py -v

# Lint
.venv/bin/ruff check .
```

---

## Test files

### `test_api.py` 
Basic checks that the API surface is correct and required fields are validated.

| Test | What it checks |
|------|----------------|
| `test_health` | `GET /health` returns `status: ok` and `model_ready: true` |
| `test_health_not_ready` | Returns `model_ready: false` before model loads |
| `test_health_returns_voice_count` | Voice count reflects loaded voices |
| `test_list_voices_no_uid` | `GET /voices` works without a uid |
| `test_list_voices_unknown_uid` | Returns empty list for unknown uid |
| `test_clone_missing_fields` | `POST /voices/upload` returns 422 if fields are missing |
| `test_tts_missing_fields` | `POST /tts` returns 422 if fields are missing |
| `test_tts_stream_missing_fields` | `POST /tts/stream` returns 422 if fields are missing |
| `test_tts_ws_missing_fields` | `WS /tts/ws` returns error message if fields are missing |

---

### `test_tts.py` (TTS generation endpoints)
Tests for `/tts`, `/tts/stream`, and `/tts/ws`.

**`POST /tts` (non-streaming)**

| Test | What it checks |
|------|----------------|
| `test_tts_empty_text` | Returns 400 for empty text |
| `test_tts_missing_voice` | Returns 400 when voice field is absent |
| `test_tts_unknown_voice` | Returns 404 for an unrecognised voice name |
| `test_tts_valid_request` | Returns 200 with WAV audio bytes |
| `test_tts_voice_id_alias` | `voice_id` field works as an alias for `voice` |

**`POST /tts/stream` (HTTP streaming)**

| Test | What it checks |
|------|----------------|
| `test_tts_stream_no_voices` | Returns 503 when no voices are loaded |
| `test_tts_stream_unknown_voice` | Returns 404 for unknown voice |
| `test_tts_stream_valid_request` | Returns streaming audio, respects `X-Sample-Rate` header |

**`WS /tts/ws` (WebSocket streaming)**

| Test | What it checks |
|------|----------------|
| `test_tts_ws_invalid_json` | Sends error message on malformed JSON |
| `test_tts_ws_empty_text` | Sends error message for empty text |
| `test_tts_ws_unknown_voice` | Sends error message for unknown voice |
| `test_tts_ws_no_voices` | Sends error message when no voices loaded |
| `test_tts_ws_valid_request_end_message` | Valid request receives a final `end` message with `total_chunks` |

---

### `test_voice_get.py` (`GET /voices`)
Tests for listing voices, with and without GCS.

**Without GCS**

| Test | What it checks |
|------|----------------|
| `test_no_uid` | Returns system voices and empty user voices |
| `test_empty_uid` | Empty uid treated same as no uid |
| `test_uid_not_found` | Unknown uid returns empty user voices |
| `test_uid_with_voices` | Returns correct user voices with `loaded` and `has_embedding` fields |
| `test_system_voices_listed` | System voices appear in response |
| `test_has_embedding_false_when_vcp_none` | `has_embedding: false` when voice has no voice clone prompt |

**With GCS**

| Test | What it checks |
|------|----------------|
| `test_gcs_returns_voices` | Lists voices from GCS bucket when configured |
| `test_gcs_connection_failure` | Returns empty user voices gracefully on GCS error |

---

### `test_voice_upload_post.py` (`POST /voices/upload`)
Tests for uploading a reference voice (WAV + transcript).

**Validation**

| Test | What it checks |
|------|----------------|
| `test_missing_all_fields` | Returns 422 when all fields absent |
| `test_missing_wav_file` | Returns 422 when WAV is missing |
| `test_missing_txt_file` | Returns 422 when transcript is missing |
| `test_missing_voice_name` | Returns 422 when voice name is missing |
| `test_empty_voice_name` | Returns 400 for blank voice name |
| `test_audio_too_short` | Returns 400 when audio is under 3 seconds |
| `test_invalid_audio_format` | Returns 400 for non-WAV file |

**Success cases**

| Test | What it checks |
|------|----------------|
| `test_success` | Returns 200 with voice name on valid upload |
| `test_embedding_extraction_failure` | Returns 200 even if embedding extraction fails |
| `test_voice_name_slash_stripped` | Slashes stripped from voice names |
| `test_upload_with_uid` | Voice stored under `uid/name` when uid is provided |

**With GCS**

| Test | What it checks |
|------|----------------|
| `test_gcs_upload_success` | WAV and transcript are both uploaded to GCS |
| `test_gcs_upload_failure` | Returns 500 on GCS upload error |

---

### `test_sample_rate.py` (`FasterQwen3TTS` sample rate logic)
Unit tests for the model wrapper's `sample_rate` property and `speech_tokenizer` accessor. No API involved.

| Test | What it checks |
|------|----------------|
| `test_uses_speech_tokenizer_sample_rate_when_available` | Uses `model.speech_tokenizer.sample_rate` when present |
| `test_falls_back_to_base_model_sample_rate` | Falls back to `base_model.sample_rate` |
| `test_defaults_to_24khz_when_sample_rate_unavailable` | Defaults to 24000 Hz when neither is available |
| `test_exposes_speech_tokenizer_property` | `.speech_tokenizer` returns the correct object |
| `test_speech_tokenizer_property_raises_when_missing` | Raises `AttributeError` when tokenizer is absent |
| `test_wrapper_exposes_example_decode_path` | Tokenizer `.decode()` returns audio and sample rate |

---

### `test_sampling.py` (Generation sampling logic)
Unit tests for the token sampling helpers in `faster_qwen3_tts/sampling.py` and `generate.py`. No API involved.

| Test | What it checks |
|------|----------------|
| `test_repetition_penalty_uses_all_history` | Repetition penalty is applied correctly over full token history |
| `test_min_new_tokens_suppresses_early_eos` | *(CUDA only, skipped in CI)* EOS token is suppressed for the first `min_new_tokens` steps |

---

### `test_voice_clone_prompt_api.py` (`_prepare_generation()` method)
Unit tests for the voice clone prompt handling inside `FasterQwen3TTS`. No API or GPU involved.

| Test | What it checks |
|------|----------------|
| `test_public_api_exposes_voice_clone_prompt_parameter` | Both `generate_voice_clone` and `generate_voice_clone_streaming` have `voice_clone_prompt` in the right position |
| `test_prepare_generation_uses_precomputed_xvec_prompt_without_prompt_extraction` | Pre-computed prompt skips `create_voice_clone_prompt` |
| `test_prepare_generation_warns_for_instruct_with_xvec_only` | Logs warning when instruct + x-vector-only mode combined |
| `test_prepare_generation_rejects_missing_voice_clone_prompt_keys` | Raises `ValueError` for incomplete prompt dict |
| `test_prepare_generation_accepts_icl_prompt_with_ref_text` | ICL prompt works when `ref_text` is provided |
| `test_prepare_generation_accepts_upstream_prompt_items` | Accepts a list of prompt item objects |
| `test_prepare_generation_ignores_ref_text_with_precomputed_prompt` | x-vector-only mode ignores `ref_text` |
| `test_prepare_generation_icl_prompt_requires_ref_text` | ICL mode raises `ValueError` when `ref_text` is empty |
| `test_prepare_generation_requires_ref_audio_without_precomputed_prompt` | Raises `ValueError` when no `ref_audio` and no pre-computed prompt |

---

### `test_e2e_parity.py` — End-to-end GPU parity *(excluded from CI)*
Loads real model weights and checks that `FasterQwen3TTS` (CUDA graph mode) produces the same tokens as the base `Qwen3TTSModel`. Requires a GPU and downloaded model weights. Excluded from CI via `pyproject.toml`.

Run manually on a GPU machine:
```bash
.venv/bin/pytest tests/test_e2e_parity.py -v
```

---

## Infrastructure

### `conftest.py`
Shared fixtures and mock setup loaded automatically by pytest.

- Mocks `torchaudio` and `init_voices` before importing `app.py`
- Loads the real `faster_qwen3_tts` package (no GPU calls at import time)
- Mocks GCS (`google.cloud.storage`) with a `MagicMock`
- Sets `VOICES_DIR` to a temporary directory
- `client` fixture — replaces the FastAPI lifespan with a mock that sets `model`, `voices`, and `model_ready`
- `gcs_env` fixture — sets `VOICE_CACHE_BUCKET` and `VOICE_CACHE_PREFIX` env vars
- `wav_4s` / `wav_1s` fixtures — synthetic 440 Hz sine WAV bytes (4s and 1s)
- `txt_file` fixture — sample transcript bytes

### `helpers.py`
`make_wav_bytes(duration_seconds)` — generates a synthetic 440 Hz sine wave WAV file in memory, used to produce test audio without needing real recordings.
