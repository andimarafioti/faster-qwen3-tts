"""Optional GGML/qwentts.cpp backend adapter.

This module keeps the public faster-qwen3-tts API shape while delegating
generation to the separately packaged qwentts.cpp C ABI wrapper.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import warnings
from pathlib import Path
from typing import Callable, Generator, Optional, Tuple, Union

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

_QWEN_FRAME_RATE = 24000 / 1920
_VOICE_REF_CACHE_VERSION = 1
_NON_PREFILL_TEXT_WARNING = (
    "The GGML backend does not expose Qwen3-TTS step-by-step text feeding "
    "(`non_streaming_mode=False`); qwentts.cpp ignores this option and uses "
    "its native prompt layout."
)


def _require_qwentts_cpp():
    try:
        from qwentts_cpp import QwenTTS, load_speaker_embedding
    except ImportError as exc:
        raise ImportError(
            "backend='ggml' requires the optional qwentts-cpp-python package. "
            "Install that package first, then retry with backend='ggml'."
        ) from exc
    return QwenTTS, load_speaker_embedding


def _resample_linear(audio: np.ndarray, src_sr: int, dst_sr: int = 24000) -> np.ndarray:
    if src_sr == dst_sr:
        return np.ascontiguousarray(audio, dtype=np.float32)
    if audio.size == 0:
        return np.zeros(0, dtype=np.float32)
    duration = audio.shape[0] / float(src_sr)
    dst_n = max(1, int(round(duration * dst_sr)))
    src_x = np.linspace(0.0, duration, num=audio.shape[0], endpoint=False)
    dst_x = np.linspace(0.0, duration, num=dst_n, endpoint=False)
    return np.interp(dst_x, src_x, audio).astype(np.float32)


def _load_ref_audio_24k(
    ref_audio: Union[str, Path],
    *,
    append_silence: bool = True,
    silence_secs: float = 0.5,
) -> np.ndarray:
    audio, sr = sf.read(str(ref_audio), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if append_silence and silence_secs > 0:
        audio = np.concatenate([audio, np.zeros(int(sr * silence_secs), dtype=np.float32)])
    return _resample_linear(np.asarray(audio, dtype=np.float32), int(sr), 24000)


def _default_voice_ref_cache_dir() -> Path:
    env_value = os.environ.get("FQWEN3TTS_QWENTTS_REF_CACHE_DIR")
    if env_value:
        return Path(env_value)
    return Path.home() / ".cache" / "faster-qwen3-tts" / "qwentts_refs"


def _path_identity(path: Union[str, Path]) -> str:
    p = Path(path)
    try:
        stat = p.stat()
        return f"{p.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
    except OSError:
        return str(p)


def _warn_non_prefill_text_mode(non_streaming_mode: Optional[bool]) -> None:
    if non_streaming_mode is False:
        warnings.warn(_NON_PREFILL_TEXT_WARNING, RuntimeWarning, stacklevel=3)


class GGMLQwen3TTS:
    """FasterQwen3TTS-compatible wrapper backed by qwentts.cpp/GGML."""

    sample_rate = 24000

    def __init__(
        self,
        runtime,
        *,
        model_identity: str = "unknown",
        voice_ref_cache_dir: Optional[Union[str, Path]] = None,
    ):
        self.runtime = runtime
        self.model_identity = str(model_identity)
        self.voice_ref_cache_dir = (
            Path(voice_ref_cache_dir) if voice_ref_cache_dir is not None else _default_voice_ref_cache_dir()
        )
        self._voice_ref_cache = {}
        self.last_adapter_profile: Optional[dict] = None

    def get_supported_speakers(self) -> list[str]:
        if not hasattr(self.runtime, "speaker_names"):
            return []
        return list(self.runtime.speaker_names())

    def warmup(self, prefill_len: int = 100) -> None:
        """Warm up the backend before generation.

        qwentts.cpp has no separate CUDA-graph capture step, so this is a
        deliberate no-op. ``prefill_len`` is accepted for API compatibility
        with the Torch backend.
        """
        return None

    @classmethod
    def from_gguf(
        cls,
        talker_path: Union[str, Path],
        codec_path: Union[str, Path],
        *,
        library_path: Optional[Union[str, Path]] = None,
        use_fa: bool = True,
        clamp_fp16: bool = False,
        voice_ref_cache_dir: Optional[Union[str, Path]] = None,
    ) -> "GGMLQwen3TTS":
        QwenTTS, _load_speaker_embedding = _require_qwentts_cpp()
        runtime = QwenTTS(
            talker_path=talker_path,
            codec_path=codec_path,
            library_path=library_path,
            use_fa=use_fa,
            clamp_fp16=clamp_fp16,
        )
        model_identity = f"gguf:{_path_identity(talker_path)}|{_path_identity(codec_path)}"
        return cls(runtime, model_identity=model_identity, voice_ref_cache_dir=voice_ref_cache_dir)

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        *,
        quant: str = "BF16",
        cache_dir: Optional[Union[str, Path]] = None,
        local_files_only: bool = False,
        library_path: Optional[Union[str, Path]] = None,
        use_fa: bool = True,
        clamp_fp16: bool = False,
        voice_ref_cache_dir: Optional[Union[str, Path]] = None,
    ) -> "GGMLQwen3TTS":
        QwenTTS, _load_speaker_embedding = _require_qwentts_cpp()
        runtime = QwenTTS.from_pretrained(
            model_name,
            quant=quant,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            library_path=library_path,
            use_fa=use_fa,
            clamp_fp16=clamp_fp16,
        )
        return cls(
            runtime,
            model_identity=f"hf:{model_name}:{quant}",
            voice_ref_cache_dir=voice_ref_cache_dir,
        )

    def generate_voice_clone(
        self,
        text: str,
        language: str,
        ref_audio: Optional[Union[str, Path]] = None,
        ref_text: str = "",
        max_new_tokens: int = 2048,
        min_new_tokens: int = 2,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 1.0,
        do_sample: bool = True,
        repetition_penalty: float = 1.05,
        xvec_only: bool = False,
        non_streaming_mode: Optional[bool] = None,
        append_silence: bool = True,
        instruct: Optional[str] = None,
        ref_spk: Optional[Union[str, Path]] = None,
        ref_rvq: Optional[Union[str, Path]] = None,
        ref_spk_emb: Optional[np.ndarray] = None,
        ref_codes: Optional[np.ndarray] = None,
        voice_clone_prompt=None,
    ) -> Tuple[list, int]:
        _warn_non_prefill_text_mode(non_streaming_mode)
        if voice_clone_prompt is not None:
            raise NotImplementedError(
                "The GGML backend cannot consume torch voice_clone_prompt objects; "
                "use ref_spk/ref_rvq cached qwentts.cpp latents instead."
            )
        if instruct:
            raise NotImplementedError("qwentts.cpp currently rejects instruct for base voice-clone models")
        ref_kwargs, _adapter_prepare_ms, adapter_profile = self._resolve_clone_reference(
            ref_audio=ref_audio,
            ref_text=ref_text,
            xvec_only=xvec_only,
            append_silence=append_silence,
            ref_spk=ref_spk,
            ref_rvq=ref_rvq,
            ref_spk_emb=ref_spk_emb,
            ref_codes=ref_codes,
        )
        self.last_adapter_profile = adapter_profile
        audio, sr = self.runtime.synthesize(
            text=text,
            lang=language,
            **ref_kwargs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        return [audio], sr

    def generate_voice_clone_streaming(
        self,
        text: str,
        language: str,
        ref_audio: Optional[Union[str, Path]] = None,
        ref_text: str = "",
        max_new_tokens: int = 2048,
        min_new_tokens: int = 2,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 1.0,
        do_sample: bool = True,
        repetition_penalty: float = 1.05,
        chunk_size: int = 12,
        should_stop: Optional[Callable[[], bool]] = None,
        xvec_only: bool = False,
        non_streaming_mode: Optional[bool] = None,
        append_silence: bool = True,
        parity_mode: bool = False,
        instruct: Optional[str] = None,
        ref_spk: Optional[Union[str, Path]] = None,
        ref_rvq: Optional[Union[str, Path]] = None,
        ref_spk_emb: Optional[np.ndarray] = None,
        ref_codes: Optional[np.ndarray] = None,
        voice_clone_prompt=None,
    ) -> Generator[Tuple[np.ndarray, int, dict], None, None]:
        _warn_non_prefill_text_mode(non_streaming_mode)
        if voice_clone_prompt is not None:
            raise NotImplementedError(
                "The GGML backend cannot consume torch voice_clone_prompt objects; "
                "use ref_spk/ref_rvq cached qwentts.cpp latents instead."
            )
        if instruct:
            raise NotImplementedError("qwentts.cpp currently rejects instruct for base voice-clone models")
        ref_kwargs, adapter_prepare_ms, adapter_profile = self._resolve_clone_reference(
            ref_audio=ref_audio,
            ref_text=ref_text,
            xvec_only=xvec_only,
            append_silence=append_silence,
            ref_spk=ref_spk,
            ref_rvq=ref_rvq,
            ref_spk_emb=ref_spk_emb,
            ref_codes=ref_codes,
        )
        yield from self._stream_runtime(
            should_stop=should_stop,
            text=text,
            lang=language,
            **ref_kwargs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            chunk_size=chunk_size,
            adapter_prepare_ms=adapter_prepare_ms,
            adapter_profile=adapter_profile,
        )

    def _resolve_clone_reference(
        self,
        *,
        ref_audio: Optional[Union[str, Path]],
        ref_text: str,
        xvec_only: bool,
        append_silence: bool,
        ref_spk: Optional[Union[str, Path]],
        ref_rvq: Optional[Union[str, Path]],
        ref_spk_emb: Optional[np.ndarray],
        ref_codes: Optional[np.ndarray],
    ) -> Tuple[dict, float, dict]:
        adapter_start = time.perf_counter()
        adapter_profile = {
            "mode": "clone",
            "voice_ref_cache": "explicit" if any(value is not None for value in (ref_spk, ref_rvq, ref_spk_emb, ref_codes)) else "none",
        }
        has_cached_ref = any(
            value is not None for value in (ref_spk, ref_rvq, ref_spk_emb, ref_codes)
        )
        if ref_audio is not None and has_cached_ref:
            raise ValueError(
                "ref_audio is mutually exclusive with cached qwentts.cpp references "
                "(ref_spk/ref_rvq/ref_spk_emb/ref_codes)."
            )
        if ref_spk is not None and ref_spk_emb is not None:
            raise ValueError("Use either ref_spk or ref_spk_emb, not both.")
        if ref_rvq is not None and ref_codes is not None:
            raise ValueError("Use either ref_rvq or ref_codes, not both.")
        if xvec_only and (ref_rvq is not None or ref_codes is not None):
            raise ValueError("ref_rvq/ref_codes require ICL mode; set xvec_only=False.")

        if ref_audio is not None:
            ref_audio_24k = _load_ref_audio_24k(ref_audio, append_silence=append_silence)
            voice_ref, cache_profile = self._get_or_extract_voice_ref(
                ref_audio_24k,
                append_silence=append_silence,
            )
            adapter_profile.update(cache_profile)
            ref_kwargs = self._voice_ref_to_kwargs(
                voice_ref,
                xvec_only=xvec_only,
                ref_text=ref_text,
            )
            return ref_kwargs, (time.perf_counter() - adapter_start) * 1000, adapter_profile

        if not has_cached_ref:
            raise ValueError("Voice cloning requires ref_audio or cached ref_spk/ref_spk_emb.")

        spk_emb = self._load_cached_speaker(ref_spk, ref_spk_emb)
        codes = self._load_cached_codes(ref_rvq, ref_codes)
        if codes is not None and not ref_text:
            raise ValueError("ref_text is required when using ref_rvq/ref_codes.")

        ref_kwargs = {
            "ref_spk_emb": spk_emb,
            "ref_text": ref_text if codes is not None else None,
        }
        if codes is not None:
            ref_kwargs["ref_codes"] = codes
        return ref_kwargs, (time.perf_counter() - adapter_start) * 1000, adapter_profile

    def _voice_ref_to_kwargs(self, voice_ref, *, xvec_only: bool, ref_text: str) -> dict:
        include_codes = (not xvec_only) and bool(ref_text)
        ref_kwargs = {
            "ref_spk_emb": np.ascontiguousarray(voice_ref.ref_spk_emb, dtype=np.float32).reshape(-1),
            "ref_text": ref_text if include_codes else None,
        }
        if include_codes:
            ref_kwargs["ref_codes"] = np.ascontiguousarray(voice_ref.ref_codes, dtype=np.int32)
        return ref_kwargs

    def _get_or_extract_voice_ref(self, ref_audio_24k: np.ndarray, *, append_silence: bool):
        audio = np.ascontiguousarray(ref_audio_24k, dtype=np.float32).reshape(-1)
        key, metadata = self._voice_ref_cache_key(audio, append_silence=append_silence)
        key_short = key[:12]
        cache_start = time.perf_counter()

        if key in self._voice_ref_cache:
            return self._voice_ref_cache[key], {
                "voice_ref_cache": "memory",
                "voice_ref_cache_key": key_short,
                "voice_ref_cache_ms": (time.perf_counter() - cache_start) * 1000,
            }

        cached = self._load_voice_ref_from_disk(key, metadata)
        if cached is not None:
            self._voice_ref_cache[key] = cached
            return cached, {
                "voice_ref_cache": "disk",
                "voice_ref_cache_key": key_short,
                "voice_ref_cache_ms": (time.perf_counter() - cache_start) * 1000,
            }

        extract = getattr(self.runtime, "extract_voice_ref", None)
        if extract is None:
            raise NotImplementedError(
                "Raw ref_audio caching requires qwentts-cpp-python >= 0.3.0 "
                "with qt_extract_voice_ref support."
            )

        extract_start = time.perf_counter()
        voice_ref = extract(audio)
        extract_ms = (time.perf_counter() - extract_start) * 1000
        self._voice_ref_cache[key] = voice_ref
        self._save_voice_ref_to_disk(key, metadata, voice_ref)
        profile = {
            "voice_ref_cache": "miss",
            "voice_ref_cache_key": key_short,
            "voice_ref_extract_ms": extract_ms,
        }
        native_profile = getattr(self.runtime, "last_extract_voice_ref_profile", None)
        if native_profile:
            profile["voice_ref_extract_profile"] = dict(native_profile)
        return voice_ref, profile

    def _voice_ref_cache_key(self, ref_audio_24k: np.ndarray, *, append_silence: bool) -> tuple[str, dict]:
        audio_hash = hashlib.sha256(ref_audio_24k.tobytes()).hexdigest()
        metadata = {
            "version": _VOICE_REF_CACHE_VERSION,
            "model_identity": self.model_identity,
            "native_version": self._runtime_version(),
            "sample_rate": 24000,
            "dtype": "float32",
            "n_samples": int(ref_audio_24k.shape[0]),
            "append_silence": bool(append_silence),
            "audio_sha256": audio_hash,
        }
        key_payload = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(key_payload).hexdigest(), metadata

    def _runtime_version(self) -> str:
        version = getattr(self.runtime, "version", None)
        if callable(version):
            try:
                return str(version())
            except Exception:
                pass
        library = getattr(self.runtime, "library", None)
        library_version = getattr(library, "version", None)
        if callable(library_version):
            try:
                return str(library_version())
            except Exception:
                pass
        return "unknown"

    def _voice_ref_paths(self, key: str) -> tuple[Path, Path, Path]:
        base = self.voice_ref_cache_dir / key
        return base.with_suffix(".spk"), base.with_suffix(".rvq"), base.with_suffix(".json")

    def _load_voice_ref_from_disk(self, key: str, metadata: dict):
        spk_path, rvq_path, meta_path = self._voice_ref_paths(key)
        if not (spk_path.is_file() and rvq_path.is_file() and meta_path.is_file()):
            return None
        try:
            cached_metadata = json.loads(meta_path.read_text())
            if cached_metadata != metadata:
                return None
            load_voice_ref = getattr(self.runtime, "load_voice_ref", None)
            if load_voice_ref is None:
                return None
            return load_voice_ref(spk_path, rvq_path)
        except Exception as exc:
            logger.warning("Failed to load cached qwentts voice reference %s: %s", key[:12], exc)
            return None

    def _save_voice_ref_to_disk(self, key: str, metadata: dict, voice_ref) -> None:
        save = getattr(voice_ref, "save", None)
        if save is None:
            return
        try:
            self.voice_ref_cache_dir.mkdir(parents=True, exist_ok=True)
            spk_path, rvq_path, meta_path = self._voice_ref_paths(key)
            tmp_prefix = self.voice_ref_cache_dir / f".{key}.{os.getpid()}"
            tmp_spk = tmp_prefix.with_suffix(".spk")
            tmp_rvq = tmp_prefix.with_suffix(".rvq")
            tmp_meta = tmp_prefix.with_suffix(".json")
            save(tmp_spk, tmp_rvq)
            tmp_meta.write_text(json.dumps(metadata, sort_keys=True))
            tmp_spk.replace(spk_path)
            tmp_rvq.replace(rvq_path)
            tmp_meta.replace(meta_path)
        except Exception as exc:
            logger.warning("Failed to save qwentts voice reference cache %s: %s", key[:12], exc)

    def _load_cached_speaker(
        self,
        ref_spk: Optional[Union[str, Path]],
        ref_spk_emb: Optional[np.ndarray],
    ) -> np.ndarray:
        if ref_spk_emb is not None:
            spk_emb = np.ascontiguousarray(ref_spk_emb, dtype=np.float32).reshape(-1)
        elif ref_spk is not None:
            _QwenTTS, load_speaker_embedding = _require_qwentts_cpp()
            spk_emb = load_speaker_embedding(ref_spk)
        else:
            raise ValueError("ref_spk/ref_spk_emb is required for cached voice cloning.")

        if spk_emb.size == 0:
            raise ValueError("ref_spk_emb must not be empty.")
        return spk_emb

    def _load_cached_codes(
        self,
        ref_rvq: Optional[Union[str, Path]],
        ref_codes: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        if ref_codes is not None:
            return np.ascontiguousarray(ref_codes, dtype=np.int32)
        if ref_rvq is None:
            return None
        load_rvq_codes: Optional[Callable[..., np.ndarray]] = getattr(
            self.runtime,
            "load_rvq_codes",
            None,
        )
        if load_rvq_codes is None:
            raise NotImplementedError(
                "Cached RVQ references require qwentts-cpp-python >= 0.3.0 "
                "and qwentts.cpp ABI v2."
            )
        return load_rvq_codes(ref_rvq)

    def generate_custom_voice(
        self,
        text: str,
        speaker: str,
        language: str,
        instruct: Optional[str] = None,
        non_streaming_mode: Optional[bool] = None,
        max_new_tokens: int = 2048,
        min_new_tokens: int = 2,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 1.0,
        do_sample: bool = True,
        repetition_penalty: float = 1.05,
    ) -> Tuple[list, int]:
        _warn_non_prefill_text_mode(non_streaming_mode)
        audio, sr = self.runtime.synthesize(
            text=text,
            lang=language,
            speaker=speaker,
            instruct=instruct or None,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        return [audio], sr

    def generate_custom_voice_streaming(
        self,
        text: str,
        speaker: str,
        language: str,
        instruct: Optional[str] = None,
        non_streaming_mode: Optional[bool] = None,
        max_new_tokens: int = 2048,
        min_new_tokens: int = 2,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 1.0,
        do_sample: bool = True,
        repetition_penalty: float = 1.05,
        chunk_size: int = 12,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> Generator[Tuple[np.ndarray, int, dict], None, None]:
        _warn_non_prefill_text_mode(non_streaming_mode)
        adapter_start = time.perf_counter()
        yield from self._stream_runtime(
            should_stop=should_stop,
            text=text,
            lang=language,
            speaker=speaker,
            instruct=instruct or None,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            chunk_size=chunk_size,
            adapter_prepare_ms=(time.perf_counter() - adapter_start) * 1000,
        )

    def generate_voice_design(
        self,
        text: str,
        instruct: str,
        language: str,
        non_streaming_mode: Optional[bool] = None,
        max_new_tokens: int = 2048,
        min_new_tokens: int = 2,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 1.0,
        do_sample: bool = True,
        repetition_penalty: float = 1.05,
    ) -> Tuple[list, int]:
        _warn_non_prefill_text_mode(non_streaming_mode)
        audio, sr = self.runtime.synthesize(
            text=text,
            lang=language,
            instruct=instruct,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        return [audio], sr

    def generate_voice_design_streaming(
        self,
        text: str,
        instruct: str,
        language: str,
        non_streaming_mode: Optional[bool] = None,
        max_new_tokens: int = 2048,
        min_new_tokens: int = 2,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 1.0,
        do_sample: bool = True,
        repetition_penalty: float = 1.05,
        chunk_size: int = 12,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> Generator[Tuple[np.ndarray, int, dict], None, None]:
        _warn_non_prefill_text_mode(non_streaming_mode)
        adapter_start = time.perf_counter()
        yield from self._stream_runtime(
            should_stop=should_stop,
            text=text,
            lang=language,
            instruct=instruct,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            chunk_size=chunk_size,
            adapter_prepare_ms=(time.perf_counter() - adapter_start) * 1000,
        )

    def _stream_runtime(
        self,
        *,
        chunk_size: int,
        adapter_prepare_ms: float = 0.0,
        adapter_profile: Optional[dict] = None,
        should_stop: Optional[Callable[[], bool]] = None,
        **kwargs,
    ):
        chunk_sec = max(1, int(chunk_size)) / _QWEN_FRAME_RATE
        start = time.perf_counter()
        last = start
        for idx, (chunk, sr) in enumerate(
            self.runtime.stream(codec_chunk_sec=chunk_sec, **kwargs)
        ):
            if should_stop is not None and should_stop():
                break  # caller gone — stop consuming the runtime stream (chunk granularity)
            now = time.perf_counter()
            native_profile = getattr(self.runtime, "last_stream_profile", None)
            yield chunk, sr, {
                "chunk_index": idx,
                "decode_ms": (now - last) * 1000,
                "total_ms": (now - start) * 1000,
                "adapter_prepare_ms": adapter_prepare_ms if idx == 0 else 0.0,
                "adapter_profile": dict(adapter_profile) if (idx == 0 and adapter_profile) else None,
                "ggml_profile": dict(native_profile) if native_profile else None,
                "is_final": False,
            }
            last = now
