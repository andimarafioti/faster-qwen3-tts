"""Asynchronous pronunciation advice using the AGX local Ollama model."""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, request

DEFAULT_BASE_URL = "http://agx.taild500c8.ts.net:11434"
DEFAULT_MODEL = "caps-voice-edit-qwen3-4b:latest"
DEFAULT_AUTO_ACCEPT_CONFIDENCE = 0.92

COMMON_WORDS = {
    "and",
    "are",
    "com",
    "for",
    "login",
    "open",
    "path",
    "the",
    "video",
    "watch",
    "with",
    "www",
}
TERM_RE = re.compile(
    r"(?<![A-Za-z0-9_])([A-Za-z][A-Za-z0-9]*(?:[-._/][A-Za-z0-9]+)*)(?![A-Za-z0-9_])"
)


@dataclass(frozen=True)
class PronunciationAdvice:
    term: str
    pronunciation: str
    confidence: float
    reason: str = ""
    action: str = "pending"
    source: str = "agx"


@dataclass(frozen=True)
class AdviceJob:
    validation_id: str
    expected_text: str
    asr_text: str
    issues: tuple[str, ...]
    similarity: float | None
    trace_id: str = ""
    endpoint: str = ""


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def advisor_enabled() -> bool:
    return _env_bool("QWEN_TTS_PRON_ADVISOR_ENABLED", False)


def _base_url() -> str:
    return os.getenv("QWEN_TTS_PRON_ADVISOR_BASE_URL", DEFAULT_BASE_URL).strip().rstrip("/")


def _model() -> str:
    return os.getenv("QWEN_TTS_PRON_ADVISOR_MODEL", DEFAULT_MODEL).strip()


def _timeout_s() -> float:
    return max(0.5, min(30.0, float(os.getenv("QWEN_TTS_PRON_ADVISOR_TIMEOUT_S", "8"))))


def _queue_size() -> int:
    return max(1, int(os.getenv("QWEN_TTS_PRON_ADVISOR_QUEUE_SIZE", "100")))


def _auto_accept_confidence() -> float:
    return max(0.0, min(1.0, float(os.getenv("QWEN_TTS_PRON_ADVISOR_AUTO_ACCEPT_CONFIDENCE", str(DEFAULT_AUTO_ACCEPT_CONFIDENCE)))))


def _config_dir() -> Path:
    return Path(os.getenv("QWEN_TTS_CONFIG_DIR", "~/.config/faster-qwen3-tts")).expanduser()


def _terms_path() -> Path:
    return Path(os.getenv("QWEN_TTS_PRONUNCIATION_TERMS", str(_config_dir() / "pronunciation_terms.json"))).expanduser()


def _pending_path() -> Path:
    return Path(os.getenv("QWEN_TTS_PRONUNCIATION_PENDING", str(_config_dir() / "pronunciation_terms.pending.json"))).expanduser()


def _audit_path() -> Path:
    return Path(os.getenv("QWEN_TTS_PRON_ADVISOR_AUDIT", str(_config_dir() / "pronunciation_advice.audit.jsonl"))).expanduser()


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json_object(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _split_compound(term: str) -> list[str]:
    parts = re.split(r"[-._/]+", term)
    return [part for part in parts if part]


def _is_candidate(term: str) -> bool:
    if not term or term.lower() in COMMON_WORDS:
        return False
    if len(term) < 2 or len(term) > 48:
        return False
    if any(char.isdigit() for char in term):
        return True
    if any(sep in term for sep in "-._/"):
        return True
    if any(char.isupper() for char in term[1:]):
        return True
    if term.lower() in {"agx", "api", "asr", "github", "gpu", "npm", "npmjs", "ocr", "qwen", "tts", "webui", "youtube"}:
        return True
    return False


def extract_candidate_terms(text: str) -> list[str]:
    seen: set[str] = set()
    terms: list[str] = []
    for match in TERM_RE.finditer(text or ""):
        term = match.group(1).strip()
        candidates = [term]
        if any(sep in term for sep in "-._/"):
            candidates.extend(_split_compound(term))
        for candidate in candidates:
            key = candidate.lower()
            if key not in seen and _is_candidate(candidate):
                seen.add(key)
                terms.append(candidate)
    return terms


def should_request_advice(record: dict[str, Any]) -> bool:
    issues = set(record.get("issues") or [])
    similarity = record.get("similarity")
    if issues & {"text_mismatch", "possible_truncation"}:
        return True
    try:
        if similarity is not None and float(similarity) < 0.99 and extract_candidate_terms(str(record.get("expected_text") or "")):
            return True
    except (TypeError, ValueError):
        return False
    return False


def _prompt_for(term: str, job: AdviceJob) -> str:
    return (
        "你是中文TTS发音词典助手。为给定技术词选择适合写进中文TTS输入文本的 pronunciation。"
        "只输出JSON对象，不要解释。字段必须包含 term, pronunciation, confidence, reason, action。"
        "action 只能是 accept 或 pending。规则："
        "1) 优先让语音可听懂、ASR可还原；2) 缩写用大写字母加空格，如 A G X；"
        "3) 用户名/包名拆开英文词和数字；4) 不要简单原样返回失败写法；"
        "5) 若不确定，action=pending；6) 命中已验证示例时 confidence 至少 0.95。"
        "已验证示例：qwen -> Q wen；AGX -> A G X；"
        "OCR -> O C R；WebUI -> Web U I；npmjs -> N P M J S。"
        f"待判断 term={term!r}。"
        f"期望文本片段={job.expected_text[:180]!r}。ASR文本片段={job.asr_text[:180]!r}。"
        f"验证问题={','.join(job.issues) or '-'}；similarity={job.similarity}。"
    )


def _parse_advice(term: str, payload: str) -> PronunciationAdvice | None:
    content = (payload or "").strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.IGNORECASE | re.DOTALL).strip()
    try:
        decoded = json.loads(content)
    except json.JSONDecodeError:
        return None
    if isinstance(decoded, list):
        decoded = decoded[0] if decoded else {}
    if not isinstance(decoded, dict):
        return None
    pronunciation = str(decoded.get("pronunciation") or decoded.get("replacement") or "").strip()
    if not pronunciation:
        return None
    confidence = max(0.0, min(1.0, float(decoded.get("confidence") or 0.0)))
    action = str(decoded.get("action") or ("accept" if confidence >= _auto_accept_confidence() else "pending")).strip().lower()
    if action not in {"accept", "pending"}:
        action = "pending"
    return PronunciationAdvice(
        term=str(decoded.get("term") or term).strip() or term,
        pronunciation=pronunciation,
        confidence=confidence,
        reason=str(decoded.get("reason") or "").strip(),
        action=action,
        source="agx",
    )


def request_advice(term: str, job: AdviceJob) -> PronunciationAdvice | None:
    body = {
        "model": _model(),
        "stream": False,
        "options": {"temperature": 0.0, "top_p": 0.7},
        "prompt": _prompt_for(term, job),
    }
    req = request.Request(
        f"{_base_url()}/api/generate",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with request.urlopen(req, timeout=_timeout_s()) as response:
        decoded = json.loads(response.read().decode("utf-8"))
    if not isinstance(decoded, dict):
        return None
    return _parse_advice(term, str(decoded.get("response") or ""))


def apply_advice(advice: PronunciationAdvice, job: AdviceJob) -> None:
    row = {
        "ts": int(time.time()),
        "validation_id": job.validation_id,
        "trace_id": job.trace_id,
        "endpoint": job.endpoint,
        "term": advice.term,
        "pronunciation": advice.pronunciation,
        "confidence": advice.confidence,
        "reason": advice.reason,
        "action": advice.action,
        "source": advice.source,
    }
    _append_jsonl(_audit_path(), row)
    if advice.action == "accept" and advice.confidence >= _auto_accept_confidence():
        terms = {str(k).lower(): str(v) for k, v in _read_json_object(_terms_path()).items()}
        terms[advice.term.lower()] = advice.pronunciation
        _write_json_object(_terms_path(), terms)
        return
    pending = _read_json_object(_pending_path())
    pending[advice.term.lower()] = {
        "replacement": advice.pronunciation,
        "score": round(advice.confidence, 3),
        "reasons": ["agx_advisor"],
        "reason": advice.reason,
        "updated_at": int(time.time()),
    }
    _write_json_object(_pending_path(), pending)


_queue: queue.Queue[AdviceJob] | None = None
_started = False
_lock = threading.Lock()


def _worker_loop() -> None:
    assert _queue is not None
    while True:
        job = _queue.get()
        try:
            existing = {str(key).lower() for key in _read_json_object(_terms_path())}
            for term in extract_candidate_terms(job.expected_text):
                if term.lower() in existing:
                    continue
                advice = request_advice(term, job)
                if advice is not None:
                    apply_advice(advice, job)
                    existing.add(advice.term.lower())
        except (error.URLError, TimeoutError, OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
            _append_jsonl(
                _audit_path(),
                {
                    "ts": int(time.time()),
                    "validation_id": job.validation_id,
                    "trace_id": job.trace_id,
                    "error": f"{type(exc).__name__}: {exc}",
                    "action": "error",
                },
            )
        finally:
            _queue.task_done()


def _ensure_started() -> None:
    global _queue, _started
    if _started:
        return
    with _lock:
        if _started:
            return
        _queue = queue.Queue(maxsize=_queue_size())
        thread = threading.Thread(target=_worker_loop, name="tts-pronunciation-advisor", daemon=True)
        thread.start()
        _started = True


def enqueue_advice(job: AdviceJob) -> bool:
    if not advisor_enabled():
        return False
    _ensure_started()
    assert _queue is not None
    try:
        _queue.put_nowait(job)
        return True
    except queue.Full:
        _append_jsonl(
            _audit_path(),
            {
                "ts": int(time.time()),
                "validation_id": job.validation_id,
                "trace_id": job.trace_id,
                "action": "queue_full",
            },
        )
        return False


def enqueue_from_validation(record: dict[str, Any]) -> bool:
    if not should_request_advice(record):
        return False
    job = AdviceJob(
        validation_id=str(record.get("validation_id") or ""),
        expected_text=str(record.get("expected_text") or ""),
        asr_text=str(record.get("asr_text") or ""),
        issues=tuple(str(item) for item in (record.get("issues") or [])),
        similarity=record.get("similarity"),
        trace_id=str(record.get("trace_id") or ""),
        endpoint=str(record.get("endpoint") or ""),
    )
    if not job.validation_id or not job.expected_text:
        return False
    return enqueue_advice(job)


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ask the AGX pronunciation advisor for one technical term.")
    parser.add_argument("term", help="Technical term to ask about, for example qwen or WebUI.")
    parser.add_argument("--expected-text", default="", help="Original text containing the term.")
    parser.add_argument("--asr-text", default="", help="ASR text observed from the generated audio.")
    parser.add_argument("--issue", action="append", default=[], help="Validation issue, can be repeated.")
    parser.add_argument("--similarity", type=float, default=None, help="Validation similarity score.")
    parser.add_argument("--apply", action="store_true", help="Write accepted advice into the configured lexicon files.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_cli_parser().parse_args(argv)
    job = AdviceJob(
        validation_id=f"manual-{int(time.time())}",
        expected_text=args.expected_text or args.term,
        asr_text=args.asr_text,
        issues=tuple(args.issue),
        similarity=args.similarity,
        endpoint="manual",
    )
    advice = request_advice(args.term, job)
    if advice is None:
        print(json.dumps({"term": args.term, "action": "error", "reason": "no_advice"}, ensure_ascii=False))
        return 1
    print(json.dumps(advice.__dict__, ensure_ascii=False, sort_keys=True))
    if args.apply:
        apply_advice(advice, job)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
