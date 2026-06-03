#!/usr/bin/env python3
"""Pronunciation lexicon learner for lowercase technical terms."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

DEFAULT_TERMS = {
    "ai": "A I",
    "api": "A P I",
    "asr": "A S R",
    "gpu": "G P U",
    "llm": "L L M",
    "ocr": "O C R",
    "tts": "T T S",
    "ui": "U I",
}
TECH_CONTEXT_RE = re.compile(
    r"(模型|服务|接口|识别|语音|推理|部署|设备|显卡|算法|系统|目录|端口|"
    r"api|gpu|tts|asr|ai|model|server|audio|speech|code|agent)",
    re.IGNORECASE,
)
TOKEN_RE = re.compile(r"(?<![A-Za-z0-9_])([a-z]{2,8})(?![A-Za-z0-9_])")
COMMON_WORDS = {
    "am",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "go",
    "in",
    "is",
    "it",
    "of",
    "on",
    "open",
    "or",
    "the",
    "to",
    "us",
    "video",
    "with",
}

def _config_dir() -> Path:
    return Path(os.getenv("QWEN_TTS_CONFIG_DIR", "~/.config/faster-qwen3-tts")).expanduser()


def _terms_path() -> Path:
    return Path(os.getenv("QWEN_TTS_PRONUNCIATION_TERMS", str(_config_dir() / "pronunciation_terms.json"))).expanduser()


def _pending_path() -> Path:
    return Path(os.getenv("QWEN_TTS_PRONUNCIATION_PENDING", str(_config_dir() / "pronunciation_terms.pending.json"))).expanduser()


def _audit_path() -> Path:
    return Path(os.getenv("QWEN_TTS_PRONUNCIATION_AUDIT", str(_config_dir() / "pronunciation_terms.audit.jsonl"))).expanduser()


def _candidate_log_path() -> Path:
    return Path(
        os.getenv("QWEN_TTS_PRONUNCIATION_CANDIDATES", str(_config_dir() / "pronunciation_candidates.jsonl"))
    ).expanduser()


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


def _spell_letters(token: str) -> str:
    return " ".join(token.upper())


def _read_candidate_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []
    except Exception:
        return []
    for line in lines:
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _extract_candidates(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    for item in rows:
        token = str(item.get("token") or "").lower()
        if not re.fullmatch(r"[a-z]{2,8}", token):
            continue
        row = stats.setdefault(token, {"count": 0, "fail_count": 0, "contexts": Counter()})
        row["count"] += 1
        issues = item.get("issues") or []
        if "text_mismatch" in issues or "possible_truncation" in issues or not bool(item.get("present_in_asr")):
            row["fail_count"] += 1
        context = f"{item.get('expected_text') or ''} {item.get('asr_text') or ''}"
        if TECH_CONTEXT_RE.search(context):
            row["contexts"]["technical"] += 1
    return stats


def _score(token: str, row: dict[str, Any]) -> tuple[float, list[str]]:
    reasons: list[str] = []
    score = 0.0
    if token in DEFAULT_TERMS:
        score += 0.8
        reasons.append("default_term")
    if 2 <= len(token) <= 8:
        score += 0.1
        reasons.append("short_token")
    if int(row.get("fail_count") or 0) > 0:
        score += 0.2
        reasons.append("validation_failure")
    if int(row.get("count") or 0) >= 3:
        score += 0.1
        reasons.append("repeated")
    contexts = row.get("contexts") or {}
    if int(contexts.get("technical") or 0) > 0:
        score += 0.2
        reasons.append("technical_context")
    if token in COMMON_WORDS:
        score -= 0.6
        reasons.append("common_word_penalty")
    return max(0.0, min(1.0, score)), reasons


def learn_terms(*, threshold: float, dry_run: bool) -> dict[str, Any]:
    existing = {str(k).lower(): str(v) for k, v in _read_json_object(_terms_path()).items()}
    pending = _read_json_object(_pending_path())
    rows = _read_candidate_rows(_candidate_log_path())
    candidates = _extract_candidates(rows)
    accepted: dict[str, str] = {}
    pending_updates: dict[str, Any] = {}

    for token, row in sorted(candidates.items()):
        if token in existing:
            continue
        score, reasons = _score(token, row)
        item = {
            "replacement": _spell_letters(token),
            "score": round(score, 3),
            "count": row.get("count", 0),
            "fail_count": row.get("fail_count", 0),
            "reasons": reasons,
            "updated_at": int(time.time()),
        }
        if score >= threshold:
            accepted[token] = item["replacement"]
        elif score >= 0.6:
            pending_updates[token] = item

    if not dry_run:
        if accepted:
            existing.update(accepted)
            _write_json_object(_terms_path(), existing)
            _audit_path().parent.mkdir(parents=True, exist_ok=True)
            with _audit_path().open("a", encoding="utf-8") as fh:
                for token, replacement in accepted.items():
                    fh.write(json.dumps({"ts": int(time.time()), "token": token, "replacement": replacement}, ensure_ascii=False) + "\n")
        if pending_updates:
            pending.update(pending_updates)
            _write_json_object(_pending_path(), pending)

    return {
        "success": True,
        "dry_run": dry_run,
        "candidate_rows": len(rows),
        "accepted": accepted,
        "pending": pending_updates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=float(os.getenv("QWEN_TTS_LEARN_THRESHOLD", "0.85")))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(json.dumps(learn_terms(threshold=args.threshold, dry_run=args.dry_run), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
