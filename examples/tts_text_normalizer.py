"""Text normalization helpers shared by the example TTS servers."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache

READABLE_RE = re.compile(r"[A-Za-z0-9\u4e00-\u9fff]")
IMAGE_PLACEHOLDER_RE = re.compile(r"\[\s*image\s*#?\s*\d+\s*\]", re.IGNORECASE)
INLINE_SEPARATOR_RE = re.compile(r"\s+(?:-{3,}|—{2,})\s+")
TRAILING_SEPARATOR_RE = re.compile(r"\s+(?:-{3,}|—{2,})\s*$")


@dataclass(frozen=True)
class NormalizedText:
    text: str
    changed: bool
    normalizer: str


def has_readable_text(text: str) -> bool:
    return bool(READABLE_RE.search(text or ""))


@lru_cache(maxsize=1)
def _markdown_parser():
    from markdown_it import MarkdownIt

    return MarkdownIt("commonmark", {"html": False})


def _inline_text(children: list | None, fallback: str) -> str:
    if not children:
        return fallback
    parts: list[str] = []
    for child in children:
        if child.type == "image":
            continue
        if child.type in {"text", "code_inline"}:
            parts.append(child.content)
        elif child.type in {"softbreak", "hardbreak"}:
            parts.append(" ")
    return "".join(parts)


def _extract_markdown_text(text: str) -> str:
    tokens = _markdown_parser().parse(text or "")
    parts: list[str] = []
    for token in tokens:
        if token.type != "inline":
            continue
        content = _inline_text(token.children, token.content).strip()
        if content:
            parts.append(content)
    return "\n".join(parts)


def _clean_application_noise(text: str) -> str:
    content = _extract_markdown_text(text)
    content = IMAGE_PLACEHOLDER_RE.sub(" ", content)
    content = INLINE_SEPARATOR_RE.sub("。", content)
    content = TRAILING_SEPARATOR_RE.sub("", content)
    content = re.sub(r"([。.!?！？])(?:\s*[。.!?！？])+", r"\1", content)
    return re.sub(r"\s+", " ", content).strip()


def _normalizer_lang(lang_hint: str | None) -> str:
    hint = (lang_hint or "").strip().lower()
    if hint in {"zh", "cn", "chinese", "中文", "汉语"}:
        return "zh"
    if hint in {"en", "english", "英文", "英语"}:
        return "en"
    if hint in {"ja", "jp", "japanese", "日文", "日语"}:
        return "ja"
    return "auto"


@lru_cache(maxsize=4)
def _wetext_normalizer(lang: str):
    from wetext import Normalizer

    return Normalizer(
        lang=lang,
        operator="tn",
        traditional_to_simple=True,
        full_to_half=True,
        remove_puncts=False,
    )


def _normalize_with_wetext(text: str, lang_hint: str | None) -> str:
    lang = _normalizer_lang(lang_hint)
    return _wetext_normalizer(lang).normalize(text)


def normalize_for_tts(text: str, lang_hint: str | None = None) -> NormalizedText:
    original = text or ""
    content = _clean_application_noise(original)
    if not content or not has_readable_text(content):
        return NormalizedText("", content != original.strip(), "basic")

    requested = os.getenv("QWEN_TTS_NORMALIZER", "wetext").strip().lower()
    if requested in {"off", "none", "basic"}:
        return NormalizedText(content, content != original.strip(), "basic")

    try:
        normalized = _normalize_with_wetext(content, lang_hint)
        normalized = _clean_application_noise(normalized)
        if normalized and has_readable_text(normalized):
            return NormalizedText(normalized, normalized != original.strip(), "wetext")
    except Exception:
        pass

    return NormalizedText(content, content != original.strip(), "basic")
