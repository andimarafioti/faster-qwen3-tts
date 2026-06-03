"""Text normalization helpers shared by the example TTS servers."""

from __future__ import annotations

import os
import re
import unicodedata
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

READABLE_RE = re.compile(r"[A-Za-z0-9\u4e00-\u9fff]")
IMAGE_PLACEHOLDER_RE = re.compile(r"\[\s*image\s*#?\s*\d+\s*\]", re.IGNORECASE)
INLINE_SEPARATOR_RE = re.compile(r"\s+(?:-{3,}|—{2,})\s+")
TRAILING_SEPARATOR_RE = re.compile(r"\s+(?:-{3,}|—{2,})\s*$")
INVISIBLE_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060\ufeff]")
TECH_TOKEN_RE = re.compile(
    r"(?P<env>\b[A-Z][A-Z0-9_]{2,}=[A-Za-z0-9_./:-]+)"
    r"|(?P<cli>(?<!\w)--[A-Za-z0-9][A-Za-z0-9_-]*)"
    r"|(?P<api_path>(?<!\w)/(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+)"
    r"|(?P<symbol>!=|==|->|=>|<=|>=)"
    r"|(?P<colon_none>(?P<colon_prefix>[\u4e00-\u9fffA-Za-z0-9]+):无)"
    r"|(?P<dotted>\b[A-Za-z][A-Za-z0-9_-]*(?:\.[A-Za-z0-9_-]+)+\b)"
    r"|(?P<hyphen>\b[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+\b)"
    r"|(?P<mixed>\b(?=[A-Za-z0-9]*[A-Z])(?=[A-Za-z0-9]*[a-z])[A-Za-z]+[A-Za-z0-9]*\b)"
    r"|(?P<upper>(?<![A-Za-z0-9_])[A-Z]{2,}(?![A-Za-z0-9_]))"
    r"|(?P<pinyin>\b[a-z]{4,}\b)"
)

DIGIT_REPLACEMENTS = {
    "0": "零",
    "1": "一",
    "2": "二",
    "3": "三",
    "4": "四",
    "5": "五",
    "6": "六",
    "7": "七",
    "8": "八",
    "9": "九",
}
CODE_SYMBOL_REPLACEMENTS = {
    "!=": " 不等于 ",
    "==": " 等于 ",
    "->": " 箭头 ",
    "=>": " 箭头 ",
    "<=": " 小于等于 ",
    ">=": " 大于等于 ",
}
PINYIN_CARRIER_OVERRIDES = {
    "si": "斯",
    "yuan": "元",
    "xiao": "肖",
    "hong": "红",
    "shu": "书",
}
DEFAULT_PRONUNCIATION_TERMS = {
    "ai": "A I",
    "api": "A P I",
    "asr": "A S R",
    "gpu": "G P U",
    "llm": "L L M",
    "ocr": "O C R",
    "tts": "T T S",
    "ui": "U I",
}


@dataclass(frozen=True)
class NormalizedText:
    text: str
    changed: bool
    normalizer: str
    normalization_trace: tuple[dict[str, str], ...] = ()


def has_readable_text(text: str) -> bool:
    return bool(READABLE_RE.search(text or ""))


@lru_cache(maxsize=1)
def _markdown_parser():
    from markdown_it import MarkdownIt

    return MarkdownIt("commonmark", {"html": False})


@lru_cache(maxsize=1)
def _pinyin_tokenizer():
    from py_pinyin_split import PinyinTokenizer

    return PinyinTokenizer()


def _english_zipf_frequency(token: str) -> float:
    try:
        from wordfreq import zipf_frequency

        return float(zipf_frequency(token, "en"))
    except Exception:
        return 0.0


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
    content = unicodedata.normalize("NFKC", text or "")
    content = INVISIBLE_RE.sub("", content)
    content = _extract_markdown_text(content)
    content = IMAGE_PLACEHOLDER_RE.sub(" ", content)
    content = INLINE_SEPARATOR_RE.sub("。", content)
    content = TRAILING_SEPARATOR_RE.sub("", content)
    content = re.sub(r"([。.!?！？])(?:\s*[。.!?！？])+", r"\1", content)
    return re.sub(r"\s+", " ", content).strip()


def _enabled_env(name: str, default: str = "1") -> bool:
    return os.getenv(name, default).strip().lower() not in {"0", "false", "off", "no"}


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


def _spell_letters(text: str) -> str:
    return " ".join(char.upper() for char in text if char.isalpha())


def _spell_digits(text: str) -> str:
    return " ".join(DIGIT_REPLACEMENTS.get(char, char) for char in text)


def _wordish_parts(text: str) -> list[str]:
    parts = re.split(r"([A-Z]+(?=[A-Z][a-z]|\d|$)|[A-Z]?[a-z]+|[0-9]+)", text)
    return [part for part in parts if part]


def _verbalize_part(part: str) -> str:
    if not part:
        return ""
    if part.isdigit():
        return _spell_digits(part)
    if part.isupper() and len(part) > 1:
        return _spell_letters(part)
    return part


def _verbalize_identifier(text: str) -> str:
    fragments: list[str] = []
    for group in re.split(r"([._/-])", text):
        if not group or group in "._/-":
            continue
        for part in _wordish_parts(group):
            verbalized = _verbalize_part(part)
            if verbalized:
                fragments.append(verbalized)
    return " ".join(fragments)


def _config_dir() -> Path:
    return Path(os.getenv("QWEN_TTS_CONFIG_DIR", "~/.config/faster-qwen3-tts")).expanduser()


def _pronunciation_terms_path() -> Path:
    return Path(os.getenv("QWEN_TTS_PRONUNCIATION_TERMS", str(_config_dir() / "pronunciation_terms.json"))).expanduser()


@lru_cache(maxsize=8)
def _load_user_pronunciation_terms(path_value: str) -> tuple[tuple[str, str], ...]:
    path = Path(path_value).expanduser()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return ()
    except Exception:
        return ()
    if not isinstance(payload, dict):
        return ()
    rows: list[tuple[str, str]] = []
    for key, value in payload.items():
        source = str(key).strip().lower()
        replacement = str(value).strip()
        if re.fullmatch(r"[a-z]{2,12}", source) and replacement:
            rows.append((source, replacement))
    return tuple(sorted(rows))


def pronunciation_terms() -> dict[str, str]:
    terms = dict(DEFAULT_PRONUNCIATION_TERMS)
    terms.update(dict(_load_user_pronunciation_terms(str(_pronunciation_terms_path()))))
    return terms


def _is_likely_english_word(token: str) -> bool:
    return _english_zipf_frequency(token) >= float(os.getenv("QWEN_TTS_ENGLISH_WORD_ZIPF_MIN", "2.7"))


def _split_pinyin_syllables(token: str) -> list[str] | None:
    if not _enabled_env("QWEN_TTS_PINYIN_FALLBACK", "1"):
        return None
    if _is_likely_english_word(token):
        return None
    try:
        syllables = _pinyin_tokenizer().tokenize(token)
    except ValueError:
        return None
    if len(syllables) < 2 or "".join(syllables).lower() != token.lower():
        return None
    return [str(item).lower() for item in syllables]


def _pinyin_carrier(syllable: str) -> str:
    return PINYIN_CARRIER_OVERRIDES.get(syllable, syllable)


def _trace(source: str, kind: str, strategy: str, replacement: str, confidence: str) -> dict[str, str]:
    return {
        "source": source,
        "kind": kind,
        "strategy": strategy,
        "replacement": replacement,
        "confidence": confidence,
    }


def _apply_pronunciation_terms(text: str) -> tuple[str, tuple[dict[str, str], ...]]:
    if not _enabled_env("QWEN_TTS_PRONUNCIATION_TERMS_ENABLED", "1"):
        return text, ()
    terms = pronunciation_terms()
    if not terms:
        return text, ()

    trace: list[dict[str, str]] = []

    def replace(match: re.Match[str]) -> str:
        source = match.group(0)
        replacement = terms.get(source.lower())
        if not replacement:
            return source
        trace.append(_trace(source, "pronunciation_term", "lexicon", replacement, "high"))
        return f" {replacement} "

    pattern = re.compile(
        r"(?<![A-Za-z0-9_])(" + "|".join(re.escape(key) for key in sorted(terms, key=len, reverse=True)) + r")(?![A-Za-z0-9_])",
        re.IGNORECASE,
    )
    content = pattern.sub(replace, text)
    return re.sub(r"\s+", " ", content).strip(), tuple(trace)


def _verbalize_tech_match(match: re.Match[str]) -> tuple[str, dict[str, str] | None]:
    source = match.group(0)
    if match.lastgroup == "env":
        key, value = source.split("=", 1)
        replacement = f" 环境变量 {_verbalize_identifier(key)} 等于 {_verbalize_identifier(value)} "
        return replacement, _trace(source, "env_assignment", "structure", replacement.strip(), "high")
    if match.lastgroup == "cli":
        replacement = f" 参数 {_verbalize_identifier(source[2:])} "
        return replacement, _trace(source, "cli_flag", "structure", replacement.strip(), "high")
    if match.lastgroup == "api_path":
        replacement = f" 路径 {_verbalize_identifier(source)} "
        return replacement, _trace(source, "api_path", "structure", replacement.strip(), "high")
    if match.lastgroup == "symbol":
        replacement = CODE_SYMBOL_REPLACEMENTS[source]
        return replacement, _trace(source, "code_symbol", "operator", replacement.strip(), "high")
    if match.lastgroup == "colon_none":
        prefix = match.group("colon_prefix")
        replacement = f"{prefix}，无"
        return replacement, _trace(source, "colon_value", "punctuation", replacement, "high")
    if match.lastgroup in {"dotted", "hyphen", "mixed"}:
        replacement = f" {_verbalize_identifier(source)} "
        kind = "mixed_identifier" if match.lastgroup == "mixed" else match.lastgroup
        return replacement, _trace(source, kind, "boundary_split", replacement.strip(), "medium")
    if match.lastgroup == "upper":
        replacement = f" {_spell_letters(source)} "
        return replacement, _trace(source, "acronym", "spell_letters", replacement.strip(), "medium")
    if match.lastgroup == "pinyin":
        syllables = _split_pinyin_syllables(source)
        if syllables is not None:
            replacement = f" {' '.join(_pinyin_carrier(item) for item in syllables)} "
            return replacement, _trace(source, "pinyin_fallback", "py_pinyin_split", replacement.strip(), "low")
    return source, None


def _normalize_technical_tokens(text: str) -> tuple[str, tuple[dict[str, str], ...]]:
    if not _enabled_env("QWEN_TTS_TECH_NORMALIZER", "1"):
        return text, ()
    text, term_trace = _apply_pronunciation_terms(text)
    trace: list[dict[str, str]] = []
    parts: list[str] = []
    last = 0
    for match in TECH_TOKEN_RE.finditer(text):
        parts.append(text[last : match.start()])
        replacement, item = _verbalize_tech_match(match)
        parts.append(replacement)
        if item is not None and item["source"] != item["replacement"]:
            trace.append(item)
        last = match.end()
    parts.append(text[last:])
    content = re.sub(r"\s+", " ", "".join(parts)).strip()
    return content, term_trace + tuple(trace)


def normalize_for_tts(text: str, lang_hint: str | None = None) -> NormalizedText:
    original = text or ""
    content = _clean_application_noise(original)
    if not content or not has_readable_text(content):
        return NormalizedText("", content != original.strip(), "basic")

    content, trace = _normalize_technical_tokens(content)
    normalizer_name = "basic+tech" if trace else "basic"

    requested = os.getenv("QWEN_TTS_NORMALIZER", "wetext").strip().lower()
    if requested in {"off", "none", "basic"}:
        return NormalizedText(content, content != original.strip(), normalizer_name, trace)

    try:
        normalized = _normalize_with_wetext(content, lang_hint)
        normalized = _clean_application_noise(normalized)
        if normalized and has_readable_text(normalized):
            name = "wetext+tech" if trace else "wetext"
            return NormalizedText(normalized, normalized != original.strip(), name, trace)
    except Exception:
        pass

    return NormalizedText(content, content != original.strip(), normalizer_name, trace)
