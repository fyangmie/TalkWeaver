"""Language-aware text normalization for ASR baseline evaluation."""

from __future__ import annotations

import unicodedata
import warnings
from functools import lru_cache
from typing import Any


MANDARIN_LANGUAGE_CODES = {
    "cmn",
    "cmn-cn",
    "mandarin",
    "zh",
    "zh-cn",
    "zh-hans",
    "zh_cn",
}
INTERNAL_PUNCTUATION = {"'", "-", ".", "_"}
DISFLUENCY_TOKENS = {
    "um",
    "uh",
    "mm-hmm",
    "mm",
    "hmm",
    "mhm",
    "mmhmm",
}


def canonical_language(language: str | None) -> str:
    """Return a lowercase language identifier with normalized separators."""

    return str(language or "").strip().lower().replace("_", "-")


def is_mandarin_language(language: str | None) -> bool:
    """Return whether the manifest language should use character error rate."""

    normalized = canonical_language(language)
    return normalized in MANDARIN_LANGUAGE_CODES or normalized.startswith(
        "zh-"
    )


@lru_cache(maxsize=1)
def _chinese_converter() -> Any | None:
    try:
        from opencc import OpenCC
    except ImportError:
        return None
    return OpenCC("t2s")


def chinese_script_normalization_available() -> bool:
    """Return whether Traditional-to-Simplified conversion is available."""

    return _chinese_converter() is not None


def chinese_script_normalization_notes() -> str:
    """Describe the active Chinese script normalization policy."""

    if chinese_script_normalization_available():
        return (
            "OpenCC t2s normalization applied before Mandarin CER scoring."
        )
    return (
        "OpenCC unavailable; Mandarin CER retains the original Chinese "
        "script, so Traditional/Simplified differences may inflate CER."
    )


def normalize_chinese_script(text: str) -> str:
    """Convert Traditional Chinese to Simplified when OpenCC is available."""

    converter = _chinese_converter()
    if converter is None:
        warnings.warn(
            chinese_script_normalization_notes(),
            RuntimeWarning,
            stacklevel=2,
        )
        return str(text)
    return str(converter.convert(str(text)))


def _is_internal_punctuation(text: str, index: int) -> bool:
    if text[index] not in INTERNAL_PUNCTUATION:
        return False
    if index == 0 or index == len(text) - 1:
        return False
    return text[index - 1].isalnum() and text[index + 1].isalnum()


def normalize_for_wer(text: str) -> str:
    """Normalize whitespace-language text without splitting technical terms."""

    normalized = unicodedata.normalize("NFKC", str(text)).lower()
    normalized = normalized.replace("’", "'").replace("`", "'")
    characters: list[str] = []
    for index, character in enumerate(normalized):
        category = unicodedata.category(character)
        if character.isspace():
            characters.append(" ")
        elif category.startswith("P"):
            characters.append(
                character
                if _is_internal_punctuation(normalized, index)
                else " "
            )
        else:
            characters.append(character)
    return " ".join("".join(characters).split())


def normalize_for_cer(text: str) -> str:
    """Normalize Mandarin text into a punctuation-free character sequence."""

    normalized = unicodedata.normalize(
        "NFKC",
        normalize_chinese_script(text),
    ).lower()
    return "".join(
        character
        for character in normalized
        if not character.isspace()
        and not unicodedata.category(character).startswith("P")
    )


def normalize_for_cleaned_wer(text: str) -> str:
    """Remove common meeting fillers and adjacent repeated words."""

    tokens = [
        token
        for token in normalize_for_wer(text).split()
        if token not in DISFLUENCY_TOKENS
    ]
    collapsed: list[str] = []
    for token in tokens:
        if not collapsed or token != collapsed[-1]:
            collapsed.append(token)
    return " ".join(collapsed)
