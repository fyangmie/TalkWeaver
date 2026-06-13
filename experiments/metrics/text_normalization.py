"""Language-aware text normalization for ASR baseline evaluation."""

from __future__ import annotations

import unicodedata


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


def canonical_language(language: str | None) -> str:
    """Return a lowercase language identifier with normalized separators."""

    return str(language or "").strip().lower().replace("_", "-")


def is_mandarin_language(language: str | None) -> bool:
    """Return whether the manifest language should use character error rate."""

    normalized = canonical_language(language)
    return normalized in MANDARIN_LANGUAGE_CODES or normalized.startswith(
        "zh-"
    )


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

    normalized = unicodedata.normalize("NFKC", str(text)).lower()
    return "".join(
        character
        for character in normalized
        if not character.isspace()
        and not unicodedata.category(character).startswith("P")
    )
