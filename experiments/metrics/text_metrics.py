"""Dependency-light WER and CER implementations."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

from experiments.metrics.text_normalization import (
    canonical_language,
    is_mandarin_language,
    normalize_for_cer,
    normalize_for_wer,
)


def edit_distance(reference: Sequence[Any], hypothesis: Sequence[Any]) -> int:
    """Compute Levenshtein distance for arbitrary token sequences."""

    previous = list(range(len(hypothesis) + 1))
    for row_index, reference_item in enumerate(reference, start=1):
        current = [row_index]
        for column_index, hypothesis_item in enumerate(
            hypothesis,
            start=1,
        ):
            substitution = previous[column_index - 1] + (
                reference_item != hypothesis_item
            )
            insertion = current[column_index - 1] + 1
            deletion = previous[column_index] + 1
            current.append(min(substitution, insertion, deletion))
        previous = current
    return previous[-1]


def _rate(reference: Sequence[Any], hypothesis: Sequence[Any]) -> float:
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return edit_distance(reference, hypothesis) / len(reference)


def word_error_rate(
    reference: str,
    hypothesis: str,
    *,
    normalized: bool = False,
    prefer_jiwer: bool = True,
) -> float:
    """Compute WER with optional jiwer acceleration and a local fallback."""

    normalized_reference = (
        reference if normalized else normalize_for_wer(reference)
    )
    normalized_hypothesis = (
        hypothesis if normalized else normalize_for_wer(hypothesis)
    )
    if prefer_jiwer:
        try:
            from jiwer import wer
        except ImportError:
            pass
        else:
            return float(wer(normalized_reference, normalized_hypothesis))
    return _rate(
        normalized_reference.split(),
        normalized_hypothesis.split(),
    )


def character_error_rate(
    reference: str,
    hypothesis: str,
    *,
    normalized: bool = False,
) -> float:
    """Compute CER over normalized Unicode characters."""

    normalized_reference = (
        reference if normalized else normalize_for_cer(reference)
    )
    normalized_hypothesis = (
        hypothesis if normalized else normalize_for_cer(hypothesis)
    )
    return _rate(normalized_reference, normalized_hypothesis)


def metric_name_for_language(language: str | None) -> str:
    """Select CER for Mandarin and WER for other manifest languages."""

    if is_mandarin_language(language):
        return "CER"
    normalized = canonical_language(language)
    if normalized not in {"en", "en-us", "en-gb", "fr", "fr-fr"}:
        warnings.warn(
            f"Unknown language {language!r}; using WER.",
            RuntimeWarning,
            stacklevel=2,
        )
    return "WER"


def evaluate_text(
    reference: str,
    hypothesis: str,
    language: str | None,
) -> dict[str, str | float]:
    """Normalize and score one reference/hypothesis pair."""

    metric_name = metric_name_for_language(language)
    if metric_name == "CER":
        normalized_reference = normalize_for_cer(reference)
        normalized_hypothesis = normalize_for_cer(hypothesis)
        error_rate = character_error_rate(
            normalized_reference,
            normalized_hypothesis,
            normalized=True,
        )
    else:
        normalized_reference = normalize_for_wer(reference)
        normalized_hypothesis = normalize_for_wer(hypothesis)
        error_rate = word_error_rate(
            normalized_reference,
            normalized_hypothesis,
            normalized=True,
        )
    return {
        "metric_name": metric_name,
        "error_rate": float(error_rate),
        "normalized_reference": normalized_reference,
        "normalized_hypothesis": normalized_hypothesis,
    }
