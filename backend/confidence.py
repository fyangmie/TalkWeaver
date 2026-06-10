"""Deterministic confidence rules for speaker-attributed transcripts."""

from __future__ import annotations


NORMAL_CONFIDENCE = 0.90
UNKNOWN_CONFIDENCE = 0.50
OVERLAP_CONFIDENCE = 0.55


def estimate_confidence(
    *,
    overlap: bool,
    unknown: bool = False,
    base_confidence: float = NORMAL_CONFIDENCE,
) -> float:
    """Return a stable confidence value for one alignment assignment."""

    if overlap:
        return OVERLAP_CONFIDENCE
    if unknown:
        return UNKNOWN_CONFIDENCE
    return round(max(0.0, min(1.0, base_confidence)), 2)


def confidence_for_assignment(speaker: str, *, overlap: bool) -> float:
    """Map an alignment assignment to its deterministic confidence."""

    return estimate_confidence(
        overlap=overlap,
        unknown=speaker == "UNKNOWN",
    )


def uncertainty_label(confidence: float) -> str:
    """Map a confidence value to a review label."""

    if confidence >= 0.8:
        return "high"
    if confidence >= 0.6:
        return "review"
    return "uncertain"
