"""Confidence placeholders for temporal-anchor transcript segments."""

from __future__ import annotations


def estimate_confidence(*, overlap: bool, base_confidence: float = 0.91) -> float:
    """Apply a deterministic uncertainty penalty to overlap segments."""

    confidence = base_confidence - (0.29 if overlap else 0.0)
    return round(max(0.0, min(1.0, confidence)), 2)


def uncertainty_label(confidence: float) -> str:
    """Map a confidence value to a review label."""

    if confidence >= 0.8:
        return "high"
    if confidence >= 0.6:
        return "review"
    return "uncertain"
