"""Diarization-structured prompt formatting."""

from __future__ import annotations

from typing import Any


def _timestamp(seconds: float) -> str:
    minutes, remainder = divmod(seconds, 60)
    return f"{int(minutes):02d}:{remainder:05.2f}"


def format_segment_prompt(segment: dict[str, Any]) -> str:
    """Format one speaker-time segment for constrained correction."""

    terms = ", ".join(segment.get("retrieved_terms", [])) or "none"
    overlap = str(bool(segment.get("overlap"))).lower()
    return (
        f"[{_timestamp(segment['start'])}-{_timestamp(segment['end'])}] "
        f"{segment['speaker']} | overlap={overlap} | "
        f"confidence={segment['confidence']:.2f}\n"
        f"Raw: {segment['raw_text']}\n"
        f"Retrieved terms: {terms}\n"
        "Correct conservatively. Preserve timestamps, speaker, and uncertainty."
    )
