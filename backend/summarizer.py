"""Deterministic transcript-grounded meeting understanding."""

from __future__ import annotations

import re
from typing import Any

from backend.prompting import format_timestamp


ACTION_PATTERN = re.compile(
    r"\b(?:we\s+)?(?:should|need to|must|will)\s+(.+?)(?:[.!?]|$)",
    flags=re.IGNORECASE,
)
QUESTION_TOKENS = re.compile(r"[A-Za-z0-9]+")


def _segment_text(segment: dict[str, Any]) -> str:
    return str(segment.get("corrected_text") or segment.get("raw_text", ""))


def summarize_segments(segments: list[dict[str, Any]]) -> dict[str, Any]:
    """Create an extractive summary and action items from corrected segments."""

    usable = [segment for segment in segments if _segment_text(segment).strip()]
    terms = list(
        dict.fromkeys(
            term
            for segment in usable
            for term in segment.get("retrieved_terms", [])
        )
    )
    summary_parts = [_segment_text(segment).strip() for segment in usable]
    summary = (
        " ".join(summary_parts)
        if summary_parts
        else "No transcript content was available to summarize."
    )

    action_items: list[dict[str, Any]] = []
    for segment in usable:
        for match in ACTION_PATTERN.finditer(_segment_text(segment)):
            action = match.group(1).strip().rstrip(".")
            action_items.append(
                {
                    "text": action[:1].upper() + action[1:] + ".",
                    "speaker": segment["speaker"],
                    "start": segment["start"],
                    "end": segment["end"],
                    "source_text": _segment_text(segment),
                    "uncertain": bool(segment.get("overlap")),
                }
            )

    return {
        "mode": "deterministic_extractive",
        "summary": summary,
        "action_items": action_items,
        "keywords": terms,
        "segment_count": len(usable),
        "note": (
            "Secondary transcript-grounded output. It is extractive and does "
            "not add facts beyond corrected transcript segments."
        ),
    }


def answer_question(
    question: str,
    segments: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return the most lexically relevant transcript segment."""

    query_tokens = {
        token.lower() for token in QUESTION_TOKENS.findall(question)
    }
    ranked: list[tuple[int, int, dict[str, Any]]] = []
    for index, segment in enumerate(segments):
        text = _segment_text(segment)
        text_tokens = {
            token.lower() for token in QUESTION_TOKENS.findall(text)
        }
        score = len(query_tokens & text_tokens)
        if score:
            ranked.append((score, -index, segment))
    if not ranked:
        return {
            "answer": "No transcript segment supports an answer.",
            "source": None,
            "mode": "deterministic_transcript_search",
        }
    _score, _negative_index, best = max(ranked)
    return {
        "answer": _segment_text(best),
        "source": {
            "start": best["start"],
            "end": best["end"],
            "speaker": best["speaker"],
            "timestamp": (
                f"{format_timestamp(best['start'])}-"
                f"{format_timestamp(best['end'])}"
            ),
            "overlap": best.get("overlap", False),
        },
        "mode": "deterministic_transcript_search",
    }
