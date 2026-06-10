"""Constrained correction interface with an auditable rule-based fallback."""

from __future__ import annotations

import re
from typing import Any

from backend.rag import retrieve_terms


CORRECTIONS = {
    r"\bpiano note\b": "pyannote",
    r"\bdiary station\b": "speaker diarization",
    r"\brack\b": "RAG",
    r"\bwhere\b": "WER",
    r"\bthe ear\b": "DER",
}


def _rule_based_text(text: str) -> str:
    corrected = text
    for pattern, replacement in CORRECTIONS.items():
        corrected = re.sub(pattern, replacement, corrected, flags=re.IGNORECASE)
    return corrected


def correct_segments(
    segments: list[dict[str, Any]],
    *,
    mock: bool = True,
) -> list[dict[str, Any]]:
    """Correct segments independently while preserving the audit trail."""

    if not mock:
        raise RuntimeError(
            "API-based LLM correction is scheduled for Phase 4. Use mock mode."
        )

    corrected_segments: list[dict[str, Any]] = []
    for segment in segments:
        updated = dict(segment)
        terms = retrieve_terms(str(segment["raw_text"]))
        updated["retrieved_terms"] = terms
        updated["corrected_text"] = _rule_based_text(str(segment["raw_text"]))
        updated["correction_mode"] = "mock_rule_based"
        updated["correction_note"] = (
            "Overlap detected; verify this conservative glossary correction."
            if segment.get("overlap")
            else "Deterministic glossary correction."
        )
        corrected_segments.append(updated)
    return corrected_segments
