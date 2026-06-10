"""Secondary meeting-understanding placeholders."""

from __future__ import annotations

from typing import Any


def summarize_segments(segments: list[dict[str, Any]]) -> dict[str, Any]:
    """Create a clearly labeled deterministic demo summary."""

    terms = sorted(
        {
            term
            for segment in segments
            for term in segment.get("retrieved_terms", [])
        }
    )
    return {
        "mode": "mock_demo",
        "summary": (
            "The speakers discuss diarization tooling, a RAG glossary, and "
            "evaluation with WER and DER."
        ),
        "action_items": [
            "Implement real ASR and preprocessing in Phase 2.",
            "Evaluate overlap-aware correction against ground truth.",
        ],
        "keywords": terms,
        "note": "Secondary demo output; not a model-generated research result.",
    }
