"""ASR/diarization alignment and temporal-anchor formatting."""

from __future__ import annotations

from typing import Any

from backend.confidence import estimate_confidence, uncertainty_label
from backend.overlap import segment_has_overlap


def _speaker_for_segment(
    segment: dict[str, Any],
    speaker_turns: list[dict[str, Any]],
) -> str:
    midpoint = (float(segment["start"]) + float(segment["end"])) / 2
    matching = [
        turn
        for turn in speaker_turns
        if float(turn["start"]) <= midpoint < float(turn["end"])
    ]
    if not matching:
        return "UNKNOWN"
    return max(
        matching,
        key=lambda turn: min(float(segment["end"]), float(turn["end"]))
        - max(float(segment["start"]), float(turn["start"])),
    )["speaker"]


def align_segments(
    asr_segments: list[dict[str, Any]],
    speaker_turns: list[dict[str, Any]],
    overlap_regions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build the initial temporal-anchor transcript."""

    aligned: list[dict[str, Any]] = []
    for segment in asr_segments:
        start = float(segment["start"])
        end = float(segment["end"])
        overlap = segment_has_overlap(start, end, overlap_regions)
        confidence = estimate_confidence(overlap=overlap)
        aligned.append(
            {
                "start": start,
                "end": end,
                "speaker": _speaker_for_segment(segment, speaker_turns),
                "raw_text": str(segment["text"]),
                "corrected_text": str(segment["text"]),
                "overlap": overlap,
                "confidence": confidence,
                "uncertainty": uncertainty_label(confidence),
                "retrieved_terms": [],
            }
        )
    return aligned
