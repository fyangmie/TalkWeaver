"""Overlap-region utilities."""

from __future__ import annotations

from typing import Any


def detect_overlap_regions(
    speaker_turns: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Find pairwise intersections between turns from different speakers."""

    regions: list[dict[str, Any]] = []
    ordered = sorted(speaker_turns, key=lambda turn: (turn["start"], turn["end"]))
    for index, first in enumerate(ordered):
        for second in ordered[index + 1 :]:
            if second["start"] >= first["end"]:
                break
            if first["speaker"] == second["speaker"]:
                continue
            start = max(float(first["start"]), float(second["start"]))
            end = min(float(first["end"]), float(second["end"]))
            if end > start:
                regions.append(
                    {
                        "start": start,
                        "end": end,
                        "speakers": [first["speaker"], second["speaker"]],
                    }
                )
    return regions


def segment_has_overlap(
    start: float,
    end: float,
    overlap_regions: list[dict[str, Any]],
) -> bool:
    """Return whether a segment intersects any overlap region."""

    return any(
        min(end, float(region["end"])) > max(start, float(region["start"]))
        for region in overlap_regions
    )
