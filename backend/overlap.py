"""Overlap-region detection for speaker turns."""

from __future__ import annotations

from typing import Any


def detect_overlap_regions(
    speaker_turns: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Find intervals containing two or more distinct active speakers."""

    normalized = [
        {
            "start": float(turn["start"]),
            "end": float(turn["end"]),
            "speaker": str(turn["speaker"]),
        }
        for turn in speaker_turns
        if float(turn["end"]) > float(turn["start"])
    ]
    boundaries = sorted(
        {
            boundary
            for turn in normalized
            for boundary in (turn["start"], turn["end"])
        }
    )
    regions: list[dict[str, Any]] = []
    for start, end in zip(boundaries, boundaries[1:]):
        if end <= start:
            continue
        midpoint = (start + end) / 2
        speakers = sorted(
            {
                turn["speaker"]
                for turn in normalized
                if turn["start"] <= midpoint < turn["end"]
            }
        )
        if len(speakers) < 2:
            continue

        if (
            regions
            and regions[-1]["speakers"] == speakers
            and abs(float(regions[-1]["end"]) - start) < 1e-9
        ):
            regions[-1]["end"] = round(end, 3)
            regions[-1]["duration"] = round(
                float(regions[-1]["end"]) - float(regions[-1]["start"]),
                3,
            )
            continue

        regions.append(
            {
                "start": round(start, 3),
                "end": round(end, 3),
                "speakers": speakers,
                "duration": round(end - start, 3),
                "type": "overlap",
            }
        )
    return regions


def segment_has_overlap(
    start: float,
    end: float,
    overlap_regions: list[dict[str, Any]],
) -> bool:
    """Return whether an interval intersects any overlap region."""

    return any(
        min(end, float(region["end"])) > max(start, float(region["start"]))
        for region in overlap_regions
    )
