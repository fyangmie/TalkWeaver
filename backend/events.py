"""Conservative overlap and interruption event detection."""

from __future__ import annotations

from typing import Any

from backend.overlap import detect_overlap_regions
from backend.schemas import ConversationEvent


def _turn_value(turn: Any, name: str, default: Any = None) -> Any:
    if isinstance(turn, dict):
        return turn.get(name, default)
    return getattr(turn, name, default)


def _normalize_turns(speaker_turns: list[Any]) -> list[dict[str, Any]]:
    return sorted(
        [
            {
                "speaker": str(_turn_value(turn, "speaker", "UNKNOWN")),
                "start": float(_turn_value(turn, "start", 0.0)),
                "end": float(_turn_value(turn, "end", 0.0)),
            }
            for turn in speaker_turns
            if float(_turn_value(turn, "end", 0.0))
            > float(_turn_value(turn, "start", 0.0))
        ],
        key=lambda item: (item["start"], item["end"], item["speaker"]),
    )


def detect_overlap_events(
    speaker_turns: list[Any],
    *,
    clip_id: str = "unknown",
) -> list[ConversationEvent]:
    """Convert simultaneous speaker intervals into auditable events."""

    regions = detect_overlap_regions(_normalize_turns(speaker_turns))
    return [
        ConversationEvent(
            event_id=f"{clip_id}_overlap_{index:03d}",
            clip_id=clip_id,
            type="overlap",
            start=float(region["start"]),
            end=float(region["end"]),
            speakers=list(region["speakers"]),
            description=(
                "Simultaneous speech interval detected from overlapping "
                "speaker-turn timestamps."
            ),
            severity=(
                "high"
                if float(region["duration"]) >= 1.0
                else "medium"
            ),
            notes=[
                "Rule-based interval event; not a semantic judgment.",
            ],
        )
        for index, region in enumerate(regions, start=1)
    ]


def detect_interruption_events(
    speaker_turns: list[Any],
    *,
    clip_id: str = "unknown",
    min_overlap_seconds: float = 0.2,
) -> list[ConversationEvent]:
    """Detect conservative floor-taking candidates from turn geometry.

    A later speaker must start before the active turn ends, overlap by at
    least ``min_overlap_seconds``, and continue after the earlier turn. This
    is a timing proxy for interruption, not a discourse-level ground truth.
    """

    turns = _normalize_turns(speaker_turns)
    events: list[ConversationEvent] = []
    seen: set[tuple[str, str, float, float]] = set()
    for earlier in turns:
        for later in turns:
            if later["speaker"] == earlier["speaker"]:
                continue
            if later["start"] <= earlier["start"]:
                continue
            overlap_start = later["start"]
            overlap_end = min(earlier["end"], later["end"])
            overlap_duration = overlap_end - overlap_start
            takes_floor = later["end"] > earlier["end"]
            if overlap_duration < min_overlap_seconds or not takes_floor:
                continue
            key = (
                earlier["speaker"],
                later["speaker"],
                overlap_start,
                overlap_end,
            )
            if key in seen:
                continue
            seen.add(key)
            events.append(
                ConversationEvent(
                    event_id=f"{clip_id}_interruption_{len(events) + 1:03d}",
                    clip_id=clip_id,
                    type="interruption",
                    start=round(overlap_start, 3),
                    end=round(overlap_end, 3),
                    speakers=[earlier["speaker"], later["speaker"]],
                    description=(
                        f"{later['speaker']} starts before "
                        f"{earlier['speaker']} finishes and continues after "
                        "the earlier turn."
                    ),
                    severity=(
                        "high" if overlap_duration >= 1.0 else "medium"
                    ),
                    notes=[
                        "Conservative timing-based floor-taking candidate.",
                        "Human review is required before treating this as intent.",
                    ],
                )
            )
    return events


def attach_event_evidence(
    events: list[ConversationEvent],
    anchors: list[Any],
) -> list[ConversationEvent]:
    """Attach IDs of temporal anchors intersecting each event."""

    for event in events:
        event.evidence_anchor_ids = [
            str(
                anchor.get("anchor_id")
                if isinstance(anchor, dict)
                else anchor.anchor_id
            )
            for anchor in anchors
            if min(
                event.end,
                float(
                    anchor.get("end")
                    if isinstance(anchor, dict)
                    else anchor.end
                ),
            )
            > max(
                event.start,
                float(
                    anchor.get("start")
                    if isinstance(anchor, dict)
                    else anchor.start
                ),
            )
        ]
    return events
