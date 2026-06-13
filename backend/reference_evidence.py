"""Load manifest-backed reference speaker, time, text, and event evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from backend.config import ROOT_DIR


def resolve_project_path(
    path: str | Path,
    *,
    root: str | Path = ROOT_DIR,
) -> Path:
    """Resolve manifest paths relative to the repository root."""

    candidate = Path(path)
    return candidate if candidate.is_absolute() else Path(root) / candidate


def read_reference_json(
    path: str | Path,
    *,
    root: str | Path = ROOT_DIR,
) -> list[dict[str, Any]]:
    """Read a reference JSON list, returning an empty list for blank paths."""

    if not str(path).strip():
        return []
    resolved = resolve_project_path(path, root=root)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Reference JSON must contain a list: {resolved}")
    return [dict(item) for item in payload]


def anchors_to_speaker_turns(
    anchors: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert reference anchors into speaker-turn evidence."""

    turns: list[dict[str, Any]] = []
    for anchor in anchors:
        start = float(anchor.get("start", 0.0))
        end = float(anchor.get("end", 0.0))
        if end <= start:
            continue
        turns.append(
            {
                "start": start,
                "end": end,
                "speaker": str(anchor.get("speaker", "UNKNOWN")),
                "confidence": 1.0,
                "source": str(
                    anchor.get(
                        "annotation_source",
                        "reference_anchor",
                    )
                ),
            }
        )
    return sorted(
        turns,
        key=lambda turn: (
            float(turn["start"]),
            float(turn["end"]),
            str(turn["speaker"]),
        ),
    )


def anchors_to_asr_segments(
    anchors: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert reference text anchors into the ASR segment contract."""

    return [
        {
            "start": float(anchor["start"]),
            "end": float(anchor["end"]),
            "text": str(anchor.get("text", "")).strip(),
            "words": [],
        }
        for anchor in anchors
        if str(anchor.get("text", "")).strip()
    ]


def load_reference_evidence(
    row: dict[str, str],
    *,
    root: str | Path = ROOT_DIR,
) -> dict[str, Any]:
    """Load all available reference evidence for one manifest row."""

    anchors = read_reference_json(
        row.get("anchors_path", ""),
        root=root,
    )
    events = read_reference_json(
        row.get("events_path", ""),
        root=root,
    )
    return {
        "anchors": anchors,
        "speaker_turns": anchors_to_speaker_turns(anchors),
        "asr_segments": anchors_to_asr_segments(anchors),
        "events": events,
    }
