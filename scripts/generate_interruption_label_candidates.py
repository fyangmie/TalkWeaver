#!/usr/bin/env python3
"""Generate human-review interruption label candidates from reference turns."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.events import detect_interruption_events  # noqa: E402
from backend.reference_evidence import load_reference_evidence  # noqa: E402
from scripts.dataset_utils import resolve_repo_path  # noqa: E402


COLUMNS = [
    "clip_id",
    "dataset_name",
    "language",
    "start",
    "end",
    "interrupter",
    "interrupted",
    "label",
    "annotator",
    "candidate_source",
    "notes",
]


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_candidates(
    *,
    manifest: str | Path,
    output: str | Path,
    min_overlap_seconds: float = 0.2,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in _read_manifest(resolve_repo_path(str(manifest))):
        reference = load_reference_evidence(row)
        turns = reference["speaker_turns"]
        if len({str(turn.get("speaker", "")) for turn in turns}) < 2:
            continue
        events = detect_interruption_events(
            turns,
            clip_id=row["clip_id"],
            min_overlap_seconds=min_overlap_seconds,
        )
        for event in events:
            speakers = list(event.speakers)
            interrupted = speakers[0] if speakers else ""
            interrupter = speakers[1] if len(speakers) > 1 else ""
            rows.append(
                {
                    "clip_id": row["clip_id"],
                    "dataset_name": row.get("dataset_name", ""),
                    "language": row.get("language", ""),
                    "start": f"{event.start:.3f}",
                    "end": f"{event.end:.3f}",
                    "interrupter": interrupter,
                    "interrupted": interrupted,
                    "label": "uncertain",
                    "annotator": "",
                    "candidate_source": "rule_floor_takeover",
                    "notes": (
                        "Review audio before changing label to interruption, "
                        "backchannel, or overlap_only."
                    ),
                }
            )
    output_path = resolve_repo_path(str(output))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-overlap-seconds", type=float, default=0.2)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        rows = build_candidates(
            manifest=args.manifest,
            output=args.output,
            min_overlap_seconds=args.min_overlap_seconds,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Interruption candidate generation failed: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote {len(rows)} interruption label candidates: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
