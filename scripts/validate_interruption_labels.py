#!/usr/bin/env python3
"""Validate human interruption labels against a manifest."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.reference_evidence import load_reference_evidence  # noqa: E402
from scripts.dataset_utils import resolve_repo_path  # noqa: E402


VALID_LABELS = {"interruption", "backchannel", "overlap_only", "uncertain"}
REQUIRED_COLUMNS = {
    "clip_id",
    "start",
    "end",
    "interrupter",
    "interrupted",
    "label",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_labels(
    *,
    labels_path: str | Path,
    manifest: str | Path,
) -> list[str]:
    label_path = resolve_repo_path(str(labels_path))
    manifest_rows = {
        row["clip_id"]: row
        for row in _read_csv(resolve_repo_path(str(manifest)))
    }
    with label_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(REQUIRED_COLUMNS - set(reader.fieldnames or []))
        if missing:
            return [f"{label_path} missing required columns: {', '.join(missing)}"]
        rows = list(reader)

    speakers_by_clip: dict[str, set[str]] = defaultdict(set)
    duration_by_clip: dict[str, float] = {}
    for clip_id, manifest_row in manifest_rows.items():
        reference = load_reference_evidence(manifest_row)
        speakers_by_clip[clip_id] = {
            str(turn.get("speaker", ""))
            for turn in reference["speaker_turns"]
            if str(turn.get("speaker", ""))
        }
        duration_by_clip[clip_id] = float(
            manifest_row.get("duration_seconds") or 0.0
        )

    errors: list[str] = []
    for index, row in enumerate(rows, start=2):
        clip_id = row.get("clip_id", "")
        if clip_id not in manifest_rows:
            errors.append(f"line {index}: unknown clip_id {clip_id!r}")
            continue
        label = row.get("label", "")
        if label not in VALID_LABELS:
            errors.append(f"line {index}: invalid label {label!r}")
        try:
            start = float(row.get("start", ""))
            end = float(row.get("end", ""))
        except ValueError:
            errors.append(f"line {index}: start/end must be numeric")
            continue
        if end <= start:
            errors.append(f"line {index}: end must be greater than start")
        duration = duration_by_clip[clip_id]
        if start < 0 or end > duration + 1e-6:
            errors.append(
                f"line {index}: interval {start:.3f}-{end:.3f} outside clip duration {duration:.3f}"
            )
        speakers = speakers_by_clip[clip_id]
        for column in ("interrupter", "interrupted"):
            value = row.get(column, "")
            if value and value not in speakers:
                errors.append(
                    f"line {index}: {column} {value!r} not in reference speakers"
                )
    return errors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        errors = validate_labels(labels_path=args.labels, manifest=args.manifest)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Interruption label validation failed: {exc}", file=sys.stderr)
        return 2
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print(f"Validated interruption labels: {args.labels}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
