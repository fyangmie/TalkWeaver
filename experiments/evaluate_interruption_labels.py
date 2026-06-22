#!/usr/bin/env python3
"""Summarize human interruption labels without fabricating recall."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Any


LABELS = {"interruption", "backchannel", "overlap_only", "uncertain"}
OUTPUT_COLUMNS = [
    "total_candidates",
    "reviewed_candidates",
    "interruption_labels",
    "backchannel_labels",
    "overlap_only_labels",
    "uncertain_labels",
    "candidate_precision",
    "recall",
    "f1",
    "notes",
]


def evaluate_labels(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    counts = Counter(row.get("label", "").strip() for row in rows)
    unknown = sorted(set(counts) - LABELS)
    if unknown:
        raise ValueError("Unknown label(s): " + ", ".join(unknown))

    reviewed = [
        row for row in rows if row.get("label", "").strip() != "uncertain"
    ]
    interruptions = counts["interruption"]
    candidate_precision = (
        round(interruptions / len(reviewed), 6)
        if reviewed
        else ""
    )
    return {
        "total_candidates": len(rows),
        "reviewed_candidates": len(reviewed),
        "interruption_labels": interruptions,
        "backchannel_labels": counts["backchannel"],
        "overlap_only_labels": counts["overlap_only"],
        "uncertain_labels": counts["uncertain"],
        "candidate_precision": candidate_precision,
        "recall": "",
        "f1": "",
        "notes": (
            "Recall/F1 are unavailable until an exhaustive annotated timeline "
            "or sampled non-candidate negatives are added. This summary does "
            "not treat uncertain rows as gold labels."
        ),
    }


def write_summary(output: str | Path, row: dict[str, Any]) -> None:
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=OUTPUT_COLUMNS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerow(row)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        summary = evaluate_labels(args.labels)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Interruption label evaluation failed: {exc}")
        return 2
    write_summary(args.output, summary)
    print(f"Wrote interruption label summary: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
