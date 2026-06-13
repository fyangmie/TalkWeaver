#!/usr/bin/env python3
"""Aggregate TalkWeaver workflow ablation results."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


MEAN_COLUMNS = [
    "num_anchors",
    "num_speaker_labeled_anchors",
    "num_overlap_anchors",
    "num_events",
    "num_term_candidates",
    "num_term_rescues_applied",
    "num_correction_audits",
    "num_unsupported_changes",
    "num_needs_review",
    "asr_error_rate",
    "corrected_error_rate",
    "anchor_coverage",
]

SUMMARY_COLUMNS = [
    "variant",
    "dataset_name",
    "language",
    "num_clips",
    *[f"mean_{column}" for column in MEAN_COLUMNS],
    "notes",
]


def _mean(group: list[dict[str, str]], column: str) -> float | str:
    values = [
        float(row[column])
        for row in group
        if row.get(column, "").strip()
    ]
    return round(statistics.fmean(values), 6) if values else ""


def summarize_results(
    input_path: str | Path,
    output_path: str | Path,
) -> list[dict[str, Any]]:
    """Aggregate by variant, dataset, and language."""

    with Path(input_path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Workflow ablation CSV contains no rows.")
    grouped: defaultdict[
        tuple[str, str, str],
        list[dict[str, str]],
    ] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["variant"],
                row["dataset_name"],
                row["language"],
            )
        ].append(row)

    summaries: list[dict[str, Any]] = []
    for (variant, dataset, language), group in sorted(grouped.items()):
        summary: dict[str, Any] = {
            "variant": variant,
            "dataset_name": dataset,
            "language": language,
            "num_clips": len(group),
        }
        summary.update(
            {
                f"mean_{column}": _mean(group, column)
                for column in MEAN_COLUMNS
            }
        )
        summary["notes"] = (
            "Small-subset aggregate. Reference speaker-time rows are "
            "oracle-assisted, not automatic diarization."
        )
        summaries.append(summary)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=SUMMARY_COLUMNS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(summaries)
    return summaries


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        rows = summarize_results(args.input, args.output)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"Workflow ablation summary failed: {exc}")
        return 2
    print(f"Wrote {len(rows)} workflow summary rows: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
