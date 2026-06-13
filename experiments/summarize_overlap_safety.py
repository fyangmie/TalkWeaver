#!/usr/bin/env python3
"""Aggregate controlled overlap-aware correction safety results."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.metrics.correction_safety_metrics import (
    conservative_rejection_rate,
    overcorrection_rate,
)


SUMMARY_COLUMNS = [
    "variant",
    "overlap",
    "uncertainty_level",
    "num_cases",
    "mean_text_error_before",
    "mean_text_error_after",
    "unsupported_change_count",
    "invented_content_count",
    "forbidden_change_count",
    "speaker_attribution_change_count",
    "needs_review_count",
    "correction_rejected_count",
    "safety_pass_rate",
    "review_flag_accuracy",
    "overcorrection_rate",
    "conservative_rejection_rate",
    "api_used_count",
    "fallback_used_count",
    "notes",
]


def _true_count(rows: list[dict[str, str]], column: str) -> int:
    return sum(row[column].strip().casefold() == "true" for row in rows)


def _mean(rows: list[dict[str, str]], column: str) -> float:
    return round(
        statistics.fmean(float(row[column]) for row in rows),
        6,
    )


def _list_count(rows: list[dict[str, str]], column: str) -> int:
    return sum(len(json.loads(row[column] or "[]")) for row in rows)


def summarize_results(
    input_path: str | Path,
    output_path: str | Path,
) -> list[dict[str, Any]]:
    """Aggregate by variant, overlap flag, and uncertainty level."""

    with Path(input_path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Controlled overlap safety CSV contains no rows.")
    grouped: defaultdict[
        tuple[str, str, str],
        list[dict[str, str]],
    ] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["variant"],
                row["overlap"],
                row["uncertainty_level"],
            )
        ].append(row)

    summaries: list[dict[str, Any]] = []
    for (variant, overlap, uncertainty), group in sorted(grouped.items()):
        count = len(group)
        summaries.append(
            {
                "variant": variant,
                "overlap": overlap,
                "uncertainty_level": uncertainty,
                "num_cases": count,
                "mean_text_error_before": _mean(
                    group, "text_error_before"
                ),
                "mean_text_error_after": _mean(
                    group, "text_error_after"
                ),
                "unsupported_change_count": _list_count(
                    group, "unsupported_changes"
                ),
                "invented_content_count": _true_count(
                    group, "invented_content"
                ),
                "forbidden_change_count": sum(
                    int(row["forbidden_change_count"])
                    for row in group
                ),
                "speaker_attribution_change_count": _true_count(
                    group, "speaker_attribution_changed"
                ),
                "needs_review_count": _true_count(
                    group, "needs_review"
                ),
                "correction_rejected_count": _true_count(
                    group, "correction_rejected"
                ),
                "safety_pass_rate": round(
                    _true_count(group, "safety_pass") / count,
                    6,
                ),
                "review_flag_accuracy": _mean(
                    group, "review_flag_accuracy"
                ),
                "overcorrection_rate": round(
                    overcorrection_rate(
                        row["overcorrection"].strip().casefold() == "true"
                        for row in group
                    ),
                    6,
                ),
                "conservative_rejection_rate": round(
                    conservative_rejection_rate(
                        row["conservative_rejection"].strip().casefold()
                        == "true"
                        for row in group
                    ),
                    6,
                ),
                "api_used_count": _true_count(group, "api_used"),
                "fallback_used_count": _true_count(
                    group, "fallback_used"
                ),
                "notes": (
                    "Controlled overlap text fixtures only; AMI evidence "
                    "is contextual and remains a separate public-data result."
                ),
            }
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
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
    except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError) as exc:
        print(f"Overlap safety summary failed: {exc}")
        return 2
    print(f"Wrote {len(rows)} overlap safety summary rows: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
