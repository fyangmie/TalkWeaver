#!/usr/bin/env python3
"""Aggregate controlled term recovery and correction safety results."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


MEAN_COLUMNS = [
    "term_precision",
    "term_recall",
    "term_f1",
    "text_error_before",
    "text_error_after",
]
SUMMARY_COLUMNS = [
    "variant",
    "language",
    "difficulty",
    "num_cases",
    *[f"mean_{column}" for column in MEAN_COLUMNS],
    "false_positive_count",
    "missed_term_count",
    "unsupported_change_count",
    "needs_review_count",
    "api_used_count",
    "fallback_used_count",
    "notes",
]


def _mean(rows: list[dict[str, str]], column: str) -> float:
    return round(
        statistics.fmean(float(row[column]) for row in rows),
        6,
    )


def _list_count(rows: list[dict[str, str]], column: str) -> int:
    return sum(len(json.loads(row[column] or "[]")) for row in rows)


def _true_count(rows: list[dict[str, str]], column: str) -> int:
    return sum(row[column].strip().casefold() == "true" for row in rows)


def summarize_results(
    input_path: str | Path,
    output_path: str | Path,
) -> list[dict[str, Any]]:
    """Aggregate by variant, language, and fixture difficulty."""

    with Path(input_path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Controlled term rescue CSV contains no rows.")
    grouped: defaultdict[
        tuple[str, str, str],
        list[dict[str, str]],
    ] = defaultdict(list)
    for row in rows:
        grouped[
            (row["variant"], row["language"], row["difficulty"])
        ].append(row)

    summaries: list[dict[str, Any]] = []
    for (variant, language, difficulty), group in sorted(grouped.items()):
        summary: dict[str, Any] = {
            "variant": variant,
            "language": language,
            "difficulty": difficulty,
            "num_cases": len(group),
        }
        summary.update(
            {
                f"mean_{column}": _mean(group, column)
                for column in MEAN_COLUMNS
            }
        )
        summary.update(
            {
                "false_positive_count": _list_count(
                    group, "false_positive_terms"
                ),
                "missed_term_count": _list_count(group, "missed_terms"),
                "unsupported_change_count": _list_count(
                    group, "unsupported_changes"
                ),
                "needs_review_count": _true_count(group, "needs_review"),
                "api_used_count": _true_count(group, "api_used"),
                "fallback_used_count": _true_count(group, "fallback_used"),
                "notes": (
                    "Controlled text fixtures only; results are separate "
                    "from public-audio evaluation."
                ),
            }
        )
        summaries.append(summary)

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
        print(f"Controlled term summary failed: {exc}")
        return 2
    print(f"Wrote {len(rows)} controlled term summary rows: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
