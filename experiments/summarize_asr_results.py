#!/usr/bin/env python3
"""Aggregate per-clip real ASR benchmark results."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


SUMMARY_COLUMNS = [
    "model_name",
    "language",
    "dataset_name",
    "metric_name",
    "num_clips",
    "total_duration_seconds",
    "mean_error_rate",
    "median_error_rate",
    "mean_rtf",
    "median_rtf",
    "mean_runtime_seconds",
    "notes",
]


def summarize_results(
    input_path: str | Path,
    output_path: str | Path,
) -> list[dict[str, Any]]:
    """Aggregate results by model, language, and dataset."""

    source = Path(input_path)
    destination = Path(output_path)
    with source.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("ASR benchmark CSV contains no rows.")

    grouped: defaultdict[
        tuple[str, str, str, str],
        list[dict[str, str]],
    ] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["model_name"],
                row["language"],
                row["dataset_name"],
                row["metric_name"],
            )
        ].append(row)

    summaries: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        model_name, language, dataset_name, metric_name = key
        error_rates = [float(row["error_rate"]) for row in group]
        rtfs = [float(row["rtf"]) for row in group]
        runtimes = [float(row["runtime_seconds"]) for row in group]
        summaries.append(
            {
                "model_name": model_name,
                "language": language,
                "dataset_name": dataset_name,
                "metric_name": metric_name,
                "num_clips": len(group),
                "total_duration_seconds": round(
                    sum(float(row["duration_seconds"]) for row in group),
                    6,
                ),
                "mean_error_rate": round(statistics.fmean(error_rates), 6),
                "median_error_rate": round(
                    statistics.median(error_rates),
                    6,
                ),
                "mean_rtf": round(statistics.fmean(rtfs), 6),
                "median_rtf": round(statistics.median(rtfs), 6),
                "mean_runtime_seconds": round(
                    statistics.fmean(runtimes),
                    6,
                ),
                "notes": (
                    "Small-subset formal evaluation aggregate; not full "
                    "dataset performance."
                ),
            }
        )

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
        print(f"ASR summary failed: {exc}")
        return 2
    print(f"Wrote {len(rows)} ASR summary rows: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
