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
    "benchmark_scope",
    "language",
    "dataset_name",
    "metric_name",
    "vad_filter",
    "num_clips",
    "total_duration_seconds",
    "mean_error_rate",
    "median_error_rate",
    "cleaned_metric_name",
    "mean_cleaned_error_rate",
    "median_cleaned_error_rate",
    "mean_rtf",
    "median_rtf",
    "mean_runtime_seconds",
    "cold_model_load_seconds",
    "script_normalized",
    "normalization_notes",
    "notes",
]


def summarize_results(
    input_path: str | Path,
    output_path: str | Path,
    *,
    benchmark_scope: str | None = None,
) -> list[dict[str, Any]]:
    """Aggregate results by model, language, and dataset."""

    source = Path(input_path)
    destination = Path(output_path)
    with source.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("ASR benchmark CSV contains no rows.")
    default_scope = benchmark_scope or "; ".join(
        sorted(
            {
                row.get("benchmark_scope", "").strip()
                for row in rows
                if row.get("benchmark_scope", "").strip()
            }
        )
    )
    if not default_scope:
        default_scope = "small-subset formal evaluation"

    grouped: defaultdict[
        tuple[str, str, str, str, str],
        list[dict[str, str]],
    ] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["model_name"],
                row["language"],
                row["dataset_name"],
                row["metric_name"],
                row.get("vad_filter", ""),
            )
        ].append(row)

    summaries: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        (
            model_name,
            language,
            dataset_name,
            metric_name,
            vad_filter,
        ) = key
        error_rates = [float(row["error_rate"]) for row in group]
        cleaned_error_rates = [
            float(row["cleaned_error_rate"])
            for row in group
            if row.get("cleaned_error_rate", "").strip()
        ]
        rtfs = [float(row["rtf"]) for row in group]
        runtimes = [float(row["runtime_seconds"]) for row in group]
        cold_loads = [
            float(row["cold_model_load_seconds"])
            for row in group
            if row.get("cold_model_load_seconds", "").strip()
        ]
        normalization_notes = sorted(
            {
                row.get("normalization_notes", "").strip()
                for row in group
                if row.get("normalization_notes", "").strip()
            }
        )
        script_states = sorted(
            {
                row.get("script_normalized", "").strip()
                for row in group
                if row.get("script_normalized", "").strip()
            }
        )
        scopes = sorted(
            {
                row.get("benchmark_scope", "").strip()
                for row in group
                if row.get("benchmark_scope", "").strip()
            }
        )
        scope = benchmark_scope or "; ".join(scopes) or default_scope
        summaries.append(
            {
                "model_name": model_name,
                "benchmark_scope": scope,
                "language": language,
                "dataset_name": dataset_name,
                "metric_name": metric_name,
                "vad_filter": vad_filter,
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
                "cleaned_metric_name": (
                    group[0].get("cleaned_metric_name", "")
                    if cleaned_error_rates
                    else ""
                ),
                "mean_cleaned_error_rate": (
                    round(statistics.fmean(cleaned_error_rates), 6)
                    if cleaned_error_rates
                    else ""
                ),
                "median_cleaned_error_rate": (
                    round(statistics.median(cleaned_error_rates), 6)
                    if cleaned_error_rates
                    else ""
                ),
                "mean_rtf": round(statistics.fmean(rtfs), 6),
                "median_rtf": round(statistics.median(rtfs), 6),
                "mean_runtime_seconds": round(
                    statistics.fmean(runtimes),
                    6,
                ),
                "cold_model_load_seconds": (
                    round(statistics.fmean(cold_loads), 6)
                    if cold_loads
                    else ""
                ),
                "script_normalized": "; ".join(script_states),
                "normalization_notes": "; ".join(normalization_notes),
                "notes": (
                    f"{scope} aggregate; not full "
                    "dataset performance. RTF excludes model loading."
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
    parser.add_argument(
        "--benchmark-scope",
        help="Override the benchmark scope label in summary rows.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        rows = summarize_results(
            args.input,
            args.output,
            benchmark_scope=args.benchmark_scope,
        )
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"ASR summary failed: {exc}")
        return 2
    print(f"Wrote {len(rows)} ASR summary rows: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
