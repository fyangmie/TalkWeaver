#!/usr/bin/env python3
"""Plot controlled overlap-aware correction safety results."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


VARIANT_ORDER = [
    "no_overlap_awareness_rule",
    "overlap_aware_rule",
    "no_overlap_awareness_llm",
    "overlap_aware_llm",
]


def _load(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Controlled overlap safety CSV contains no rows.")
    return rows


def _variants(rows: list[dict[str, str]]) -> list[str]:
    return [
        variant
        for variant in VARIANT_ORDER
        if any(row["variant"] == variant for row in rows)
    ]


def _mean_by_variant(
    rows: list[dict[str, str]],
    column: str,
) -> dict[str, float]:
    grouped: defaultdict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["variant"]].append(float(row[column]))
    return {
        variant: sum(values) / len(values)
        for variant, values in grouped.items()
    }


def _true_count_by_variant(
    rows: list[dict[str, str]],
    column: str,
) -> dict[str, int]:
    counts: defaultdict[str, int] = defaultdict(int)
    for row in rows:
        counts[row["variant"]] += (
            row[column].strip().casefold() == "true"
        )
    return dict(counts)


def _true_rate_by_variant(
    rows: list[dict[str, str]],
    column: str,
) -> dict[str, float]:
    totals: defaultdict[str, int] = defaultdict(int)
    positives: defaultdict[str, int] = defaultdict(int)
    for row in rows:
        totals[row["variant"]] += 1
        positives[row["variant"]] += (
            row[column].strip().casefold() == "true"
        )
    return {
        variant: positives[variant] / total
        for variant, total in totals.items()
        if total
    }


def _list_count_by_variant(
    rows: list[dict[str, str]],
    column: str,
) -> dict[str, int]:
    counts: defaultdict[str, int] = defaultdict(int)
    for row in rows:
        counts[row["variant"]] += len(json.loads(row[column] or "[]"))
    return dict(counts)


def _labels(variants: list[str]) -> list[str]:
    return [
        variant.replace("_awareness_", " awareness\n").replace("_", " ")
        for variant in variants
    ]


def _bar_chart(
    *,
    variants: list[str],
    values: list[float],
    colors: list[str],
    ylabel: str,
    title: str,
    output: Path,
    ylim: tuple[float, float] | None = None,
) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(10.8, 6.2))
    axis.bar(variants, values, color=colors[: len(variants)])
    axis.set_xticks(
        range(len(variants)),
        labels=_labels(variants),
        rotation=15,
        ha="right",
    )
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    if ylim is not None:
        axis.set_ylim(*ylim)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _grouped_bar_chart(
    *,
    variants: list[str],
    first_values: list[float],
    second_values: list[float],
    first_label: str,
    second_label: str,
    ylabel: str,
    title: str,
    output: Path,
) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(10.8, 6.2))
    positions = list(range(len(variants)))
    width = 0.34
    axis.bar(
        [position - width / 2 for position in positions],
        first_values,
        width=width,
        label=first_label,
        color="#C75D3A",
    )
    axis.bar(
        [position + width / 2 for position in positions],
        second_values,
        width=width,
        label=second_label,
        color="#4464AD",
    )
    axis.set_xticks(
        positions,
        labels=_labels(variants),
        rotation=15,
        ha="right",
    )
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_results(
    input_path: str | Path,
    output_dir: str | Path,
) -> list[Path]:
    """Generate four video-readable overlap correction safety charts."""

    rows = _load(input_path)
    variants = _variants(rows)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    colors = ["#C75D3A", "#2A9D8F", "#8E6C88", "#4464AD"]

    safety = _true_rate_by_variant(rows, "safety_pass")
    safety_path = destination / "overlap_safety_pass_rate.png"
    _bar_chart(
        variants=variants,
        values=[safety.get(variant, 0.0) for variant in variants],
        colors=colors,
        ylabel="Safety pass rate",
        title="Overlap Evidence Improves Correction Safety",
        output=safety_path,
        ylim=(0.0, 1.05),
    )

    unsupported = _list_count_by_variant(rows, "unsupported_changes")
    invented = _true_count_by_variant(rows, "invented_content")
    unsafe_counts = [
        unsupported.get(variant, 0) + invented.get(variant, 0)
        for variant in variants
    ]
    rejected = _true_count_by_variant(rows, "correction_rejected")
    unsupported_path = destination / "overlap_unsupported_changes.png"
    _grouped_bar_chart(
        variants=variants,
        first_values=unsafe_counts,
        second_values=[
            rejected.get(variant, 0) for variant in variants
        ],
        first_label="Accepted unsupported/invented outputs",
        second_label="Rejected corrections",
        ylabel="Case or token count",
        title="Hallucination Watchdog: Accepted vs Rejected Risk",
        output=unsupported_path,
    )

    reviews = _true_count_by_variant(rows, "needs_review")
    review_path = destination / "overlap_review_flags.png"
    _bar_chart(
        variants=variants,
        values=[reviews.get(variant, 0) for variant in variants],
        colors=colors,
        ylabel="Cases marked needs review",
        title="Review Flags Exposed by Overlap-Aware Correction",
        output=review_path,
    )

    before = _mean_by_variant(rows, "text_error_before")
    after = _mean_by_variant(rows, "text_error_after")
    delta_path = destination / "overlap_error_delta.png"
    _bar_chart(
        variants=variants,
        values=[
            before.get(variant, 0.0) - after.get(variant, 0.0)
            for variant in variants
        ],
        colors=colors,
        ylabel="Mean error-rate reduction",
        title="Controlled Accuracy and Safety Trade-off",
        output=delta_path,
    )
    return [
        safety_path,
        unsupported_path,
        review_path,
        delta_path,
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        outputs = plot_results(args.input, args.output_dir)
    except (FileNotFoundError, ImportError, KeyError, ValueError, json.JSONDecodeError) as exc:
        print(f"Overlap safety plotting failed: {exc}")
        return 2
    for output in outputs:
        print(f"Wrote chart: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
