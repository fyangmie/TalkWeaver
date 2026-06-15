#!/usr/bin/env python3
"""Plot selective-correction feasibility pilot results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "always_accept": "#A8A8A8",
    "always_review": "#6C8EAD",
    "EccoGate": "#D1495B",
    "llm_self_judge_no_evidence": "#F4A261",
    "llm_self_judge_with_evidence": "#2A9D8F",
}


def _bar_chart(
    summary: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    path: Path,
) -> None:
    ordered = summary.sort_values(metric, ascending=(metric == "unsafe_accept_rate"))
    figure, axis = plt.subplots(figsize=(10.5, 5.3))
    bars = axis.bar(
        ordered["method"],
        ordered[metric].astype(float),
        color=[COLORS.get(method, "#5B6C8F") for method in ordered["method"]],
    )
    axis.set_ylim(0, 1.05)
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.tick_params(axis="x", rotation=20)
    axis.grid(axis="y", alpha=0.22)
    for bar, value in zip(bars, ordered[metric].astype(float)):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.025,
            f"{value:.2f}",
            ha="center",
            fontsize=9,
        )
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def plot_pilot_results(
    summary: pd.DataFrame,
    results: pd.DataFrame,
    output_dir: str | Path,
) -> list[Path]:
    """Create four report-ready pilot charts."""

    required_summary = {
        "method",
        "macro_f1",
        "unsafe_accept_rate",
        "needs_review_recall",
    }
    required_results = {"method", "category", "gold_label", "unsafe_accept"}
    if missing := required_summary - set(summary.columns):
        raise ValueError(f"Summary is missing columns: {sorted(missing)}")
    if missing := required_results - set(results.columns):
        raise ValueError(f"Results are missing columns: {sorted(missing)}")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = [
        output / "pilot_unsafe_accept_rate.png",
        output / "pilot_needs_review_recall.png",
        output / "pilot_macro_f1.png",
        output / "pilot_category_failure.png",
    ]
    _bar_chart(
        summary,
        "unsafe_accept_rate",
        "Pilot unsafe-accept rate (lower is safer)",
        "Unsafe-accept rate",
        paths[0],
    )
    _bar_chart(
        summary,
        "needs_review_recall",
        "Pilot abstention recall on review-needed proposals",
        "Needs-review recall",
        paths[1],
    )
    _bar_chart(
        summary,
        "macro_f1",
        "Pilot three-way decision macro F1",
        "Macro F1",
        paths[2],
    )

    categories = sorted(results["category"].dropna().unique())
    methods = list(summary["method"])
    matrix = np.zeros((len(methods), len(categories)))
    for row_index, method in enumerate(methods):
        scoped = results[
            results["method"].eq(method)
            & ~results["gold_label"].eq("accept")
        ]
        for column_index, category in enumerate(categories):
            category_rows = scoped[scoped["category"].eq(category)]
            if category_rows.empty:
                continue
            values = category_rows["unsafe_accept"].map(
                lambda value: str(value).casefold() in {"true", "1"}
            )
            matrix[row_index, column_index] = float(values.mean())
    figure, axis = plt.subplots(figsize=(13.5, 5.8))
    image = axis.imshow(matrix, cmap="Reds", vmin=0, vmax=1, aspect="auto")
    axis.set_xticks(range(len(categories)), categories, rotation=30, ha="right")
    axis.set_yticks(range(len(methods)), methods)
    axis.set_title("Unsafe accepts by hard-case category")
    for row in range(len(methods)):
        for column in range(len(categories)):
            axis.text(
                column,
                row,
                f"{matrix[row, column]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if matrix[row, column] > 0.55 else "black",
            )
    figure.colorbar(image, ax=axis, fraction=0.03, label="Unsafe-accept rate")
    figure.tight_layout()
    figure.savefig(paths[3], dpi=170)
    plt.close(figure)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the selective-correction feasibility pilot."
    )
    parser.add_argument(
        "--summary",
        default=(
            "experiments/results/pilot/"
            "selective_correction_pilot_summary.csv"
        ),
    )
    parser.add_argument(
        "--results",
        default=(
            "experiments/results/pilot/"
            "selective_correction_pilot_results.csv"
        ),
    )
    parser.add_argument("--output-dir", default="assets/result_charts")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = plot_pilot_results(
        pd.read_csv(args.summary),
        pd.read_csv(args.results),
        args.output_dir,
    )
    for path in paths:
        print(f"Created: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
