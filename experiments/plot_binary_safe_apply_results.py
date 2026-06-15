#!/usr/bin/env python3
"""Plot binary safe-to-apply correction benchmark results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "always_apply": "#A8A8A8",
    "never_apply": "#6C8EAD",
    "retrieval_only": "#E9C46A",
    "overlap_unaware_policy": "#F4A261",
    "binary_eccogate": "#D1495B",
    "llm_self_judge_no_evidence": "#7A5195",
    "llm_self_judge_with_evidence": "#2A9D8F",
}


def _bar(
    frame: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    path: Path,
    *,
    fixed_unit_range: bool = True,
) -> None:
    ordered = frame.sort_values(
        metric,
        ascending=metric in {"unsafe_apply_rate", "false_block_rate"},
    )
    values = pd.to_numeric(ordered[metric], errors="coerce").fillna(0.0)
    figure, axis = plt.subplots(figsize=(11.5, 5.5))
    bars = axis.bar(
        ordered["method"],
        values,
        color=[COLORS.get(method, "#5B6C8F") for method in ordered["method"]],
    )
    if fixed_unit_range:
        axis.set_ylim(0, max(1.05, float(values.max()) * 1.15))
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.tick_params(axis="x", rotation=23)
    axis.grid(axis="y", alpha=0.22)
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(0.02, float(values.max()) * 0.03),
            f"{value:.3f}",
            ha="center",
            fontsize=8,
        )
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def plot_binary_results(
    summary: pd.DataFrame,
    results: pd.DataFrame,
    output_dir: str | Path,
) -> list[Path]:
    """Generate five binary benchmark charts."""

    required_summary = {
        "method",
        "macro_f1",
        "unsafe_apply_rate",
        "false_block_rate",
        "error_delta_when_applied",
    }
    required_results = {"method", "category", "gold_label", "unsafe_apply"}
    if missing := required_summary - set(summary.columns):
        raise ValueError(f"Summary is missing columns: {sorted(missing)}")
    if missing := required_results - set(results.columns):
        raise ValueError(f"Results are missing columns: {sorted(missing)}")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = [
        output / "binary_safe_apply_macro_f1.png",
        output / "binary_unsafe_apply_rate.png",
        output / "binary_false_block_rate.png",
        output / "binary_error_delta_when_applied.png",
        output / "binary_category_failure.png",
    ]
    _bar(
        summary,
        "macro_f1",
        "Binary safe-to-apply macro F1",
        "Macro F1",
        paths[0],
    )
    _bar(
        summary,
        "unsafe_apply_rate",
        "Unsafe application rate (lower is safer)",
        "Unsafe-apply rate",
        paths[1],
    )
    _bar(
        summary,
        "false_block_rate",
        "False blocking of beneficial corrections",
        "False-block rate",
        paths[2],
    )
    _bar(
        summary,
        "error_delta_when_applied",
        "Reference error reduction among applied corrections",
        "Mean error_before - error_after",
        paths[3],
        fixed_unit_range=False,
    )

    categories = sorted(results["category"].dropna().unique())
    methods = list(summary["method"])
    matrix = np.zeros((len(methods), len(categories)))
    for row_index, method in enumerate(methods):
        scoped = results[
            results["method"].eq(method)
            & results["gold_label"].eq("do_not_apply")
        ]
        for column_index, category in enumerate(categories):
            category_rows = scoped[scoped["category"].eq(category)]
            if category_rows.empty:
                continue
            unsafe = category_rows["unsafe_apply"].map(
                lambda value: str(value).casefold() in {"true", "1"}
            )
            matrix[row_index, column_index] = float(unsafe.mean())
    figure, axis = plt.subplots(figsize=(15.0, 6.0))
    image = axis.imshow(matrix, cmap="Reds", vmin=0, vmax=1, aspect="auto")
    axis.set_xticks(range(len(categories)), categories, rotation=32, ha="right")
    axis.set_yticks(range(len(methods)), methods)
    axis.set_title("Unsafe application by proposal category")
    for row in range(len(methods)):
        for column in range(len(categories)):
            axis.text(
                column,
                row,
                f"{matrix[row, column]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if matrix[row, column] > 0.55 else "black",
            )
    figure.colorbar(image, ax=axis, fraction=0.025)
    figure.tight_layout()
    figure.savefig(paths[4], dpi=170)
    plt.close(figure)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the binary safe-to-apply benchmark."
    )
    parser.add_argument(
        "--summary",
        default=(
            "experiments/results/binary_safe_apply/"
            "binary_safe_apply_summary.csv"
        ),
    )
    parser.add_argument(
        "--results",
        default=(
            "experiments/results/binary_safe_apply/"
            "binary_safe_apply_results.csv"
        ),
    )
    parser.add_argument("--output-dir", default="assets/result_charts")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = plot_binary_results(
        pd.read_csv(args.summary),
        pd.read_csv(args.results),
        args.output_dir,
    )
    for path in paths:
        print(f"Created: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
