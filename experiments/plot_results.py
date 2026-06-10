#!/usr/bin/env python3
"""Generate TalkWeaver ablation charts from the result CSV."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
RESULT_PATH = ROOT_DIR / "experiments" / "results" / "ablation_results.csv"
CHART_DIR = ROOT_DIR / "assets" / "result_charts"

CHARTS = (
    (
        "wer",
        "Word Error Rate",
        "wer_comparison.png",
        "#176B87",
    ),
    (
        "speaker_error_or_wder",
        "Simplified Speaker Error / WDER",
        "wder_comparison.png",
        "#2E7D5B",
    ),
    (
        "term_error_rate",
        "Term Error Rate",
        "term_error_comparison.png",
        "#8A5A44",
    ),
    (
        "latency_seconds",
        "Latency (seconds)",
        "latency_comparison.png",
        "#6C5B7B",
    ),
    (
        "hallucinated_corrections",
        "Hallucinated Correction Count",
        "hallucination_comparison.png",
        "#C2413B",
    ),
)


def _is_true(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def load_ablation_rows(path: str | Path = RESULT_PATH) -> list[dict[str, str]]:
    """Load and validate numeric ablation rows."""

    result_path = Path(path)
    if not result_path.exists():
        raise FileNotFoundError(
            "No ablation CSV found. Run "
            "`python experiments/run_ablation.py --mock` first."
        )
    with result_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Ablation CSV contains no rows.")
    return rows


def generate_charts(
    *,
    result_path: str | Path = RESULT_PATH,
    chart_dir: str | Path = CHART_DIR,
) -> list[Path]:
    """Generate all required metric charts and return their paths."""

    import matplotlib.pyplot as plt

    rows = load_ablation_rows(result_path)
    output_dir = Path(chart_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = [row["pipeline"] for row in rows]
    is_mock = all(_is_true(row.get("is_mock")) for row in rows)
    outputs: list[Path] = []

    for metric, ylabel, filename, color in CHARTS:
        try:
            values = [float(row[metric]) for row in rows]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Metric column {metric!r} is missing or non-numeric."
            ) from exc

        figure, axis = plt.subplots(figsize=(10.5, 5.8))
        positions = range(len(labels))
        bars = axis.barh(
            positions,
            values,
            color=color,
            alpha=0.9,
            hatch="//" if is_mock else None,
        )
        axis.set_yticks(list(positions), labels=labels)
        axis.invert_yaxis()
        axis.set_xlabel(ylabel)
        title_suffix = " - Deterministic Mock/Demo" if is_mock else ""
        axis.set_title(f"TalkWeaver {ylabel}{title_suffix}")
        axis.grid(axis="x", alpha=0.2)
        maximum = max(values, default=0.0)
        axis.set_xlim(0, max(0.01, maximum * 1.2))
        label_offset = maximum * 0.015 if maximum > 0 else 0.0002
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_width() + label_offset,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.4f}" if metric != "hallucinated_corrections" else f"{value:.0f}",
                va="center",
                fontsize=9,
            )
        if is_mock:
            figure.text(
                0.5,
                0.01,
                "Demonstration metrics from built-in mock references; not real model performance.",
                ha="center",
                fontsize=9,
                color="#7A2E2A",
            )
        figure.tight_layout(rect=(0, 0.04, 1, 1))
        output = output_dir / filename
        figure.savefig(output, dpi=180, bbox_inches="tight")
        plt.close(figure)
        outputs.append(output)
    return outputs


def main() -> int:
    try:
        outputs = generate_charts()
    except FileNotFoundError as exc:
        print(exc)
        return 0
    except ImportError:
        print("matplotlib is not installed; install requirements.txt to plot.")
        return 0
    for output in outputs:
        print(f"Wrote chart: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
