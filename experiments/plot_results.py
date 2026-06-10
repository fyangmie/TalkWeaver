#!/usr/bin/env python3
"""Plot measured TalkWeaver ablation metrics when available."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
RESULT_PATH = ROOT_DIR / "experiments" / "results" / "ablation_results.csv"
CHART_DIR = ROOT_DIR / "assets" / "result_charts"


def main() -> int:
    if not RESULT_PATH.exists():
        print("No ablation CSV found. Run experiments/run_ablation.py --mock.")
        return 0

    with RESULT_PATH.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    measured = [row for row in rows if row.get("wer", "").strip()]
    if not measured:
        print(
            "Ablation CSV contains no measured metrics. "
            "Charts were not generated from placeholder data."
        )
        return 0

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; install requirements.txt to plot.")
        return 0

    CHART_DIR.mkdir(parents=True, exist_ok=True)
    labels = [row["group"] for row in measured]
    values = [float(row["wer"]) for row in measured]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(labels, values, color="#287271")
    axis.set_title("TalkWeaver WER Comparison")
    axis.set_xlabel("Ablation Group")
    axis.set_ylabel("WER")
    figure.tight_layout()
    output = CHART_DIR / "wer_comparison.png"
    figure.savefig(output, dpi=160)
    plt.close(figure)
    print(f"Wrote chart: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
