#!/usr/bin/env python3
"""Generate EvidenceGate evaluation charts with matplotlib only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

LABELS = ("accept", "reject", "needs_review")
COLORS = ("#227C9D", "#D1495B", "#F4A261")


def _test_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    if "split" in metrics:
        test = metrics[metrics["split"].eq("test")].copy()
        if not test.empty:
            return test
    return metrics.copy()


def _save_bar(
    labels: list[str],
    values: list[float],
    destination: Path,
    title: str,
    ylabel: str,
    *,
    color: str = "#227C9D",
    upper: float | None = 1.0,
) -> None:
    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    positions = np.arange(len(labels))
    bars = axis.bar(positions, values, color=color, width=0.62)
    axis.set_xticks(positions, labels, rotation=18, ha="right")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    if upper is not None:
        axis.set_ylim(0, upper)
    axis.grid(axis="y", alpha=0.22)
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (0.02 if upper else 0.0),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    figure.tight_layout()
    figure.savefig(destination, dpi=170)
    plt.close(figure)


def plot_evidence_gate_results(
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    feature_importance: pd.DataFrame,
    output_dir: str | Path,
) -> list[Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    test_metrics = _test_rows(metrics)
    if test_metrics.empty:
        raise ValueError("EvidenceGate metrics contain no rows.")
    best_model = str(
        test_metrics.sort_values(
            ["macro_f1", "unsafe_accept_rate"],
            ascending=[False, True],
        ).iloc[0]["model_name"]
    )

    paths = {
        "macro": output / "evidence_gate_macro_f1.png",
        "false_accept": output / "evidence_gate_false_accept_rate.png",
        "confusion": output / "evidence_gate_confusion_matrix.png",
        "importance": output / "evidence_gate_feature_importance.png",
        "recall": output / "evidence_gate_class_recall.png",
    }
    model_labels = test_metrics["model_name"].astype(str).tolist()
    _save_bar(
        model_labels,
        test_metrics["macro_f1"].astype(float).tolist(),
        paths["macro"],
        "EvidenceGate test macro F1",
        "Macro F1",
    )
    _save_bar(
        model_labels,
        test_metrics["false_accept_rate"].astype(float).tolist(),
        paths["false_accept"],
        "EvidenceGate false-accept rate",
        "False-accept rate",
        color="#D1495B",
    )

    best_predictions = predictions[
        predictions["model_name"].eq(best_model)
        & predictions["split"].eq("test")
    ]
    matrix = confusion_matrix(
        best_predictions["true_label"],
        best_predictions["predicted_label"],
        labels=list(LABELS),
    )
    figure, axis = plt.subplots(figsize=(6.2, 5.2))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(len(LABELS)), LABELS, rotation=18, ha="right")
    axis.set_yticks(range(len(LABELS)), LABELS)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title(f"EvidenceGate confusion matrix: {best_model}")
    for row in range(len(LABELS)):
        for column in range(len(LABELS)):
            axis.text(column, row, int(matrix[row, column]), ha="center", va="center")
    figure.colorbar(image, ax=axis, fraction=0.046)
    figure.tight_layout()
    figure.savefig(paths["confusion"], dpi=170)
    plt.close(figure)

    best_importance = feature_importance[
        feature_importance["model_name"].eq(best_model)
    ].nlargest(12, "importance")
    figure, axis = plt.subplots(figsize=(8.5, 5.6))
    ordered = best_importance.sort_values("importance")
    axis.barh(ordered["feature"], ordered["importance"], color="#227C9D")
    axis.set_title(f"EvidenceGate feature importance: {best_model}")
    axis.set_xlabel("Model importance")
    axis.grid(axis="x", alpha=0.22)
    figure.tight_layout()
    figure.savefig(paths["importance"], dpi=170)
    plt.close(figure)

    recall_rows = []
    for row in test_metrics.to_dict("records"):
        per_class = json.loads(row["per_class_metrics"])
        for label in LABELS:
            recall_rows.append(
                {
                    "model_name": row["model_name"],
                    "label": label,
                    "recall": float(per_class[label]["recall"]),
                }
            )
    recall = pd.DataFrame(recall_rows)
    figure, axis = plt.subplots(figsize=(9.2, 5.0))
    positions = np.arange(len(model_labels))
    width = 0.24
    for index, (label, color) in enumerate(zip(LABELS, COLORS)):
        values = [
            float(
                recall[
                    recall["model_name"].eq(model) & recall["label"].eq(label)
                ]["recall"].iloc[0]
            )
            for model in model_labels
        ]
        axis.bar(
            positions + (index - 1) * width,
            values,
            width=width,
            label=label,
            color=color,
        )
    axis.set_xticks(positions, model_labels, rotation=18, ha="right")
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Recall")
    axis.set_title("EvidenceGate class recall")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.22)
    figure.tight_layout()
    figure.savefig(paths["recall"], dpi=170)
    plt.close(figure)
    return list(paths.values())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot EvidenceGate model metrics and audit behavior."
    )
    parser.add_argument(
        "--metrics",
        default=(
            "experiments/results/evidence_gate/"
            "evidence_gate_metrics.csv"
        ),
    )
    parser.add_argument(
        "--predictions",
        default=(
            "experiments/results/evidence_gate/"
            "evidence_gate_predictions.csv"
        ),
    )
    parser.add_argument(
        "--feature-importance",
        default=(
            "experiments/results/evidence_gate/"
            "evidence_gate_feature_importance.csv"
        ),
    )
    parser.add_argument("--output-dir", default="assets/result_charts")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = plot_evidence_gate_results(
        pd.read_csv(args.metrics),
        pd.read_csv(args.predictions),
        pd.read_csv(args.feature_importance),
        args.output_dir,
    )
    for path in paths:
        print(f"Created: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
