#!/usr/bin/env python3
"""Plot EvidenceGate leakage and strict validation results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

LABELS = ("accept", "reject", "needs_review")
FEATURE_SET_COLORS = {
    "audit_aware": "#7A5195",
    "evidence_only": "#227C9D",
    "risk_only": "#F4A261",
}


def _model_metrics(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    return frame[
        frame["split"].eq(split) & ~frame["is_baseline"].astype(bool)
    ].copy()


def plot_evidence_gate_validation(
    leakage_audit: pd.DataFrame,
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    output_dir: str | Path,
) -> list[Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "leakage": output / "evidence_gate_feature_leakage_audit.png",
        "macro": output / "evidence_gate_feature_set_macro_f1.png",
        "unsafe": output / "evidence_gate_feature_set_unsafe_accept.png",
        "confusion": output / "evidence_gate_heldout_confusion_matrix.png",
    }

    category_order = (
        "allowed_pre_decision_features",
        "risky_reference_derived_features",
        "direct_label_proxy_features",
        "final_audit_outcome_features",
    )
    category_counts = (
        leakage_audit["category"].value_counts().reindex(category_order).fillna(0)
    )
    figure, axis = plt.subplots(figsize=(9.2, 4.8))
    bars = axis.bar(
        ["Pre-decision", "Reference-derived", "Direct proxy", "Final outcome"],
        category_counts.values,
        color=["#227C9D", "#F4A261", "#D1495B", "#8C2F39"],
    )
    axis.set_title("EvidenceGate feature leakage audit")
    axis.set_ylabel("Fields / features")
    axis.grid(axis="y", alpha=0.22)
    for bar, value in zip(bars, category_counts.values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.15,
            str(int(value)),
            ha="center",
        )
    figure.tight_layout()
    figure.savefig(paths["leakage"], dpi=170)
    plt.close(figure)

    models = sorted(metrics.loc[~metrics["is_baseline"].astype(bool), "model_name"].unique())
    feature_sets = ("audit_aware", "evidence_only", "risk_only")
    scopes = ("grouped_test", "independent_heldout")
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), sharey=True)
    for axis, scope in zip(axes, scopes):
        scoped = _model_metrics(metrics, scope)
        positions = np.arange(len(models))
        width = 0.24
        for index, feature_set in enumerate(feature_sets):
            values = []
            for model in models:
                row = scoped[
                    scoped["feature_set"].eq(feature_set)
                    & scoped["model_name"].eq(model)
                ]
                values.append(float(row["macro_f1"].iloc[0]) if not row.empty else 0.0)
            axis.bar(
                positions + (index - 1) * width,
                values,
                width,
                label=feature_set,
                color=FEATURE_SET_COLORS[feature_set],
            )
        axis.set_xticks(positions, models, rotation=18, ha="right")
        axis.set_ylim(0, 1.05)
        axis.set_title(scope.replace("_", " ").title())
        axis.grid(axis="y", alpha=0.22)
    axes[0].set_ylabel("Macro F1")
    axes[1].legend(frameon=False)
    figure.suptitle("EvidenceGate macro F1 by feature set")
    figure.tight_layout()
    figure.savefig(paths["macro"], dpi=170)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), sharey=True)
    for axis, scope in zip(axes, scopes):
        scoped = _model_metrics(metrics, scope)
        positions = np.arange(len(models))
        width = 0.24
        for index, feature_set in enumerate(feature_sets):
            values = []
            for model in models:
                row = scoped[
                    scoped["feature_set"].eq(feature_set)
                    & scoped["model_name"].eq(model)
                ]
                values.append(
                    float(row["unsafe_accept_rate"].iloc[0])
                    if not row.empty
                    else 0.0
                )
            axis.bar(
                positions + (index - 1) * width,
                values,
                width,
                label=feature_set,
                color=FEATURE_SET_COLORS[feature_set],
            )
        axis.set_xticks(positions, models, rotation=18, ha="right")
        axis.set_ylim(0, 1.05)
        axis.set_title(scope.replace("_", " ").title())
        axis.grid(axis="y", alpha=0.22)
    axes[0].set_ylabel("Unsafe-accept rate")
    axes[1].legend(frameon=False)
    figure.suptitle("EvidenceGate unsafe accepts by feature set")
    figure.tight_layout()
    figure.savefig(paths["unsafe"], dpi=170)
    plt.close(figure)

    heldout = _model_metrics(metrics, "independent_heldout")
    strict = heldout[heldout["feature_set"].isin(["evidence_only", "risk_only"])]
    best = strict.sort_values(
        ["macro_f1", "unsafe_accept_rate", "model_name"],
        ascending=[False, True, True],
    ).iloc[0]
    selected = predictions[
        predictions["split"].eq("independent_heldout")
        & predictions["feature_set"].eq(best["feature_set"])
        & predictions["model_name"].eq(best["model_name"])
    ]
    matrix = confusion_matrix(
        selected["true_label"],
        selected["predicted_label"],
        labels=list(LABELS),
    )
    figure, axis = plt.subplots(figsize=(6.4, 5.4))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(3), LABELS, rotation=18, ha="right")
    axis.set_yticks(range(3), LABELS)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title(
        f"Independent heldout: {best['feature_set']} / {best['model_name']}"
    )
    for row in range(3):
        for column in range(3):
            axis.text(column, row, int(matrix[row, column]), ha="center", va="center")
    figure.colorbar(image, ax=axis, fraction=0.046)
    figure.tight_layout()
    figure.savefig(paths["confusion"], dpi=170)
    plt.close(figure)
    return list(paths.values())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot EvidenceGate leakage and independent validation."
    )
    parser.add_argument(
        "--leakage-audit",
        default=(
            "experiments/results/evidence_gate/"
            "evidence_gate_feature_leakage_audit.csv"
        ),
    )
    parser.add_argument(
        "--metrics",
        default=(
            "experiments/results/evidence_gate/"
            "evidence_gate_validation_metrics.csv"
        ),
    )
    parser.add_argument(
        "--predictions",
        default=(
            "experiments/results/evidence_gate/"
            "evidence_gate_validation_predictions.csv"
        ),
    )
    parser.add_argument("--output-dir", default="assets/result_charts")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = plot_evidence_gate_validation(
        pd.read_csv(args.leakage_audit),
        pd.read_csv(args.metrics),
        pd.read_csv(args.predictions),
        args.output_dir,
    )
    for path in paths:
        print(f"Created: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
