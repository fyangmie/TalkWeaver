#!/usr/bin/env python3
"""Evaluate EvidenceGate predictions and transparent decision baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

LABELS = ("accept", "reject", "needs_review")


def compute_evidence_gate_metrics(
    true_labels: Iterable[str],
    predicted_labels: Iterable[str],
) -> dict[str, Any]:
    true = list(true_labels)
    predicted = list(predicted_labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        true,
        predicted,
        labels=list(LABELS),
        zero_division=0,
    )
    matrix = confusion_matrix(true, predicted, labels=list(LABELS))
    non_accept_count = sum(label != "accept" for label in true)
    reject_count = sum(label == "reject" for label in true)
    false_accepts = sum(
        actual != "accept" and guess == "accept"
        for actual, guess in zip(true, predicted)
    )
    unsafe_accepts = sum(
        actual == "reject" and guess == "accept"
        for actual, guess in zip(true, predicted)
    )
    per_class = {
        label: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(LABELS)
    }
    return {
        "num_examples": len(true),
        "accuracy": float(accuracy_score(true, predicted)) if true else 0.0,
        "macro_f1": float(sum(f1) / len(LABELS)),
        "false_accept_rate": (
            false_accepts / non_accept_count if non_accept_count else 0.0
        ),
        "unsafe_accept_rate": (
            unsafe_accepts / reject_count if reject_count else 0.0
        ),
        "reject_precision": per_class["reject"]["precision"],
        "reject_recall": per_class["reject"]["recall"],
        "needs_review_recall": per_class["needs_review"]["recall"],
        "accept_precision": per_class["accept"]["precision"],
        "per_class_metrics": json.dumps(per_class, sort_keys=True),
        "confusion_matrix": json.dumps(matrix.tolist()),
    }


def rule_policy_predictions(frame: pd.DataFrame) -> list[str]:
    predictions = []
    for row in frame.to_dict("records"):
        unsafe = (
            float(row.get("correction_rejected_input_flag", 0) or 0) > 0
            or float(row.get("invented_content_flag", 0) or 0) > 0
            or float(row.get("speaker_attribution_changed_flag", 0) or 0) > 0
            or float(row.get("forbidden_change_count", 0) or 0) > 0
            or float(row.get("unsupported_change_count", 0) or 0) > 0
        )
        if unsafe:
            predictions.append("reject")
        elif (
            float(row.get("needs_review_input_flag", 0) or 0) > 0
            or float(row.get("heavy_overlap_flag", 0) or 0) > 0
        ):
            predictions.append("needs_review")
        else:
            predictions.append("accept")
    return predictions


def llm_raw_predictions(frame: pd.DataFrame) -> list[str]:
    predictions = []
    for row in frame.to_dict("records"):
        if float(row.get("llm_variant_flag", 0) or 0) <= 0:
            predictions.append("accept")
        elif float(row.get("correction_rejected_input_flag", 0) or 0) > 0:
            predictions.append("reject")
        elif float(row.get("needs_review_input_flag", 0) or 0) > 0:
            predictions.append("needs_review")
        else:
            predictions.append("accept")
    return predictions


def evaluate_prediction_frame(
    frame: pd.DataFrame,
    *,
    include_baselines: bool = True,
) -> pd.DataFrame:
    required = {"true_label", "predicted_label", "model_name", "split"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Prediction CSV is missing columns: {sorted(missing)}")

    rows: list[dict[str, Any]] = []
    for (model_name, split), group in frame.groupby(["model_name", "split"]):
        metrics = compute_evidence_gate_metrics(
            group["true_label"],
            group["predicted_label"],
        )
        rows.append(
            {
                "model_name": model_name,
                "split": split,
                "is_baseline": False,
                **metrics,
            }
        )

    if include_baselines:
        unique_examples = frame.drop_duplicates(["example_id", "split"])
        for split, group in unique_examples.groupby("split"):
            baseline_predictions = {
                "always_accept": ["accept"] * len(group),
                "always_review": ["needs_review"] * len(group),
                "rule_policy_baseline": rule_policy_predictions(group),
                "llm_variant_raw": llm_raw_predictions(group),
            }
            for name, predictions in baseline_predictions.items():
                rows.append(
                    {
                        "model_name": name,
                        "split": split,
                        "is_baseline": True,
                        **compute_evidence_gate_metrics(
                            group["true_label"],
                            predictions,
                        ),
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["split", "is_baseline", "macro_f1"],
        ascending=[True, True, False],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate EvidenceGate model predictions and decision baselines."
        )
    )
    parser.add_argument(
        "--predictions",
        default=(
            "experiments/results/evidence_gate/"
            "evidence_gate_predictions.csv"
        ),
    )
    parser.add_argument(
        "--output",
        default=(
            "experiments/results/evidence_gate/"
            "evidence_gate_eval_summary.csv"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = Path(args.predictions)
    if not source.exists():
        raise FileNotFoundError(f"Prediction file not found: {source}")
    result = evaluate_prediction_frame(pd.read_csv(source))
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination, index=False)
    test_rows = result[result["split"].eq("test")]
    if not test_rows.empty:
        print(
            test_rows[
                [
                    "model_name",
                    "macro_f1",
                    "false_accept_rate",
                    "unsafe_accept_rate",
                    "needs_review_recall",
                    "reject_recall",
                ]
            ].to_string(index=False)
        )
    print(f"Output: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
