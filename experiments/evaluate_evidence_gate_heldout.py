#!/usr/bin/env python3
"""Evaluate all EvidenceGate feature sets on grouped and independent tests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.evidence_gate import (
    AUDIT_AWARE_FEATURES,
    FEATURE_SETS,
    EvidenceGateModel,
)
from experiments.evaluate_evidence_gate import evaluate_prediction_frame
from experiments.train_evidence_gate import SUPPORTED_MODELS


def _heldout_prediction_rows(
    model: EvidenceGateModel,
    frame: pd.DataFrame,
    feature_set: str,
) -> list[dict]:
    records = frame.to_dict("records")
    labels = model.predict(records)
    probabilities = model.predict_proba(records)
    rows = []
    for source, label, probability in zip(records, labels, probabilities):
        row = {
            column: source.get(column, "")
            for column in (
                "example_id",
                "source_experiment",
                "case_id",
                "variant",
                "template_group",
                "is_augmented",
                "raw_text",
                "corrected_text",
                "reference_text",
                "language",
                "notes",
            )
        }
        row.update(
            {
                "true_label": source["expected_label"],
                "predicted_label": label,
                "model_name": model.model_name,
                "feature_set": feature_set,
                "split": "independent_heldout",
                "prob_accept": probability.get("accept", 0.0),
                "prob_reject": probability.get("reject", 0.0),
                "prob_needs_review": probability.get("needs_review", 0.0),
                "probabilities": json.dumps(probability, sort_keys=True),
            }
        )
        row.update(
            {
                feature: float(source.get(feature, 0.0) or 0.0)
                for feature in AUDIT_AWARE_FEATURES
            }
        )
        rows.append(row)
    return rows


def evaluate_all_feature_sets(
    heldout: pd.DataFrame,
    results_dir: str | Path,
    models_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    results = Path(results_dir)
    model_root = Path(models_dir)
    prediction_frames: list[pd.DataFrame] = []

    for feature_set in FEATURE_SETS:
        grouped_path = (
            results / f"evidence_gate_{feature_set}_predictions.csv"
        )
        if not grouped_path.exists():
            raise FileNotFoundError(
                f"Grouped prediction file is missing for {feature_set}: "
                f"{grouped_path}"
            )
        grouped = pd.read_csv(grouped_path)
        grouped = grouped[grouped["split"].eq("test")].copy()
        grouped["split"] = "grouped_test"
        prediction_frames.append(grouped)

        for model_name in SUPPORTED_MODELS:
            model_path = (
                model_root
                / f"evidence_gate_{feature_set}_{model_name}.joblib"
            )
            if not model_path.exists():
                raise FileNotFoundError(f"Model is missing: {model_path}")
            model = EvidenceGateModel.load(model_path)
            prediction_frames.append(
                pd.DataFrame(
                    _heldout_prediction_rows(model, heldout, feature_set)
                )
            )

    predictions = pd.concat(prediction_frames, ignore_index=True, sort=False)
    metrics = evaluate_prediction_frame(predictions, include_baselines=True)
    metrics["claim_level"] = metrics["feature_set"].map(
        {
            "audit_aware": "policy_distillation_sanity_check",
            "evidence_only": "strict_controlled_validation",
            "risk_only": "strict_controlled_validation",
        }
    ).fillna("baseline")
    return predictions, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate audit-aware, evidence-only, and risk-only EvidenceGate "
            "models on grouped test rows and independent manual proposals."
        )
    )
    parser.add_argument(
        "--heldout",
        default=(
            "data/controlled_evidence_gate/"
            "evidence_gate_independent_heldout.csv"
        ),
    )
    parser.add_argument(
        "--results-dir",
        default="experiments/results/evidence_gate",
    )
    parser.add_argument("--models-dir", default="models/evidence_gate")
    parser.add_argument(
        "--predictions-output",
        default=(
            "experiments/results/evidence_gate/"
            "evidence_gate_validation_predictions.csv"
        ),
    )
    parser.add_argument(
        "--output",
        default=(
            "experiments/results/evidence_gate/"
            "evidence_gate_validation_metrics.csv"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    heldout_path = Path(args.heldout)
    if not heldout_path.exists():
        raise FileNotFoundError(
            f"Independent heldout is missing: {heldout_path}"
        )
    heldout = pd.read_csv(heldout_path)
    predictions, metrics = evaluate_all_feature_sets(
        heldout,
        args.results_dir,
        args.models_dir,
    )
    prediction_path = Path(args.predictions_output)
    metric_path = Path(args.output)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    metric_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(prediction_path, index=False)
    metrics.to_csv(metric_path, index=False)
    heldout_metrics = metrics[
        metrics["split"].eq("independent_heldout")
        & ~metrics["is_baseline"].astype(bool)
    ]
    print(
        heldout_metrics[
            [
                "feature_set",
                "model_name",
                "macro_f1",
                "unsafe_accept_rate",
                "needs_review_recall",
                "reject_recall",
            ]
        ].to_string(index=False)
    )
    print(f"Predictions: {prediction_path}")
    print(f"Metrics: {metric_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
