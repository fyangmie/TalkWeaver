#!/usr/bin/env python3
"""Train and persist the lightweight TalkWeaver EvidenceGate classifiers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_sample_weight

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.evidence_gate import (
    FEATURE_COLUMNS,
    EvidenceGateModel,
)
from experiments.evaluate_evidence_gate import evaluate_prediction_frame


SUPPORTED_MODELS = (
    "logistic_regression",
    "random_forest",
    "gradient_boosting",
)


def group_train_validation_test_split(
    frame: pd.DataFrame,
    group_column: str,
    *,
    random_seed: int = 42,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split whole template groups into train, validation, and test sets."""

    if group_column not in frame:
        raise ValueError(f"Group split column is missing: {group_column}")
    if frame[group_column].nunique() < 3:
        raise ValueError("Group-aware split requires at least three groups.")
    holdout_fraction = 1.0 - train_fraction
    if holdout_fraction <= 0 or validation_fraction <= 0:
        raise ValueError("Train and validation fractions must leave a test set.")

    first = GroupShuffleSplit(
        n_splits=1,
        train_size=train_fraction,
        random_state=random_seed,
    )
    train_index, holdout_index = next(
        first.split(frame, groups=frame[group_column])
    )
    train = frame.iloc[train_index].copy()
    holdout = frame.iloc[holdout_index].copy()

    relative_validation = validation_fraction / holdout_fraction
    second = GroupShuffleSplit(
        n_splits=1,
        train_size=relative_validation,
        random_state=random_seed + 1,
    )
    validation_index, test_index = next(
        second.split(holdout, groups=holdout[group_column])
    )
    validation = holdout.iloc[validation_index].copy()
    test = holdout.iloc[test_index].copy()

    group_sets = [
        set(part[group_column].astype(str))
        for part in (train, validation, test)
    ]
    if (
        group_sets[0] & group_sets[1]
        or group_sets[0] & group_sets[2]
        or group_sets[1] & group_sets[2]
    ):
        raise RuntimeError("Template-group leakage detected after split.")
    return train, validation, test


def _prediction_rows(
    model: EvidenceGateModel,
    frame: pd.DataFrame,
    split: str,
) -> list[dict]:
    records = frame.to_dict("records")
    predicted = model.predict(records)
    probabilities = model.predict_proba(records)
    rows = []
    metadata_columns = (
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
        "label_reason",
    )
    for source, label, probability in zip(records, predicted, probabilities):
        row = {
            column: source.get(column, "")
            for column in metadata_columns
        }
        row.update(
            {
                "true_label": source["expected_label"],
                "predicted_label": label,
                "model_name": model.model_name,
                "split": split,
                "prob_accept": probability.get("accept", 0.0),
                "prob_reject": probability.get("reject", 0.0),
                "prob_needs_review": probability.get("needs_review", 0.0),
                "probabilities": json.dumps(probability, sort_keys=True),
            }
        )
        row.update(
            {
                feature: float(source.get(feature, 0.0) or 0.0)
                for feature in FEATURE_COLUMNS
            }
        )
        rows.append(row)
    return rows


def train_evidence_gate_models(
    frame: pd.DataFrame,
    output_dir: str | Path,
    models: Iterable[str],
    *,
    group_split_column: str = "template_group",
    random_seed: int = 42,
    models_dir: str | Path | None = None,
) -> dict[str, Path | pd.DataFrame]:
    """Train requested models and write reproducible result artifacts."""

    unknown = set(models) - set(SUPPORTED_MODELS)
    if unknown:
        raise ValueError(f"Unsupported model(s): {sorted(unknown)}")
    missing = set(FEATURE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing feature columns: {sorted(missing)}")
    if set(frame["expected_label"]) - {"accept", "reject", "needs_review"}:
        raise ValueError("Dataset contains unsupported expected_label values.")

    train, validation, test = group_train_validation_test_split(
        frame,
        group_split_column,
        random_seed=random_seed,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model_output = Path(models_dir or (ROOT_DIR / "models" / "evidence_gate"))
    model_output.mkdir(parents=True, exist_ok=True)

    prediction_rows: list[dict] = []
    importance_rows: list[dict] = []
    model_paths: dict[str, Path] = {}
    train_records = train.to_dict("records")
    train_labels = train["expected_label"].astype(str).tolist()
    weights = compute_sample_weight("balanced", train_labels)
    for model_name in models:
        model = EvidenceGateModel(model_name, random_seed=random_seed)
        model.fit(train_records, train_labels, sample_weight=weights)
        model_path = model.save(model_output / f"evidence_gate_{model_name}.joblib")
        model_paths[model_name] = model_path
        prediction_rows.extend(_prediction_rows(model, validation, "validation"))
        prediction_rows.extend(_prediction_rows(model, test, "test"))
        importances = model.feature_importance()
        for rank, (feature, importance) in enumerate(
            sorted(importances.items(), key=lambda item: item[1], reverse=True),
            start=1,
        ):
            importance_rows.append(
                {
                    "model_name": model_name,
                    "feature": feature,
                    "importance": importance,
                    "rank": rank,
                }
            )

    predictions = pd.DataFrame(prediction_rows)
    metrics = evaluate_prediction_frame(predictions, include_baselines=False)
    importance = pd.DataFrame(importance_rows)
    split_summary = pd.DataFrame(
        [
            {
                "split": name,
                "num_examples": len(part),
                "num_template_groups": part[group_split_column].nunique(),
                "accept": int(part["expected_label"].eq("accept").sum()),
                "reject": int(part["expected_label"].eq("reject").sum()),
                "needs_review": int(
                    part["expected_label"].eq("needs_review").sum()
                ),
            }
            for name, part in (
                ("train", train),
                ("validation", validation),
                ("test", test),
            )
        ]
    )

    prediction_path = output / "evidence_gate_predictions.csv"
    metrics_path = output / "evidence_gate_metrics.csv"
    importance_path = output / "evidence_gate_feature_importance.csv"
    split_path = output / "evidence_gate_split_summary.csv"
    predictions.to_csv(prediction_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    importance.to_csv(importance_path, index=False)
    split_summary.to_csv(split_path, index=False)
    return {
        "predictions": predictions,
        "metrics": metrics,
        "feature_importance": importance,
        "split_summary": split_summary,
        "prediction_path": prediction_path,
        "metrics_path": metrics_path,
        "importance_path": importance_path,
        "split_path": split_path,
        "model_paths": model_paths,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train lightweight EvidenceGate correction-safety classifiers "
            "with a template-group-aware split."
        )
    )
    parser.add_argument(
        "--input",
        default=(
            "data/controlled_evidence_gate/"
            "evidence_gate_examples_augmented.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/results/evidence_gate",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=SUPPORTED_MODELS,
        default=list(SUPPORTED_MODELS),
    )
    parser.add_argument("--group-split-column", default="template_group")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--models-dir", default="models/evidence_gate")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = Path(args.input)
    if not source.exists():
        raise FileNotFoundError(f"EvidenceGate dataset not found: {source}")
    result = train_evidence_gate_models(
        pd.read_csv(source),
        args.output_dir,
        args.models,
        group_split_column=args.group_split_column,
        random_seed=args.random_seed,
        models_dir=args.models_dir,
    )
    print(result["split_summary"].to_string(index=False))
    test_metrics = result["metrics"]
    test_metrics = test_metrics[test_metrics["split"].eq("test")]
    print(
        test_metrics[
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
    print(f"Predictions: {result['prediction_path']}")
    print(f"Metrics: {result['metrics_path']}")
    print(f"Feature importance: {result['importance_path']}")
    print(f"Models: {args.models_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
