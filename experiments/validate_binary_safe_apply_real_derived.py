#!/usr/bin/env python3
"""
Validate the real-derived safe-to-apply dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = [
    "proposal_id",
    "source",
    "dataset",
    "sample_id",
    "language",
    "audio_path",
    "asr_model",
    "correction_source",
    "category",
    "raw_asr_text",
    "proposed_corrected_text",
    "reference_text",
    "context",
    "retrieved_terms",
    "overlap_flag",
    "heavy_overlap_flag",
    "speaker_ambiguity_flag",
    "partial_utterance_flag",
    "error_before",
    "error_after",
    "error_delta",
    "binary_label",
    "label_source",
    "label_rule",
    "needs_human_check",
    "notes",
]


def validate_dataset(input_path: str) -> bool:
    print("\n" + "=" * 72)
    print("Validating Real-Derived Safe-to-Apply Dataset")
    print("=" * 72 + "\n")

    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows from {input_path}\n")

    if df.empty:
        raise ValueError("Dataset is empty")

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    print("All required columns present")

    invalid_sources = df[~df["source"].astype(str).str.startswith("real_", na=False)]
    if len(invalid_sources) > 0:
        raise ValueError(
            f"All sources must start with real_. Invalid sources: "
            f"{invalid_sources['source'].unique().tolist()}"
        )
    print("All source values start with real_")

    forbidden_sources = {"controlled_stress_test", "heldout_controlled", "controlled_reference"}
    forbidden = df[df["source"].isin(forbidden_sources)]
    if len(forbidden) > 0:
        raise ValueError(f"Found forbidden controlled source values: {forbidden['source'].unique().tolist()}")
    print("No controlled/heldout source values found")

    fallback_rows = df[df["correction_source"] == "no_correction_fallback"]
    if len(fallback_rows) > 0:
        raise ValueError(
            f"Found {len(fallback_rows)} rows with no_correction_fallback. "
            "Primary real-derived evaluation must use system-generated correction text, "
            "not raw-text fallback."
        )
    print("No no_correction_fallback rows found")

    oracle_rows = df[df["correction_source"].astype(str).str.contains("oracle", case=False, na=False)]
    if len(oracle_rows) > 0:
        raise ValueError("Oracle correction_source rows are not allowed in primary real-derived evaluation")
    print("No oracle correction_source rows found")

    for col in ["raw_asr_text", "proposed_corrected_text", "reference_text"]:
        empty = df[df[col].isna() | (df[col].astype(str).str.strip() == "")]
        if len(empty) > 0:
            raise ValueError(f"Found {len(empty)} rows with empty {col}")
    print("No empty raw/proposed/reference text fields")

    valid_labels = {"safe_to_apply", "do_not_apply"}
    invalid_labels = df[~df["binary_label"].isin(valid_labels)]
    if len(invalid_labels) > 0:
        raise ValueError(f"Invalid binary_label values: {invalid_labels['binary_label'].unique().tolist()}")
    print(f"Binary labels: {df['binary_label'].value_counts().to_dict()}")

    invalid_label_source = df[df["label_source"] != "real_reference_derived"]
    if len(invalid_label_source) > 0:
        raise ValueError("All rows must have label_source=real_reference_derived")
    print("All rows have label_source=real_reference_derived")

    no_change_safe = df[
        (df["raw_asr_text"] == df["proposed_corrected_text"])
        & (df["raw_asr_text"] == df["reference_text"])
        & (df["binary_label"] == "safe_to_apply")
    ]
    if len(no_change_safe) > 0:
        raise ValueError(f"Found {len(no_change_safe)} no-change rows labeled safe_to_apply")
    print("No raw==proposed==reference rows labeled safe_to_apply")

    same_text_rows = df[df["raw_asr_text"] == df["proposed_corrected_text"]]
    same_ratio = len(same_text_rows) / len(df)

    print(f"\nRows with raw_asr_text == proposed_corrected_text: {len(same_text_rows)}/{len(df)}")
    if same_ratio == 1.0:
        print(
            "WARNING: all rows have raw_asr_text == proposed_corrected_text. "
            "This may be a valid negative result if the real correction pipeline made no changes, "
            "but it means the dataset contains no positive text-changing correction examples."
        )
    elif same_ratio > 0.9:
        print(
            f"WARNING: {same_ratio:.1%} rows have raw_asr_text == proposed_corrected_text. "
            "Please confirm correction extraction is working."
        )

    nonzero_delta_rows = df[df["error_delta"].abs() > 1e-9]
    print(f"Rows with non-zero error_delta: {len(nonzero_delta_rows)}/{len(df)}")

    safe_count = int((df["binary_label"] == "safe_to_apply").sum())
    if safe_count == 0:
        print(
            "WARNING: no safe_to_apply rows found. This is acceptable as a negative result, "
            "but report it as a small real-data verification set, not a balanced benchmark."
        )

    print("\n" + "=" * 72)
    print("Dataset Distributions")
    print("=" * 72)

    print("\nSource distribution:")
    print(df["source"].value_counts())

    print("\nDataset distribution:")
    print(df["dataset"].value_counts())

    print("\nCorrection source distribution:")
    print(df["correction_source"].value_counts())

    print("\nCategory distribution:")
    print(df["category"].value_counts())

    print("\nLanguage distribution:")
    print(df["language"].value_counts())

    print("\nASR model distribution:")
    print(df["asr_model"].value_counts())

    print("\nNeeds human check:")
    print(df["needs_human_check"].value_counts())

    print("\n" + "=" * 72)
    print("Validation passed")
    print("=" * 72)

    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate real-derived safe-to-apply dataset"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to real-derived dataset CSV",
    )
    args = parser.parse_args()
    validate_dataset(args.input)


if __name__ == "__main__":
    main()
