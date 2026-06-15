#!/usr/bin/env python3
"""
experiments/validate_binary_safe_apply_real_derived.py

Validate the real-derived safe-to-apply dataset.
"""

import argparse
import pandas as pd


def validate_dataset(input_path: str):
    """Validate the real-derived dataset."""
    
    print("\n" + "="*60)
    print("Validating Real-Derived Safe-to-Apply Dataset")
    print("="*60 + "\n")
    
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows from {input_path}\n")
    
    # Check 1: Required columns exist
    required_columns = [
        "proposal_id", "source", "dataset", "sample_id", "language",
        "audio_path", "asr_model", "correction_source", "category",
        "raw_asr_text", "proposed_corrected_text", "reference_text",
        "context", "retrieved_terms", "overlap_flag", "heavy_overlap_flag",
        "speaker_ambiguity_flag", "partial_utterance_flag",
        "error_before", "error_after", "error_delta",
        "binary_label", "label_source", "label_rule",
        "needs_human_check", "notes"
    ]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"❌ Missing required columns: {missing_columns}")
        raise ValueError(f"Missing required columns: {missing_columns}")
    print("✅ All required columns present")
    
    # Check 2: Source must begin with 'real_'
    invalid_sources = df[~df['source'].str.startswith('real_', na=False)]
    if len(invalid_sources) > 0:
        print(f"❌ Found {len(invalid_sources)} rows with source not starting with 'real_':")
        print(invalid_sources['source'].unique())
        raise ValueError("All sources must start with 'real_'")
    print(f"✅ All {len(df)} rows have source starting with 'real_'")
    
    # Check 3: No controlled/heldout sources
    forbidden_sources = ['controlled_stress_test', 'heldout_controlled', 'controlled_reference']
    invalid = df[df['source'].isin(forbidden_sources)]
    if len(invalid) > 0:
        raise ValueError(f"Found {len(invalid)} rows with forbidden sources: {invalid['source'].unique()}")
    print("✅ No controlled/heldout source values found")
    
    # Check 4: No empty required text fields
    empty_raw = df[df['raw_asr_text'].isna() | (df['raw_asr_text'] == '')]
    empty_proposed = df[df['proposed_corrected_text'].isna() | (df['proposed_corrected_text'] == '')]
    empty_reference = df[df['reference_text'].isna() | (df['reference_text'] == '')]
    
    if len(empty_raw) > 0:
        raise ValueError(f"Found {len(empty_raw)} rows with empty raw_asr_text")
    if len(empty_proposed) > 0:
        raise ValueError(f"Found {len(empty_proposed)} rows with empty proposed_corrected_text")
    if len(empty_reference) > 0:
        raise ValueError(f"Found {len(empty_reference)} rows with empty reference_text")
    print("✅ No empty text fields")
    
    # Check 5: Check for oracle rows (proposed == reference)
    oracle_rows = df[df['proposed_corrected_text'] == df['reference_text']]
    if len(oracle_rows) > 0:
        print(f"⚠️ Found {len(oracle_rows)} oracle rows (proposed == reference)")
        oracle_safe = oracle_rows[oracle_rows['binary_label'] == 'safe_to_apply']
        if len(oracle_safe) > 0:
            print(f"❌ Found {len(oracle_safe)} oracle rows labeled safe_to_apply")
            raise ValueError("Oracle rows cannot be safe_to_apply in primary evaluation")
    print("✅ Oracle rows handled correctly")
    
    # Check 6: Binary label values
    valid_labels = ['safe_to_apply', 'do_not_apply']
    invalid_labels = df[~df['binary_label'].isin(valid_labels)]
    if len(invalid_labels) > 0:
        raise ValueError(f"Invalid binary_label values: {invalid_labels['binary_label'].unique()}")
    print(f"✅ Binary labels: {df['binary_label'].value_counts().to_dict()}")
    
    # Check 7: Label source
    invalid_label_source = df[df['label_source'] != 'real_reference_derived']
    if len(invalid_label_source) > 0:
        raise ValueError(f"Found {len(invalid_label_source)} rows with label_source != 'real_reference_derived'")
    print("✅ All rows have label_source='real_reference_derived'")
    
    # Check 8: No safe_to_apply when raw == proposed == reference
    no_change_safe = df[
        (df['raw_asr_text'] == df['proposed_corrected_text']) &
        (df['raw_asr_text'] == df['reference_text']) &
        (df['binary_label'] == 'safe_to_apply')
    ]
    if len(no_change_safe) > 0:
        raise ValueError(f"Found {len(no_change_safe)} rows with no change labeled safe_to_apply")
    print("✅ No safe_to_apply when raw == proposed == reference")
    
    # Print distributions
    print("\n" + "="*60)
    print("Dataset Distributions")
    print("="*60)
    
    print("\nSource distribution:")
    print(df['source'].value_counts())
    
    print("\nDataset distribution:")
    print(df['dataset'].value_counts())
    
    print("\nCorrection source distribution:")
    print(df['correction_source'].value_counts())
    
    print("\nCategory distribution:")
    print(df['category'].value_counts())
    
    print("\nLanguage distribution:")
    print(df['language'].value_counts())
    
    print("\nASR model distribution:")
    print(df['asr_model'].value_counts())
    
    print("\nNeeds human check:")
    print(df['needs_human_check'].value_counts())
    
    print("\n" + "="*60)
    print("✅ Validation passed!")
    print("="*60)
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Validate real-derived safe-to-apply dataset"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to real-derived dataset CSV"
    )
    
    args = parser.parse_args()
    
    validate_dataset(args.input)


if __name__ == "__main__":
    main()
