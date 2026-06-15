#!/usr/bin/env python3
"""
experiments/build_binary_safe_apply_real_derived.py

Build real-derived safe-to-apply evaluation table from FLEURS/AMI public audio subset.
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import numpy as np


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text


def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate (WER) between reference and hypothesis."""
    ref_words = normalize_text(reference).split()
    hyp_words = normalize_text(hypothesis).split()
    
    if not ref_words:
        return 0.0
    
    len_ref = len(ref_words)
    len_hyp = len(hyp_words)
    
    dp = np.zeros((len_ref + 1, len_hyp + 1))
    for i in range(len_ref + 1):
        dp[i][0] = i
    for j in range(len_hyp + 1):
        dp[0][j] = j
    
    for i in range(1, len_ref + 1):
        for j in range(1, len_hyp + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                cost = 0
            else:
                cost = 1
            dp[i][j] = min(
                dp[i-1][j] + 1,
                dp[i][j-1] + 1,
                dp[i-1][j-1] + cost
            )
    
    wer = dp[len_ref][len_hyp] / len_ref
    return min(wer, 1.0)


def compute_cer(reference: str, hypothesis: str) -> float:
    """Compute Character Error Rate (CER) for Chinese text."""
    ref_chars = normalize_text(reference).replace(' ', '')
    hyp_chars = normalize_text(hypothesis).replace(' ', '')
    
    if not ref_chars:
        return 0.0
    
    len_ref = len(ref_chars)
    len_hyp = len(hyp_chars)
    
    dp = np.zeros((len_ref + 1, len_hyp + 1))
    for i in range(len_ref + 1):
        dp[i][0] = i
    for j in range(len_hyp + 1):
        dp[0][j] = j
    
    for i in range(1, len_ref + 1):
        for j in range(1, len_hyp + 1):
            if ref_chars[i-1] == hyp_chars[j-1]:
                cost = 0
            else:
                cost = 1
            dp[i][j] = min(
                dp[i-1][j] + 1,
                dp[i][j-1] + 1,
                dp[i-1][j-1] + cost
            )
    
    cer = dp[len_ref][len_hyp] / len_ref
    return min(cer, 1.0)


def compute_error(reference: str, hypothesis: str, language: str) -> float:
    """Compute appropriate error metric based on language."""
    if language in ['zh', 'zh-CN', 'zh_CN', 'chinese']:
        return compute_cer(reference, hypothesis)
    else:
        return compute_wer(reference, hypothesis)


def read_manifest(manifest_path: str) -> pd.DataFrame:
    """Read and validate the real manifest."""
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    
    df = pd.read_csv(manifest_path)
    print(f"Read manifest: {len(df)} rows")
    
    # Map actual column names
    if 'clip_id' in df.columns:
        df = df.rename(columns={'clip_id': 'sample_id'})
    if 'dataset_name' in df.columns:
        df = df.rename(columns={'dataset_name': 'dataset'})
    
    if 'sample_id' not in df.columns:
        df['sample_id'] = df.index.astype(str)
    
    if 'dataset' not in df.columns:
        df['dataset'] = 'unknown'
    
    if 'language' not in df.columns:
        df['language'] = 'en'
    
    if 'audio_path' not in df.columns:
        df['audio_path'] = ''
    
    # Read reference texts from transcript_path files
    if 'transcript_path' in df.columns:
        reference_texts = []
        success_count = 0
        for idx, row in df.iterrows():
            transcript_file = row['transcript_path']
            if pd.notna(transcript_file) and transcript_file and os.path.exists(transcript_file):
                try:
                    with open(transcript_file, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        reference_texts.append(content)
                        success_count += 1
                except Exception:
                    reference_texts.append('')
            else:
                reference_texts.append('')
        df['reference_text'] = reference_texts
        print(f"Loaded reference texts: {success_count}/{len(df)} successful")
    else:
        raise ValueError(f"Could not find transcript_path column. Available: {df.columns.tolist()}")
    
    # Filter out rows without reference text
    df = df[df['reference_text'].notna() & (df['reference_text'] != '')]
    print(f"Final manifest rows: {len(df)}")
    
    return df


def read_asr_results(asr_path: str) -> pd.DataFrame:
    """Read ASR benchmark results."""
    if not os.path.exists(asr_path):
        raise FileNotFoundError(f"ASR results not found: {asr_path}")
    
    df = pd.read_csv(asr_path)
    print(f"Read ASR results: {len(df)} rows")
    
    # Map column names
    if 'hypothesis_text' in df.columns:
        df = df.rename(columns={'hypothesis_text': 'raw_asr_text'})
    elif 'raw_asr_text' not in df.columns:
        raise ValueError(f"Could not find ASR text column. Available: {df.columns.tolist()}")
    
    if 'clip_id' in df.columns:
        df = df.rename(columns={'clip_id': 'sample_id'})
    elif 'sample_id' not in df.columns:
        raise ValueError(f"Could not find sample_id column. Available: {df.columns.tolist()}")
    
    if 'model_name' in df.columns:
        df = df.rename(columns={'model_name': 'asr_model'})
    elif 'asr_model' not in df.columns:
        df['asr_model'] = 'unknown'
    
    # Keep only needed columns
    keep_cols = ['sample_id', 'raw_asr_text', 'asr_model']
    df = df[[col for col in keep_cols if col in df.columns]]
    
    return df


def find_correction_text(sample_id: str, dataset: str, raw_asr_text: str) -> Tuple[Optional[str], Optional[str]]:
    """Find existing correction text for a sample."""
    conv_map_dir = Path("outputs/conversation_maps")
    if conv_map_dir.exists():
        for json_file in conv_map_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if 'corrections' in data:
                        for corr in data['corrections']:
                            if corr.get('original_text', '') == raw_asr_text:
                                corrected = corr.get('corrected_text')
                                if corrected and corrected != raw_asr_text:
                                    return corrected, 'conversation_map_correction'
            except Exception:
                continue
    return None, None


def build_real_derived_dataset(
    manifest_path: str,
    asr_results_path: str,
    output_path: str,
    margin: float = 0.01
) -> pd.DataFrame:
    """Build the real-derived safe-to-apply dataset."""
    
    print("\n" + "="*60)
    print("Building Real-Derived Safe-to-Apply Evaluation Dataset")
    print("="*60 + "\n")
    
    print("Step 1: Reading manifest...")
    manifest_df = read_manifest(manifest_path)
    
    print("\nStep 2: Reading ASR results...")
    asr_df = read_asr_results(asr_results_path)
    
    print("\nStep 3: Merging data...")
    merged_df = manifest_df.merge(asr_df, on='sample_id', how='inner')
    print(f"Merged: {len(merged_df)} rows")
    
    print("\nStep 4: Building evaluation rows...")
    rows = []
    skipped_rows = []
    correction_sources = {}
    
    for idx, row in merged_df.iterrows():
        sample_id = row['sample_id']
        dataset = row['dataset']
        language = row['language']
        audio_path = row['audio_path']
        reference_text = row['reference_text']
        raw_asr_text = row['raw_asr_text']
        asr_model = row.get('asr_model', 'unknown')
        
        if pd.isna(raw_asr_text) or not raw_asr_text:
            skipped_rows.append((sample_id, "empty_raw_asr_text"))
            continue
        if pd.isna(reference_text) or not reference_text:
            skipped_rows.append((sample_id, "empty_reference_text"))
            continue
        
        # Try to find correction, fallback to raw text
        proposed_text, correction_source = find_correction_text(sample_id, dataset, raw_asr_text)
        
        if not proposed_text:
            # Fallback: use raw ASR text (will be labeled do_not_apply)
            proposed_text = raw_asr_text
            correction_source = "no_correction_fallback"
        
        correction_sources[correction_source] = correction_sources.get(correction_source, 0) + 1
        
        error_before = compute_error(reference_text, raw_asr_text, language)
        error_after = compute_error(reference_text, proposed_text, language)
        error_delta = error_before - error_after
        
        if raw_asr_text == proposed_text:
            binary_label = "do_not_apply"
            label_rule = "no_improvement"
        elif error_before == 0 and error_after == 0:
            binary_label = "do_not_apply"
            label_rule = "no_correction_needed"
        elif error_after + margin < error_before:
            binary_label = "safe_to_apply"
            label_rule = f"error_after + {margin} < error_before"
        else:
            binary_label = "do_not_apply"
            label_rule = f"error_after + {margin} >= error_before"
        
        if "AMI" in dataset or "ami" in dataset.lower():
            category = "real_overlap_asr_error"
        else:
            category = "real_asr_error"
        
        needs_human_check = "True" if ("AMI" in dataset or "ami" in dataset.lower()) else "False"
        
        row_dict = {
            "proposal_id": f"real::{sample_id}_{asr_model}",
            "source": "real_fleurs_asr_error" if "FLEURS" in dataset else "real_ami_asr_error",
            "dataset": dataset,
            "sample_id": sample_id,
            "language": language,
            "audio_path": audio_path,
            "asr_model": asr_model,
            "correction_source": correction_source,
            "category": category,
            "raw_asr_text": raw_asr_text,
            "proposed_corrected_text": proposed_text,
            "reference_text": reference_text,
            "context": f"Real {dataset} sample; language={language}; ASR model={asr_model}",
            "retrieved_terms": "[]",
            "overlap_flag": "False",
            "heavy_overlap_flag": "False",
            "speaker_ambiguity_flag": "False",
            "partial_utterance_flag": "False",
            "error_before": round(error_before, 4),
            "error_after": round(error_after, 4),
            "error_delta": round(error_delta, 4),
            "binary_label": binary_label,
            "label_source": "real_reference_derived",
            "label_rule": label_rule,
            "needs_human_check": needs_human_check,
            "notes": f"Real-derived from {dataset}; corrected via {correction_source}"
        }
        rows.append(row_dict)
    
    result_df = pd.DataFrame(rows)
    
    print("\n" + "="*60)
    print("Build Statistics")
    print("="*60)
    print(f"Total rows generated: {len(rows)}")
    print(f"Skipped rows: {len(skipped_rows)}")
    
    if skipped_rows:
        print("\nSkipped rows by reason:")
        reason_counts = {}
        for _, reason in skipped_rows:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        for reason, count in reason_counts.items():
            print(f"  {reason}: {count}")
    
    if len(rows) > 0:
        print("\nLabel distribution:")
        print(result_df['binary_label'].value_counts())
        
        print("\nCorrection source distribution:")
        for source, count in correction_sources.items():
            print(f"  {source}: {count}")
        
        print(f"\nAverage error_before: {result_df['error_before'].mean():.4f}")
        print(f"Average error_after: {result_df['error_after'].mean():.4f}")
        print(f"Average error_delta: {result_df['error_delta'].mean():.4f}")
    
    result_df.to_csv(output_path, index=False)
    print(f"\n💾 Saved to {output_path}")
    
    if len(rows) == 0:
        raise RuntimeError("No rows produced! Check manifest and ASR results.")
    
    return result_df


def main():
    parser = argparse.ArgumentParser(
        description="Build real-derived safe-to-apply evaluation dataset"
    )
    parser.add_argument(
        "--real-manifest",
        default="data/manifests/formal_eval_real.csv",
        help="Path to real manifest CSV"
    )
    parser.add_argument(
        "--asr-results",
        default="experiments/results/asr_benchmark_real.csv",
        help="Path to ASR benchmark results CSV"
    )
    parser.add_argument(
        "--output",
        default="data/pilot/binary_safe_apply_r2_real_derived.csv",
        help="Output path for real-derived dataset"
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.01,
        help="Margin for safe-to-apply decision"
    )
    
    args = parser.parse_args()
    
    build_real_derived_dataset(
        manifest_path=args.real_manifest,
        asr_results_path=args.asr_results,
        output_path=args.output,
        margin=args.margin
    )


if __name__ == "__main__":
    main()