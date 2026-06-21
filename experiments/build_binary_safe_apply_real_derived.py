#!/usr/bin/env python3
"""
Build real-derived safe-to-apply evaluation table from existing FLEURS/AMI
public audio ASR results and TalkWeaver workflow correction outputs.

This script DOES NOT use controlled/template heldout rows.
It builds rows from:
- data/manifests/formal_eval_real.csv
- experiments/results/asr_benchmark_real.csv
- outputs/conversation_maps/ablation_real/full_talkweaver/*.json
- outputs/conversation_maps/ablation_real/constrained_correction/*.json
"""

from __future__ import annotations

import argparse
import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


TEXT_COLUMNS_REFERENCE = [
    "reference_text",
    "reference",
    "transcript",
    "text",
    "gold_text",
]
TEXT_COLUMNS_RAW_ASR = [
    "raw_asr_text",
    "hypothesis_text",
    "hypothesis",
    "prediction",
    "predicted_text",
    "asr_text",
    "transcript_pred",
    "raw_text",
]
SAMPLE_ID_COLUMNS = ["sample_id", "clip_id", "id", "uid", "audio_id"]
DATASET_COLUMNS = ["dataset", "dataset_name", "source_dataset", "corpus"]
LANGUAGE_COLUMNS = ["language", "lang"]
AUDIO_COLUMNS = ["audio_path", "path", "file_path", "wav_path"]
ASR_MODEL_COLUMNS = ["asr_model", "model_name", "model"]


def safe_str(value: Any) -> str:
    """Convert a possibly missing value to a clean string."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def first_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def normalize_text(text: str) -> str:
    text = safe_str(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def edit_distance(seq_a: List[str], seq_b: List[str]) -> int:
    """Classic Levenshtein edit distance for token/character sequences."""
    if not seq_a:
        return len(seq_b)
    if not seq_b:
        return len(seq_a)

    prev = list(range(len(seq_b) + 1))
    for i, a in enumerate(seq_a, start=1):
        cur = [i]
        for j, b in enumerate(seq_b, start=1):
            cost = 0 if a == b else 1
            cur.append(
                min(
                    prev[j] + 1,
                    cur[j - 1] + 1,
                    prev[j - 1] + cost,
                )
            )
        prev = cur
    return prev[-1]


def compute_wer(reference: str, hypothesis: str) -> float:
    ref_words = normalize_text(reference).split()
    hyp_words = normalize_text(hypothesis).split()
    if not ref_words:
        return 0.0
    return min(edit_distance(ref_words, hyp_words) / len(ref_words), 1.0)


def compute_cer(reference: str, hypothesis: str) -> float:
    ref_chars = list(normalize_text(reference).replace(" ", ""))
    hyp_chars = list(normalize_text(hypothesis).replace(" ", ""))
    if not ref_chars:
        return 0.0
    return min(edit_distance(ref_chars, hyp_chars) / len(ref_chars), 1.0)


def compute_error(reference: str, hypothesis: str, language: str) -> float:
    lang = normalize_text(language)
    if lang.startswith("zh") or "chinese" in lang:
        return compute_cer(reference, hypothesis)
    return compute_wer(reference, hypothesis)


def text_similarity(a: str, b: str) -> float:
    a_norm = normalize_text(a)
    b_norm = normalize_text(b)
    if not a_norm or not b_norm:
        return 0.0
    return SequenceMatcher(None, a_norm, b_norm).ratio()


def read_reference_from_transcript(path_value: Any) -> str:
    transcript_path = safe_str(path_value)
    if not transcript_path:
        return ""
    path = Path(transcript_path)
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def load_manifest(manifest_path: str) -> Dict[str, Dict[str, str]]:
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df = pd.read_csv(path)
    print(f"Read manifest: {len(df)} rows from {manifest_path}")

    sample_col = first_existing_column(df, SAMPLE_ID_COLUMNS)
    dataset_col = first_existing_column(df, DATASET_COLUMNS)
    language_col = first_existing_column(df, LANGUAGE_COLUMNS)
    audio_col = first_existing_column(df, AUDIO_COLUMNS)
    ref_col = first_existing_column(df, TEXT_COLUMNS_REFERENCE)
    transcript_path_col = "transcript_path" if "transcript_path" in df.columns else None

    if sample_col is None:
        raise ValueError(f"No sample id column found in manifest. Columns={df.columns.tolist()}")

    records: Dict[str, Dict[str, str]] = {}
    for _, row in df.iterrows():
        sample_id = safe_str(row.get(sample_col))
        if not sample_id:
            continue

        reference_text = safe_str(row.get(ref_col)) if ref_col else ""
        if not reference_text and transcript_path_col:
            reference_text = read_reference_from_transcript(row.get(transcript_path_col))

        records[sample_id] = {
            "sample_id": sample_id,
            "dataset": safe_str(row.get(dataset_col)) if dataset_col else "",
            "language": safe_str(row.get(language_col)) if language_col else "en",
            "audio_path": safe_str(row.get(audio_col)) if audio_col else "",
            "reference_text": reference_text,
        }

    with_ref = sum(1 for r in records.values() if r["reference_text"])
    print(f"Manifest usable records: {len(records)}; with reference: {with_ref}")
    return records


def load_asr_results(asr_results_path: str) -> List[Dict[str, str]]:
    path = Path(asr_results_path)
    if not path.exists():
        raise FileNotFoundError(f"ASR results not found: {asr_results_path}")

    df = pd.read_csv(path)
    print(f"Read ASR results: {len(df)} rows from {asr_results_path}")

    sample_col = first_existing_column(df, SAMPLE_ID_COLUMNS)
    dataset_col = first_existing_column(df, DATASET_COLUMNS)
    language_col = first_existing_column(df, LANGUAGE_COLUMNS)
    audio_col = first_existing_column(df, AUDIO_COLUMNS)
    raw_col = first_existing_column(df, TEXT_COLUMNS_RAW_ASR)
    ref_col = first_existing_column(df, TEXT_COLUMNS_REFERENCE)
    model_col = first_existing_column(df, ASR_MODEL_COLUMNS)

    if sample_col is None:
        raise ValueError(f"No sample id column found in ASR results. Columns={df.columns.tolist()}")
    if raw_col is None:
        raise ValueError(f"No raw ASR text column found in ASR results. Columns={df.columns.tolist()}")

    records: List[Dict[str, str]] = []
    for _, row in df.iterrows():
        records.append(
            {
                "sample_id": safe_str(row.get(sample_col)),
                "dataset": safe_str(row.get(dataset_col)) if dataset_col else "",
                "language": safe_str(row.get(language_col)) if language_col else "",
                "audio_path": safe_str(row.get(audio_col)) if audio_col else "",
                "raw_asr_text": safe_str(row.get(raw_col)),
                "reference_text": safe_str(row.get(ref_col)) if ref_col else "",
                "asr_model": safe_str(row.get(model_col)) if model_col else "unknown",
            }
        )
    return records


def parse_sample_and_model_from_text(text: str) -> Tuple[str, str]:
    """
    Parse sample id and ASR model from path-like text.

    Expected examples:
    - base__fleurs_en_1548.json
    - tiny__ami_es2002a_01.json
    - fleurs_zh_cn_1579
    """
    text = safe_str(text)

    model = ""
    model_match = re.search(r"(tiny|base|small|medium|large)(?:__|[_\-.])", text, flags=re.IGNORECASE)
    if model_match:
        model = model_match.group(1).lower()

    sample = ""
    sample_match = re.search(
        r"(fleurs_[A-Za-z0-9_]+?_\d+|ami_[A-Za-z0-9]+_\d+)",
        text,
        flags=re.IGNORECASE,
    )
    if sample_match:
        sample = sample_match.group(1)

    return sample, model


def get_nested_dict(data: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = data.get(key, {})
    return value if isinstance(value, dict) else {}


def get_anchor_list(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Find anchors in common ConversationMap structures."""
    candidates: List[Any] = [
        data.get("anchors"),
        get_nested_dict(data, "conversation_map").get("anchors"),
        get_nested_dict(data, "map").get("anchors"),
    ]

    for candidate in candidates:
        if isinstance(candidate, list):
            return [item for item in candidate if isinstance(item, dict)]

    return []


def first_text_from_dict(item: Dict[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        value = safe_str(item.get(key))
        if value:
            return value
    return ""


def join_nonempty(parts: Iterable[str]) -> str:
    return " ".join(part.strip() for part in parts if safe_str(part)).strip()


def extract_texts_from_conversation_map(data: Dict[str, Any]) -> Tuple[str, str]:
    """
    Extract raw and corrected text from ConversationMap JSON.

    Preferred:
    - anchors[].raw_text
    - anchors[].corrected_text

    Fallback:
    - correction_audits[]
    - corrections[]
    """
    anchors = get_anchor_list(data)
    if anchors:
        raw_parts: List[str] = []
        corrected_parts: List[str] = []
        for anchor in anchors:
            raw = first_text_from_dict(
                anchor,
                ["raw_text", "asr_text", "original_text", "text", "utterance"],
            )
            corrected = first_text_from_dict(
                anchor,
                [
                    "corrected_text",
                    "final_text",
                    "llm_corrected_text",
                    "rule_corrected_text",
                    "constrained_corrected_text",
                ],
            )
            if raw:
                raw_parts.append(raw)
            if corrected:
                corrected_parts.append(corrected)
            elif raw:
                corrected_parts.append(raw)

        raw_joined = join_nonempty(raw_parts)
        corrected_joined = join_nonempty(corrected_parts)
        if raw_joined and corrected_joined:
            return raw_joined, corrected_joined

    audits = data.get("correction_audits", [])
    if isinstance(audits, list) and audits:
        raw_parts = []
        corrected_parts = []
        for audit in audits:
            if not isinstance(audit, dict):
                continue
            raw = first_text_from_dict(
                audit,
                ["raw_text", "original_text", "before_text", "input_text"],
            )
            corrected = first_text_from_dict(
                audit,
                ["corrected_text", "after_text", "proposed_text", "final_text"],
            )
            if raw:
                raw_parts.append(raw)
            if corrected:
                corrected_parts.append(corrected)
            elif raw:
                corrected_parts.append(raw)
        if raw_parts and corrected_parts:
            return join_nonempty(raw_parts), join_nonempty(corrected_parts)

    corrections = data.get("corrections", [])
    if isinstance(corrections, list) and corrections:
        raw_parts = []
        corrected_parts = []
        for corr in corrections:
            if not isinstance(corr, dict):
                continue
            raw = first_text_from_dict(corr, ["original_text", "raw_text", "before_text"])
            corrected = first_text_from_dict(corr, ["corrected_text", "final_text", "after_text"])
            if raw:
                raw_parts.append(raw)
            if corrected:
                corrected_parts.append(corrected)
            elif raw:
                corrected_parts.append(raw)
        if raw_parts and corrected_parts:
            return join_nonempty(raw_parts), join_nonempty(corrected_parts)

    return "", ""


def extract_metadata_sample_and_model(data: Dict[str, Any], json_path: Path) -> Tuple[str, str]:
    metadata = get_nested_dict(data, "metadata")

    candidate_texts = [
        safe_str(metadata.get("clip_id")),
        safe_str(metadata.get("sample_id")),
        safe_str(metadata.get("audio_id")),
        safe_str(metadata.get("asr_prediction_json")),
        safe_str(metadata.get("prediction_json_path")),
        safe_str(metadata.get("audio_path")),
        str(json_path),
    ]

    sample_id = ""
    asr_model = ""

    for text in candidate_texts:
        parsed_sample, parsed_model = parse_sample_and_model_from_text(text)
        if parsed_sample and not sample_id:
            sample_id = parsed_sample
        if parsed_model and not asr_model:
            asr_model = parsed_model

    if not asr_model:
        asr_model = safe_str(metadata.get("model_name")) or safe_str(metadata.get("asr_model"))

    return sample_id, asr_model or "unknown"


def collect_conversation_map_records() -> List[Dict[str, str]]:
    """
    Collect system-generated correction outputs from real workflow ConversationMaps.
    """
    conv_root = Path("outputs/conversation_maps")
    if not conv_root.exists():
        raise FileNotFoundError("outputs/conversation_maps does not exist")

    preferred_dirs = [
        conv_root / "ablation_real" / "full_talkweaver",
        conv_root / "ablation_real" / "constrained_correction",
    ]

    json_files: List[Path] = []
    for directory in preferred_dirs:
        if directory.exists():
            json_files.extend(sorted(directory.glob("*.json")))

    if not json_files:
        json_files = sorted(conv_root.rglob("*.json"))

    seen = set()
    unique_json_files = []
    for path in json_files:
        if path not in seen:
            unique_json_files.append(path)
            seen.add(path)

    print(f"Found ConversationMap JSON files: {len(unique_json_files)}")

    records: List[Dict[str, str]] = []
    skipped = []

    for json_file in unique_json_files:
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
        except Exception as exc:
            skipped.append((str(json_file), f"json_error:{exc}"))
            continue

        if not isinstance(data, dict):
            skipped.append((str(json_file), "not_dict_json"))
            continue

        sample_id, asr_model = extract_metadata_sample_and_model(data, json_file)
        raw_text, corrected_text = extract_texts_from_conversation_map(data)

        if not sample_id:
            skipped.append((str(json_file), "missing_sample_id"))
            continue
        if not raw_text:
            skipped.append((str(json_file), "missing_raw_text"))
            continue
        if not corrected_text:
            skipped.append((str(json_file), "missing_corrected_text"))
            continue

        variant = json_file.parent.name
        records.append(
            {
                "sample_id": sample_id,
                "asr_model": asr_model,
                "raw_from_map": raw_text,
                "corrected_text": corrected_text,
                "correction_source": f"conversation_map_anchor:{variant}",
                "conversation_map_path": str(json_file),
                "variant": variant,
            }
        )

    print(f"Usable ConversationMap correction records: {len(records)}")
    if skipped:
        reason_counts: Dict[str, int] = {}
        for _, reason in skipped:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        print("Skipped ConversationMap files:")
        for reason, count in sorted(reason_counts.items()):
            print(f"  {reason}: {count}")

    return records


def find_asr_record(
    asr_records: List[Dict[str, str]],
    sample_id: str,
    asr_model: str,
) -> Optional[Dict[str, str]]:
    exact = [
        r
        for r in asr_records
        if r.get("sample_id") == sample_id and normalize_text(r.get("asr_model", "")) == normalize_text(asr_model)
    ]
    if exact:
        return exact[0]

    same_sample = [r for r in asr_records if r.get("sample_id") == sample_id]
    if not same_sample:
        return None

    # Prefer base if model is unknown, otherwise use the first available record.
    for record in same_sample:
        if normalize_text(record.get("asr_model", "")) == "base":
            return record

    return same_sample[0]


def infer_source(dataset: str, sample_id: str) -> str:
    text = f"{dataset} {sample_id}".lower()
    if "ami" in text:
        return "real_ami_asr_error"
    return "real_fleurs_asr_error"


def infer_flags(dataset: str, sample_id: str) -> Tuple[str, str, str, str, str]:
    text = f"{dataset} {sample_id}".lower()
    is_ami = "ami" in text
    if is_ami:
        return "True", "False", "True", "False", "True"
    return "False", "False", "False", "False", "False"


def build_real_derived_dataset(
    manifest_path: str,
    asr_results_path: str,
    output_path: str,
    margin: float = 0.01,
) -> pd.DataFrame:
    print("\n" + "=" * 72)
    print("Building Real-Derived Safe-to-Apply Evaluation Dataset")
    print("=" * 72 + "\n")

    manifest_records = load_manifest(manifest_path)
    asr_records = load_asr_results(asr_results_path)
    correction_records = collect_conversation_map_records()

    rows: List[Dict[str, Any]] = []
    skipped_rows: List[Tuple[str, str]] = []

    for corr in correction_records:
        sample_id = corr["sample_id"]
        asr_model = corr["asr_model"]
        asr_record = find_asr_record(asr_records, sample_id, asr_model)
        manifest_record = manifest_records.get(sample_id, {})

        if not manifest_record and not asr_record:
            skipped_rows.append((sample_id, "no_manifest_or_asr_record"))
            continue

        dataset = (
            safe_str(manifest_record.get("dataset"))
            or safe_str(asr_record.get("dataset") if asr_record else "")
            or "unknown"
        )
        language = (
            safe_str(manifest_record.get("language"))
            or safe_str(asr_record.get("language") if asr_record else "")
            or "en"
        )
        audio_path = (
            safe_str(manifest_record.get("audio_path"))
            or safe_str(asr_record.get("audio_path") if asr_record else "")
        )

        reference_text = (
            safe_str(manifest_record.get("reference_text"))
            or safe_str(asr_record.get("reference_text") if asr_record else "")
        )
        if not reference_text:
            skipped_rows.append((sample_id, "missing_reference_text"))
            continue

        # Prefer map raw text because it corresponds exactly to the correction output.
        raw_asr_text = corr["raw_from_map"]
        proposed_text = corr["corrected_text"]

        if not raw_asr_text:
            skipped_rows.append((sample_id, "empty_raw_from_map"))
            continue
        if not proposed_text:
            skipped_rows.append((sample_id, "empty_corrected_text"))
            continue
        if proposed_text == reference_text and "oracle" in normalize_text(corr["correction_source"]):
            skipped_rows.append((sample_id, "oracle_correction_excluded"))
            continue

        error_before = compute_error(reference_text, raw_asr_text, language)
        error_after = compute_error(reference_text, proposed_text, language)
        error_delta = error_before - error_after

        if normalize_text(raw_asr_text) == normalize_text(proposed_text):
            binary_label = "do_not_apply"
            label_rule = "no_improvement"
        elif error_before == 0.0 and error_after == 0.0:
            binary_label = "do_not_apply"
            label_rule = "no_correction_needed"
        elif error_after + margin < error_before:
            binary_label = "safe_to_apply"
            label_rule = f"error_after + {margin:.3f} < error_before"
        else:
            binary_label = "do_not_apply"
            label_rule = f"error_after + {margin:.3f} >= error_before"

        source = infer_source(dataset, sample_id)
        overlap_flag, heavy_overlap_flag, speaker_ambiguity_flag, partial_utterance_flag, needs_human_check = infer_flags(
            dataset, sample_id
        )

        if source == "real_ami_asr_error":
            category = "real_overlap_asr_error"
        elif normalize_text(raw_asr_text) == normalize_text(proposed_text):
            category = "real_no_improvement"
        else:
            category = "real_asr_error"

        proposal_id = f"real::{sample_id}_{asr_model}_{corr['variant']}"

        context = (
            f"Real {dataset} sample; language={language}; ASR model={asr_model}; "
            f"correction_source={corr['correction_source']}"
        )

        rows.append(
            {
                "proposal_id": proposal_id,
                "source": source,
                "dataset": dataset,
                "sample_id": sample_id,
                "language": language,
                "audio_path": audio_path,
                "asr_model": asr_model,
                "correction_source": corr["correction_source"],
                "category": category,
                "raw_asr_text": raw_asr_text,
                "proposed_corrected_text": proposed_text,
                "reference_text": reference_text,
                "context": context,
                "retrieved_terms": "[]",
                "overlap_flag": overlap_flag,
                "heavy_overlap_flag": heavy_overlap_flag,
                "speaker_ambiguity_flag": speaker_ambiguity_flag,
                "partial_utterance_flag": partial_utterance_flag,
                "error_before": round(error_before, 4),
                "error_after": round(error_after, 4),
                "error_delta": round(error_delta, 4),
                "binary_label": binary_label,
                "label_source": "real_reference_derived",
                "label_rule": label_rule,
                "needs_human_check": needs_human_check,
                "notes": (
                    f"Real-derived from {dataset}; map={corr['conversation_map_path']}; "
                    f"correction via {corr['correction_source']}"
                ),
            }
        )

    result_df = pd.DataFrame(rows)

    print("\n" + "=" * 72)
    print("Build Statistics")
    print("=" * 72)
    print(f"Total rows generated: {len(result_df)}")
    print(f"Skipped rows: {len(skipped_rows)}")

    if skipped_rows:
        reason_counts: Dict[str, int] = {}
        for _, reason in skipped_rows:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        print("\nSkipped rows by reason:")
        for reason, count in sorted(reason_counts.items()):
            print(f"  {reason}: {count}")

    if result_df.empty:
        raise RuntimeError(
            "No real-derived rows produced. The builder failed to connect real "
            "ConversationMap correction outputs to manifest/ASR records."
        )

    print("\nLabel distribution:")
    print(result_df["binary_label"].value_counts())

    print("\nSource distribution:")
    print(result_df["source"].value_counts())

    print("\nCorrection source distribution:")
    print(result_df["correction_source"].value_counts())

    print("\nCategory distribution:")
    print(result_df["category"].value_counts())

    same_text_count = int((result_df["raw_asr_text"] == result_df["proposed_corrected_text"]).sum())
    nonzero_delta_count = int((result_df["error_delta"].abs() > 1e-9).sum())

    print(f"\nRows where raw == proposed: {same_text_count}/{len(result_df)}")
    print(f"Rows with non-zero error_delta: {nonzero_delta_count}/{len(result_df)}")
    print(f"Average error_before: {result_df['error_before'].mean():.4f}")
    print(f"Average error_after:  {result_df['error_after'].mean():.4f}")
    print(f"Average error_delta:  {result_df['error_delta'].mean():.4f}")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output, index=False)
    print(f"\nSaved to {output}")

    return result_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build real-derived binary safe-to-apply evaluation dataset"
    )
    parser.add_argument(
        "--real-manifest",
        default="data/manifests/formal_eval_real.csv",
        help="Path to real manifest CSV",
    )
    parser.add_argument(
        "--asr-results",
        default="experiments/results/asr_benchmark_real.csv",
        help="Path to ASR benchmark results CSV",
    )
    parser.add_argument(
        "--output",
        default="data/pilot/binary_safe_apply_r2_real_derived.csv",
        help="Output path",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.01,
        help="Safe-to-apply improvement margin",
    )

    args = parser.parse_args()

    build_real_derived_dataset(
        manifest_path=args.real_manifest,
        asr_results_path=args.asr_results,
        output_path=args.output,
        margin=args.margin,
    )


if __name__ == "__main__":
    main()
