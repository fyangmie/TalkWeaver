#!/usr/bin/env python3
"""Build the controlled seed dataset for TalkWeaver EvidenceGate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.evidence_gate import (
    FEATURE_COLUMNS,
    as_bool,
    extract_evidence_features,
    normalize_evidence_label,
    parse_list,
)


BASE_COLUMNS = (
    "example_id",
    "source_experiment",
    "case_id",
    "variant",
    "raw_text",
    "corrected_text",
    "reference_text",
    "expected_label",
    "label_reason",
    "label_ambiguous",
    "safety_pass",
    "needs_review",
    "correction_rejected",
    "unsupported_change_count",
    "forbidden_change_count",
    "invented_content",
    "speaker_attribution_changed",
    "overlap",
    "uncertainty_level",
    "language",
    "notes",
    "template_group",
    "is_augmented",
)


def _json_list(value: Any) -> str:
    return json.dumps(parse_list(value), ensure_ascii=False)


def _source_row(
    row: dict[str, Any],
    source_experiment: str,
    index: int,
) -> dict[str, Any]:
    normalized = dict(row)
    normalized["raw_text"] = row.get("raw_asr_text", row.get("raw_text", ""))
    normalized["corrected_text"] = row.get("corrected_text", "")
    normalized["reference_text"] = row.get("reference_text", "")
    correction_error_value = row.get("correction_error", "")
    correction_error = (
        ""
        if pd.isna(correction_error_value)
        else str(correction_error_value).strip()
    )
    normalized["correction_error"] = correction_error

    if source_experiment == "term_rescue":
        normalized["overlap"] = False
        normalized["uncertainty_level"] = (
            "medium" if as_bool(row.get("needs_review")) else "low"
        )
        normalized["correction_rejected"] = bool(correction_error)
        normalized["invented_content"] = False
        normalized["speaker_attribution_changed"] = False
        normalized["forbidden_change_count"] = 0
        normalized["safety_pass"] = not (
            correction_error or parse_list(row.get("unsupported_changes"))
        )

    label, reason, ambiguous = normalize_evidence_label(normalized)
    features = extract_evidence_features(normalized)
    case_id = str(row.get("case_id", f"case_{index:04d}"))
    variant = str(row.get("variant", "unknown"))
    payload = {
        "example_id": f"{source_experiment}__{case_id}__{variant}__{index:04d}",
        "source_experiment": source_experiment,
        "case_id": case_id,
        "variant": variant,
        "raw_text": normalized["raw_text"],
        "corrected_text": normalized["corrected_text"],
        "reference_text": normalized["reference_text"],
        "expected_label": label,
        "label_reason": reason,
        "label_ambiguous": ambiguous,
        "safety_pass": as_bool(normalized.get("safety_pass")),
        "needs_review": as_bool(normalized.get("needs_review")),
        "correction_rejected": as_bool(normalized.get("correction_rejected")),
        "unsupported_change_count": int(features["unsupported_change_count"]),
        "forbidden_change_count": int(features["forbidden_change_count"]),
        "invented_content": as_bool(normalized.get("invented_content")),
        "speaker_attribution_changed": as_bool(
            normalized.get("speaker_attribution_changed")
        ),
        "overlap": as_bool(normalized.get("overlap")),
        "uncertainty_level": normalized.get("uncertainty_level", "low"),
        "language": normalized.get("language", "unknown"),
        "notes": normalized.get("notes", ""),
        "template_group": f"{source_experiment}:{case_id}",
        "is_augmented": False,
        "retrieved_candidates": _json_list(row.get("retrieved_candidates")),
        "applied_corrections": _json_list(
            row.get("applied_corrections", row.get("applied_changes"))
        ),
        "expected_terms": _json_list(row.get("expected_terms")),
        "true_positive_terms": _json_list(row.get("true_positive_terms")),
        "false_positive_terms": _json_list(row.get("false_positive_terms")),
        "missed_terms": _json_list(row.get("missed_terms")),
        "api_used": as_bool(row.get("api_used")),
    }
    payload.update(features)
    return payload


def build_evidence_gate_dataset(
    term_input: str | Path,
    overlap_input: str | Path,
) -> pd.DataFrame:
    """Build one normalized decision row per controlled experiment row."""

    sources = (
        ("term_rescue", Path(term_input)),
        ("overlap_safety", Path(overlap_input)),
    )
    records: list[dict[str, Any]] = []
    for source_name, source_path in sources:
        if not source_path.exists():
            raise FileNotFoundError(
                f"Required {source_name} result is missing: {source_path}"
            )
        frame = pd.read_csv(source_path)
        for index, row in enumerate(frame.to_dict("records"), start=1):
            records.append(_source_row(row, source_name, index))
    columns = list(BASE_COLUMNS) + [
        "retrieved_candidates",
        "applied_corrections",
        "expected_terms",
        "true_positive_terms",
        "false_positive_terms",
        "missed_terms",
        "api_used",
    ] + list(FEATURE_COLUMNS)
    result = pd.DataFrame(records)
    return result.reindex(columns=columns)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build controlled EvidenceGate examples from Phase 2F term rescue "
            "and Phase 2G overlap safety results."
        )
    )
    parser.add_argument(
        "--term-input",
        default="experiments/results/term_rescue_controlled.csv",
    )
    parser.add_argument(
        "--overlap-input",
        default="experiments/results/overlap_safety_controlled.csv",
    )
    parser.add_argument(
        "--output",
        default="data/controlled_evidence_gate/evidence_gate_examples.csv",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frame = build_evidence_gate_dataset(args.term_input, args.overlap_input)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    ambiguous = int(frame["label_ambiguous"].sum())
    print(f"EvidenceGate seed examples: {len(frame)}")
    print(f"Template groups: {frame['template_group'].nunique()}")
    print(f"Labels: {frame['expected_label'].value_counts().to_dict()}")
    print(f"Ambiguous cases logged: {ambiguous}")
    print(f"Output: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
