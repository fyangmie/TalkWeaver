#!/usr/bin/env python3
"""Create small, transparent rule-based EvidenceGate augmentations."""

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
    extract_evidence_features,
    normalize_evidence_label,
    parse_list,
)


def _json(items: list[Any]) -> str:
    return json.dumps(items, ensure_ascii=False)


def _augment_specs(seed: dict[str, Any]) -> list[dict[str, Any]]:
    raw = str(seed.get("raw_text", "") or "the correction evidence is unclear")
    reference = str(seed.get("reference_text", "") or raw)
    expected_terms = parse_list(seed.get("expected_terms"))
    supported = reference if reference.strip() else raw
    return [
        {
            "augmentation_type": "safe_preservation",
            "raw_text": raw,
            "corrected_text": raw,
            "reference_text": raw,
            "expected_terms": _json([]),
            "safety_pass": True,
            "needs_review": False,
            "correction_rejected": False,
            "overlap": False,
            "uncertainty_level": "low",
            "invented_content": False,
            "speaker_attribution_changed": False,
            "forbidden_change_count": 0,
            "unsupported_change_count": 0,
            "text_error_before": 0.0,
            "text_error_after": 0.0,
            "notes": "Augmented safe preservation / negative control.",
        },
        {
            "augmentation_type": "safe_supported_edit",
            "raw_text": raw,
            "corrected_text": supported,
            "reference_text": supported,
            "expected_terms": _json(expected_terms),
            "safety_pass": True,
            "needs_review": False,
            "correction_rejected": False,
            "overlap": False,
            "uncertainty_level": "low",
            "invented_content": False,
            "speaker_attribution_changed": False,
            "forbidden_change_count": 0,
            "unsupported_change_count": 0,
            "text_error_before": 0.25 if raw != supported else 0.0,
            "text_error_after": 0.0,
            "notes": "Augmented evidence-supported correction.",
        },
        {
            "augmentation_type": "unsafe_invented_claim",
            "raw_text": raw,
            "corrected_text": f"{supported} This guarantees perfect accuracy.",
            "reference_text": supported,
            "safety_pass": False,
            "needs_review": True,
            "correction_rejected": True,
            "overlap": False,
            "uncertainty_level": "medium",
            "invented_content": True,
            "speaker_attribution_changed": False,
            "forbidden_change_count": 1,
            "unsupported_change_count": 1,
            "text_error_before": 0.25 if raw != supported else 0.0,
            "text_error_after": 0.35,
            "notes": "Augmented unsupported fluent completion.",
        },
        {
            "augmentation_type": "unsafe_speaker_reassignment",
            "raw_text": raw,
            "corrected_text": f"SPEAKER_99: {supported}",
            "reference_text": supported,
            "safety_pass": False,
            "needs_review": True,
            "correction_rejected": True,
            "overlap": True,
            "uncertainty_level": "medium",
            "invented_content": False,
            "speaker_attribution_changed": True,
            "forbidden_change_count": 1,
            "unsupported_change_count": 1,
            "text_error_before": 0.25 if raw != supported else 0.0,
            "text_error_after": 0.2,
            "notes": "Augmented forbidden speaker reassignment.",
        },
        {
            "augmentation_type": "heavy_overlap_review",
            "raw_text": raw,
            "corrected_text": raw,
            "reference_text": reference,
            "safety_pass": True,
            "needs_review": True,
            "correction_rejected": False,
            "overlap": True,
            "uncertainty_level": "high",
            "invented_content": False,
            "speaker_attribution_changed": False,
            "forbidden_change_count": 0,
            "unsupported_change_count": 0,
            "text_error_before": 0.3,
            "text_error_after": 0.3,
            "notes": "Augmented heavy-overlap conservative review.",
        },
        {
            "augmentation_type": "ambiguous_supported_review",
            "raw_text": raw,
            "corrected_text": supported,
            "reference_text": reference,
            "safety_pass": True,
            "needs_review": True,
            "correction_rejected": False,
            "overlap": True,
            "uncertainty_level": "medium",
            "invented_content": False,
            "speaker_attribution_changed": False,
            "forbidden_change_count": 0,
            "unsupported_change_count": 0,
            "text_error_before": 0.25,
            "text_error_after": 0.05 if raw != supported else 0.25,
            "notes": "Augmented mild-overlap correction requiring review.",
        },
    ]


def _refresh_features(row: dict[str, Any]) -> dict[str, Any]:
    row["unsupported_changes"] = _json(
        ["unsupported"] * int(float(row.get("unsupported_change_count", 0)))
    )
    row["invented_content"] = bool(row.get("invented_content"))
    row["speaker_attribution_changed"] = bool(
        row.get("speaker_attribution_changed")
    )
    label, reason, ambiguous = normalize_evidence_label(row)
    row["expected_label"] = label
    row["label_reason"] = reason
    row["label_ambiguous"] = ambiguous
    row.update(extract_evidence_features(row))
    return row


def augment_evidence_gate_examples(
    frame: pd.DataFrame,
    *,
    augmentations_per_group: int = 6,
) -> pd.DataFrame:
    """Append deterministic augmentations while preserving template groups."""

    if "template_group" not in frame:
        raise ValueError("Input must contain template_group.")
    if not 0 <= augmentations_per_group <= 6:
        raise ValueError("augmentations_per_group must be between 0 and 6.")

    original = frame.copy()
    original["is_augmented"] = False
    augmented: list[dict[str, Any]] = []
    for group_index, (template_group, rows) in enumerate(
        original.groupby("template_group", sort=True),
        start=1,
    ):
        preferred = rows.sort_values(
            ["text_error_after", "variant"],
            ascending=[True, True],
        ).iloc[0].to_dict()
        for augmentation_index, changes in enumerate(
            _augment_specs(preferred)[:augmentations_per_group],
            start=1,
        ):
            row = dict(preferred)
            row.update(changes)
            row["example_id"] = (
                f"aug__{group_index:03d}__{augmentation_index:02d}"
            )
            row["variant"] = f"augmented_{changes['augmentation_type']}"
            row["template_group"] = str(template_group)
            row["is_augmented"] = True
            row["api_used"] = False
            row["api_used_flag"] = 0.0
            row["llm_variant_flag"] = 0.0
            row["rule_variant_flag"] = 1.0
            row["applied_corrections"] = _json(
                [changes["augmentation_type"]]
                if row["raw_text"] != row["corrected_text"]
                else []
            )
            row["retrieved_candidates"] = row.get(
                "retrieved_candidates",
                _json([]),
            )
            row["true_positive_terms"] = row.get(
                "true_positive_terms",
                _json([]),
            )
            row["false_positive_terms"] = _json([])
            row["missed_terms"] = (
                row.get("expected_terms", _json([]))
                if changes["augmentation_type"] == "heavy_overlap_review"
                else _json([])
            )
            augmented.append(_refresh_features(row))

    result = pd.concat(
        [original, pd.DataFrame(augmented)],
        ignore_index=True,
        sort=False,
    )
    ordered = list(original.columns)
    for column in ("augmentation_type", *FEATURE_COLUMNS):
        if column in result and column not in ordered:
            ordered.append(column)
    return result.reindex(columns=ordered)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add transparent, group-preserving controlled examples to the "
            "EvidenceGate seed dataset."
        )
    )
    parser.add_argument(
        "--input",
        default="data/controlled_evidence_gate/evidence_gate_examples.csv",
    )
    parser.add_argument(
        "--output",
        default=(
            "data/controlled_evidence_gate/"
            "evidence_gate_examples_augmented.csv"
        ),
    )
    parser.add_argument("--augmentations-per-group", type=int, default=6)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = Path(args.input)
    if not source.exists():
        raise FileNotFoundError(f"EvidenceGate seed dataset not found: {source}")
    frame = pd.read_csv(source)
    result = augment_evidence_gate_examples(
        frame,
        augmentations_per_group=args.augmentations_per_group,
    )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination, index=False)
    print(f"Original examples: {len(frame)}")
    print(f"Augmented examples added: {len(result) - len(frame)}")
    print(f"Total examples: {len(result)}")
    print(f"Template groups: {result['template_group'].nunique()}")
    print(f"Labels: {result['expected_label'].value_counts().to_dict()}")
    print(f"Output: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
