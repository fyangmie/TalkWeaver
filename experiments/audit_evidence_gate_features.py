#!/usr/bin/env python3
"""Audit EvidenceGate fields for reference and label-proxy leakage."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.evidence_gate import (
    AUDIT_AWARE_FEATURES,
    DIRECT_LABEL_PROXY_FEATURES,
    EVIDENCE_ONLY_FEATURES,
    FINAL_AUDIT_OUTCOME_FEATURES,
    REFERENCE_DERIVED_FEATURES,
    RISK_ONLY_FEATURES,
)


EXTRA_AUDITED_FIELDS = {
    "safety_pass": (
        "direct_label_proxy_features",
        "Final policy pass/fail value used to derive the target label.",
    ),
    "needs_review": (
        "direct_label_proxy_features",
        "Final review decision; direct target proxy.",
    ),
    "correction_rejected": (
        "direct_label_proxy_features",
        "Final rejection decision; direct target proxy.",
    ),
    "expected_label": (
        "direct_label_proxy_features",
        "Training target and never a model input.",
    ),
    "reference_text": (
        "risky_reference_derived_features",
        "Ground-truth text unavailable at deployment time.",
    ),
}


def classify_feature(feature: str) -> tuple[str, str]:
    if feature in DIRECT_LABEL_PROXY_FEATURES:
        return (
            "direct_label_proxy_features",
            "Direct encoding of the review/rejection target decision.",
        )
    if feature in FINAL_AUDIT_OUTCOME_FEATURES:
        return (
            "final_audit_outcome_features",
            "Post-correction audit outcome unavailable before gating.",
        )
    if feature in REFERENCE_DERIVED_FEATURES:
        return (
            "risky_reference_derived_features",
            "Requires reference labels, expected terms, or derived correctness.",
        )
    return (
        "allowed_pre_decision_features",
        "Available from the proposal, retrieval trace, overlap evidence, or model metadata.",
    )


def build_feature_leakage_audit() -> pd.DataFrame:
    all_features = set(AUDIT_AWARE_FEATURES) | set(EXTRA_AUDITED_FIELDS)
    rows = []
    for feature in sorted(all_features):
        category, rationale = EXTRA_AUDITED_FIELDS.get(
            feature,
            classify_feature(feature),
        )
        rows.append(
            {
                "feature_or_field": feature,
                "category": category,
                "audit_aware": feature in AUDIT_AWARE_FEATURES,
                "evidence_only": feature in EVIDENCE_ONLY_FEATURES,
                "risk_only": feature in RISK_ONLY_FEATURES,
                "excluded_from_strict_models": (
                    feature not in EVIDENCE_ONLY_FEATURES
                    and feature not in RISK_ONLY_FEATURES
                ),
                "rationale": rationale,
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Classify EvidenceGate features as pre-decision, reference-derived, "
            "direct label proxies, or final audit outcomes."
        )
    )
    parser.add_argument(
        "--output",
        default=(
            "experiments/results/evidence_gate/"
            "evidence_gate_feature_leakage_audit.csv"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frame = build_feature_leakage_audit()
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    print(frame.groupby("category").size().to_string())
    print(f"Output: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
