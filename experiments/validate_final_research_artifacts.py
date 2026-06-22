#!/usr/bin/env python3
"""Validate final TalkWeaver research artifacts and claim boundaries."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

EXPECTED_ROW_COUNTS = {
    "data/manifests/formal_eval_real.csv": 50,
    "data/manifests/mandarin_meeting_real.csv": 12,
    "data/manifests/aishell4_benchmark_60x20.csv": 60,
    "experiments/results/asr_benchmark_real.csv": 100,
    "experiments/results/asr_benchmark_aishell4_60x20_real.csv": 180,
    "experiments/results/pyannote_diarization_aishell4_60x20_real.csv": 29,
    "experiments/results/automatic_pyannote_workflow_aishell4_60x20_real.csv": 29,
    "experiments/results/earnings22_v3_blind_ablation_v3_summary.csv": 10,
    "experiments/results/workflow_ablation_real.csv": 350,
    "experiments/results/speaker_overlap_baseline_real.csv": 150,
    "experiments/results/v1/mobile_asr.csv": 100,
}

REQUIRED_FILES = [
    "docs/final_claim_matrix.md",
    "docs/final_system_error_analysis.md",
    "PROJECT_REPORT.md",
    "docs/video_script.md",
    "experiments/results/asr_benchmark_summary_real.csv",
    "experiments/results/asr_benchmark_aishell4_60x20_summary_real.csv",
    "experiments/results/pyannote_diarization_heldout_summary_real.csv",
    "experiments/results/pyannote_diarization_aishell4_60x20_summary_real.csv",
    "experiments/results/automatic_pyannote_workflow_aishell4_60x20_summary_real.csv",
    "experiments/results/pyannote_diarization_mandarin_meeting_summary_real.csv",
    "experiments/results/earnings22_v3_blind_ablation_v3_summary.csv",
    "experiments/results/interruption_label_summary_heldout_real.csv",
]

STALE_PHRASES = [
    "formal real manifest to 38 public clips",
    "For ASR, we expanded the formal real manifest to 38 public clips",
    "Mandarin meeting track is blocked",
    "Mandarin meeting data are still absent",
    "No Mandarin meeting sample is locally available yet",
]


def _read_csv_rows(relative_path: str) -> list[dict[str, str]]:
    path = ROOT / relative_path
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _check_required_files(errors: list[str]) -> None:
    for relative_path in REQUIRED_FILES:
        path = ROOT / relative_path
        if not path.exists():
            errors.append(f"Missing required file: {relative_path}")


def _check_row_counts(errors: list[str]) -> None:
    for relative_path, expected in EXPECTED_ROW_COUNTS.items():
        path = ROOT / relative_path
        if not path.exists():
            errors.append(f"Missing counted CSV: {relative_path}")
            continue
        rows = _read_csv_rows(relative_path)
        if len(rows) != expected:
            errors.append(
                f"{relative_path} has {len(rows)} rows; expected {expected}"
            )


def _check_mobile_proxy(errors: list[str]) -> None:
    rows = _read_csv_rows("experiments/results/v1/mobile_asr.csv")
    bad_claims = sorted(
        {
            row.get("claim_level", "")
            for row in rows
            if row.get("claim_level") != "mobile_style_proxy"
        }
    )
    if bad_claims:
        errors.append(f"Unexpected mobile claim levels: {bad_claims}")
    true_devices = [
        row.get("clip_id", "")
        for row in rows
        if row.get("true_mobile_device") != "false"
    ]
    if true_devices:
        errors.append(
            "mobile_asr.csv contains rows marked true_mobile_device != false"
        )


def _check_interruption_scope(errors: list[str]) -> None:
    rows = _read_csv_rows(
        "experiments/results/interruption_label_summary_heldout_real.csv"
    )
    if len(rows) != 1:
        errors.append("Interruption summary should contain exactly one row.")
        return
    row = rows[0]
    if row.get("candidate_precision") != "1.0":
        errors.append("Interruption candidate precision is not 1.0.")
    if row.get("recall") or row.get("f1"):
        errors.append("Interruption recall/F1 must remain blank.")


def _check_final_docs(errors: list[str]) -> None:
    docs = [
        ROOT / "PROJECT_REPORT.md",
        ROOT / "docs" / "video_script.md",
        ROOT / "docs" / "final_claim_matrix.md",
        ROOT / "docs" / "final_system_error_analysis.md",
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in docs)
    for phrase in STALE_PHRASES:
        if phrase in combined:
            errors.append(f"Stale phrase remains in final docs: {phrase!r}")
    required_phrases = [
        "50 public clips",
        "60 AISHELL-4",
        "29 multi-speaker",
        "AISHELL-4 evidence map",
        "glossary_candidates_only",
        "AISHELL-4",
        "candidate precision",
        "not a phone-device measurement",
        "not full FLEURS, AMI, or AISHELL-4",
    ]
    for phrase in required_phrases:
        if phrase not in combined:
            errors.append(f"Required final-doc phrase missing: {phrase!r}")


def validate() -> list[str]:
    errors: list[str] = []
    _check_required_files(errors)
    if errors:
        return errors
    _check_row_counts(errors)
    _check_mobile_proxy(errors)
    _check_interruption_scope(errors)
    _check_final_docs(errors)
    return errors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print failures.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    errors = validate()
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    if not args.quiet:
        print("Final research artifact validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
