#!/usr/bin/env python3
"""Evaluate speaker-time and overlap evidence on the formal manifest."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config import get_settings  # noqa: E402
from backend.diarization import diarize_with_metadata  # noqa: E402
from backend.events import (  # noqa: E402
    detect_interruption_events,
    detect_overlap_events,
)
from backend.reference_evidence import (  # noqa: E402
    load_reference_evidence,
    resolve_project_path,
)
from experiments.metrics.speaker_time_metrics import (  # noqa: E402
    boundary_mean_absolute_error,
    interruption_event_precision_recall_f1,
    overlap_event_precision_recall_f1,
    speaker_label_error_rate,
    turn_time_coverage,
)


OUTPUT_COLUMNS = [
    "clip_id",
    "dataset_name",
    "language",
    "mode",
    "has_reference_anchors",
    "has_reference_events",
    "num_reference_turns",
    "num_predicted_turns",
    "num_reference_overlap_events",
    "num_predicted_overlap_events",
    "speaker_label_error_rate",
    "turn_time_coverage",
    "boundary_mae",
    "overlap_precision",
    "overlap_recall",
    "overlap_f1",
    "interruption_precision",
    "interruption_recall",
    "interruption_f1",
    "notes",
]


def load_manifest_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _rounded(value: float | None) -> float | str:
    return "" if value is None else round(float(value), 6)


def _reference_overlap_count(events: list[dict[str, Any]]) -> int:
    return sum(str(event.get("type")) == "overlap" for event in events)


def _reference_event_count(
    events: list[dict[str, Any]],
    event_type: str,
) -> int:
    return sum(str(event.get("type")) == event_type for event in events)


def _module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _mode_row(
    row: dict[str, str],
    *,
    mode: str,
    reference_turns: list[dict[str, Any]],
    reference_events: list[dict[str, Any]],
    predicted_turns: list[dict[str, Any]],
    predicted_events: list[Any],
    notes: str,
) -> dict[str, Any]:
    predicted_overlap = [
        event
        for event in predicted_events
        if str(
            event.get("type")
            if isinstance(event, dict)
            else event.type
        )
        == "overlap"
    ]
    reference_overlap_count = _reference_event_count(
        reference_events,
        "overlap",
    )
    reference_interruption_count = _reference_event_count(
        reference_events,
        "interruption",
    )
    overlap = (
        overlap_event_precision_recall_f1(
            reference_events,
            predicted_events,
        )
        if reference_overlap_count
        else None
    )
    interruption = (
        interruption_event_precision_recall_f1(
            reference_events,
            predicted_events,
        )
        if reference_interruption_count
        else None
    )
    return {
        "clip_id": row["clip_id"],
        "dataset_name": row["dataset_name"],
        "language": row["language"],
        "mode": mode,
        "has_reference_anchors": str(bool(reference_turns)).lower(),
        "has_reference_events": str(bool(reference_events)).lower(),
        "num_reference_turns": len(reference_turns),
        "num_predicted_turns": len(predicted_turns),
        "num_reference_overlap_events": _reference_overlap_count(
            reference_events
        ),
        "num_predicted_overlap_events": len(predicted_overlap),
        "speaker_label_error_rate": _rounded(
            speaker_label_error_rate(reference_turns, predicted_turns)
            if reference_turns
            else None
        ),
        "turn_time_coverage": _rounded(
            turn_time_coverage(reference_turns, predicted_turns)
            if reference_turns
            else None
        ),
        "boundary_mae": _rounded(
            boundary_mean_absolute_error(
                reference_turns,
                predicted_turns,
            )
            if reference_turns
            else None
        ),
        "overlap_precision": _rounded(
            float(overlap["precision"]) if overlap else None
        ),
        "overlap_recall": _rounded(
            float(overlap["recall"]) if overlap else None
        ),
        "overlap_f1": _rounded(
            float(overlap["f1"]) if overlap else None
        ),
        "interruption_precision": _rounded(
            float(interruption["precision"]) if interruption else None
        ),
        "interruption_recall": _rounded(
            float(interruption["recall"]) if interruption else None
        ),
        "interruption_f1": _rounded(
            float(interruption["f1"]) if interruption else None
        ),
        "notes": notes,
    }


def _skipped_pyannote_row(
    row: dict[str, str],
    *,
    reference_turns: list[dict[str, Any]],
    reference_events: list[dict[str, Any]],
    reason: str,
) -> dict[str, Any]:
    return {
        "clip_id": row["clip_id"],
        "dataset_name": row["dataset_name"],
        "language": row["language"],
        "mode": "pyannote_optional",
        "has_reference_anchors": str(bool(reference_turns)).lower(),
        "has_reference_events": str(bool(reference_events)).lower(),
        "num_reference_turns": len(reference_turns),
        "num_predicted_turns": 0,
        "num_reference_overlap_events": _reference_overlap_count(
            reference_events
        ),
        "num_predicted_overlap_events": 0,
        "speaker_label_error_rate": "",
        "turn_time_coverage": "",
        "boundary_mae": "",
        "overlap_precision": "",
        "overlap_recall": "",
        "overlap_f1": "",
        "interruption_precision": "",
        "interruption_recall": "",
        "interruption_f1": "",
        "notes": f"Skipped automatic diarization: {reason}",
    }


def run_baseline(
    *,
    manifest: str | Path,
    output: str | Path,
) -> list[dict[str, Any]]:
    """Run naive, oracle/reference, and optional pyannote baselines."""

    manifest_path = resolve_project_path(manifest)
    output_path = resolve_project_path(output)
    settings = get_settings()
    pyannote_installed = _module_available("pyannote.audio")
    if not pyannote_installed:
        pyannote_reason = "pyannote.audio is not installed."
    elif not settings.hf_token:
        pyannote_reason = "HF_TOKEN is not configured."
    else:
        pyannote_reason = ""

    results: list[dict[str, Any]] = []
    for row in load_manifest_rows(manifest_path):
        reference = load_reference_evidence(row)
        reference_turns = reference["speaker_turns"]
        reference_events = reference["events"]
        duration = float(row.get("duration_seconds") or 0.0)

        no_diarization_turns = (
            [
                {
                    "start": 0.0,
                    "end": duration,
                    "speaker": "UNKNOWN",
                    "source": "naive_no_diarization",
                }
            ]
            if duration > 0
            else []
        )
        results.append(
            _mode_row(
                row,
                mode="no_diarization",
                reference_turns=reference_turns,
                reference_events=reference_events,
                predicted_turns=no_diarization_turns,
                predicted_events=[],
                notes=(
                    "Naive baseline: one UNKNOWN turn across the clip and "
                    "no predicted conversation events."
                ),
            )
        )

        reference_detected_events = [
            *detect_overlap_events(
                reference_turns,
                clip_id=row["clip_id"],
            ),
            *detect_interruption_events(
                reference_turns,
                clip_id=row["clip_id"],
            ),
        ]
        results.append(
            _mode_row(
                row,
                mode="reference_assisted",
                reference_turns=reference_turns,
                reference_events=reference_events,
                predicted_turns=reference_turns,
                predicted_events=reference_detected_events,
                notes=(
                    "Oracle/reference speaker turns; TalkWeaver rule-based "
                    "event detection. Not automatic diarization performance."
                ),
            )
        )

        if pyannote_reason:
            results.append(
                _skipped_pyannote_row(
                    row,
                    reference_turns=reference_turns,
                    reference_events=reference_events,
                    reason=pyannote_reason,
                )
            )
            continue

        audio_path = resolve_project_path(row["audio_path"])
        try:
            diarization = diarize_with_metadata(
                audio_path,
                mock=False,
                hf_token=settings.hf_token,
                fallback_to_mock=False,
                duration_seconds=duration,
            )
            if diarization.get("mode") != "pyannote":
                raise RuntimeError(
                    "Automatic diarization returned a non-pyannote mode."
                )
            predicted_turns = list(diarization["turns"])
            predicted_events = [
                *detect_overlap_events(
                    predicted_turns,
                    clip_id=row["clip_id"],
                ),
                *detect_interruption_events(
                    predicted_turns,
                    clip_id=row["clip_id"],
                ),
            ]
            results.append(
                _mode_row(
                    row,
                    mode="pyannote_optional",
                    reference_turns=reference_turns,
                    reference_events=reference_events,
                    predicted_turns=predicted_turns,
                    predicted_events=predicted_events,
                    notes=(
                        "Automatic pyannote diarization; no mock fallback."
                    ),
                )
            )
        except Exception as exc:
            results.append(
                _skipped_pyannote_row(
                    row,
                    reference_turns=reference_turns,
                    reference_events=reference_events,
                    reason=str(exc),
                )
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=OUTPUT_COLUMNS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(results)
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        rows = run_baseline(
            manifest=args.manifest,
            output=args.output,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Speaker/overlap baseline failed: {exc}", file=sys.stderr)
        return 2
    modes = sorted({str(row["mode"]) for row in rows})
    print(f"Wrote {len(rows)} rows: {args.output}")
    print(f"Modes={', '.join(modes)}")
    for mode in modes:
        mode_rows = [row for row in rows if row["mode"] == mode]
        skipped = sum(
            str(row["notes"]).startswith("Skipped") for row in mode_rows
        )
        print(f"{mode}: rows={len(mode_rows)} skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
