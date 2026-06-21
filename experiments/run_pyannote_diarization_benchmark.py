#!/usr/bin/env python3
"""Run automatic pyannote diarization and standard DER/JER scoring."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from time import perf_counter
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config import get_settings  # noqa: E402
from backend.diarization import (  # noqa: E402
    DEFAULT_DIARIZATION_MODEL,
    diarize_with_metadata,
)
from backend.events import detect_interruption_events, detect_overlap_events  # noqa: E402
from backend.reference_evidence import load_reference_evidence  # noqa: E402
from experiments.metrics.speaker_time_metrics import (  # noqa: E402
    boundary_mean_absolute_error,
    interruption_event_precision_recall_f1,
    overlap_event_precision_recall_f1,
    speaker_label_error_rate,
    turn_time_coverage,
)
from experiments.metrics.standard_diarization_metrics import compute_der_jer  # noqa: E402
from scripts.dataset_utils import resolve_repo_path  # noqa: E402


OUTPUT_COLUMNS = [
    "clip_id",
    "dataset_name",
    "language",
    "model_name",
    "metric_status",
    "duration_seconds",
    "runtime_seconds",
    "rtf",
    "num_reference_turns",
    "num_predicted_turns",
    "num_reference_speakers",
    "num_predicted_speakers",
    "der",
    "jer",
    "der_skip_overlap",
    "jer_skip_overlap",
    "project_speaker_label_error",
    "turn_time_coverage",
    "boundary_mae",
    "overlap_precision",
    "overlap_recall",
    "overlap_f1",
    "interruption_precision",
    "interruption_recall",
    "interruption_f1",
    "predicted_turns_path",
    "claim_level",
    "notes",
]

SUMMARY_COLUMNS = [
    "dataset_name",
    "language",
    "model_name",
    "metric_status",
    "num_clips",
    "mean_der",
    "mean_jer",
    "mean_der_skip_overlap",
    "mean_jer_skip_overlap",
    "mean_project_speaker_label_error",
    "mean_overlap_f1",
    "mean_rtf",
    "notes",
]


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _bool(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def _speaker_count(turns: list[dict[str, Any]]) -> int:
    return len(
        {
            str(turn.get("speaker", ""))
            for turn in turns
            if str(turn.get("speaker", "")) not in {"", "UNKNOWN", "OVERLAP"}
        }
    )


def _event_count(events: list[dict[str, Any]], event_type: str) -> int:
    return sum(str(event.get("type")) == event_type for event in events)


def _event_dicts(events: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in events:
        if isinstance(event, dict):
            rows.append(event)
        else:
            rows.append(
                {
                    "type": event.type,
                    "start": event.start,
                    "end": event.end,
                    "speakers": list(event.speakers),
                }
            )
    return rows


def _rounded(value: Any) -> str:
    if value in {"", None}:
        return ""
    return f"{float(value):.6f}"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _skipped_row(
    row: dict[str, str],
    *,
    model_name: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "clip_id": row.get("clip_id", ""),
        "dataset_name": row.get("dataset_name", ""),
        "language": row.get("language", ""),
        "model_name": model_name,
        "metric_status": "skipped",
        "duration_seconds": row.get("duration_seconds", ""),
        "runtime_seconds": "",
        "rtf": "",
        "num_reference_turns": "",
        "num_predicted_turns": "",
        "num_reference_speakers": "",
        "num_predicted_speakers": "",
        "der": "",
        "jer": "",
        "der_skip_overlap": "",
        "jer_skip_overlap": "",
        "project_speaker_label_error": "",
        "turn_time_coverage": "",
        "boundary_mae": "",
        "overlap_precision": "",
        "overlap_recall": "",
        "overlap_f1": "",
        "interruption_precision": "",
        "interruption_recall": "",
        "interruption_f1": "",
        "predicted_turns_path": "",
        "claim_level": "automatic_pyannote_real_or_skipped",
        "notes": f"Skipped automatic pyannote diarization: {reason}",
    }


def _score_prediction(
    row: dict[str, str],
    *,
    model_name: str,
    reference_turns: list[dict[str, Any]],
    reference_events: list[dict[str, Any]],
    predicted_turns: list[dict[str, Any]],
    runtime_seconds: float,
    turns_path: Path,
) -> dict[str, Any]:
    predicted_events = _event_dicts(
        [
            *detect_overlap_events(predicted_turns, clip_id=row["clip_id"]),
            *detect_interruption_events(predicted_turns, clip_id=row["clip_id"]),
        ]
    )
    reference_overlap_count = _event_count(reference_events, "overlap")
    reference_interruption_count = _event_count(reference_events, "interruption")
    overlap = (
        overlap_event_precision_recall_f1(reference_events, predicted_events)
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
    standard = compute_der_jer(
        reference_turns,
        predicted_turns,
        uri=row["clip_id"],
        collar=0.25,
        skip_overlap=False,
    )
    standard_skip = compute_der_jer(
        reference_turns,
        predicted_turns,
        uri=row["clip_id"],
        collar=0.25,
        skip_overlap=True,
    )
    duration = float(row.get("duration_seconds") or 0.0)
    status = (
        "ok"
        if standard["status"] == "ok" and standard_skip["status"] == "ok"
        else "metric_error"
    )
    reason = "; ".join(
        item
        for item in [standard.get("reason", ""), standard_skip.get("reason", "")]
        if item
    )
    return {
        "clip_id": row["clip_id"],
        "dataset_name": row.get("dataset_name", ""),
        "language": row.get("language", ""),
        "model_name": model_name,
        "metric_status": status,
        "duration_seconds": _rounded(duration),
        "runtime_seconds": _rounded(runtime_seconds),
        "rtf": _rounded(runtime_seconds / duration if duration > 0 else ""),
        "num_reference_turns": len(reference_turns),
        "num_predicted_turns": len(predicted_turns),
        "num_reference_speakers": _speaker_count(reference_turns),
        "num_predicted_speakers": _speaker_count(predicted_turns),
        "der": _rounded(standard.get("der")),
        "jer": _rounded(standard.get("jer")),
        "der_skip_overlap": _rounded(standard_skip.get("der")),
        "jer_skip_overlap": _rounded(standard_skip.get("jer")),
        "project_speaker_label_error": _rounded(
            speaker_label_error_rate(reference_turns, predicted_turns)
        ),
        "turn_time_coverage": _rounded(
            turn_time_coverage(reference_turns, predicted_turns)
        ),
        "boundary_mae": _rounded(
            boundary_mean_absolute_error(reference_turns, predicted_turns)
        ),
        "overlap_precision": _rounded(overlap["precision"] if overlap else ""),
        "overlap_recall": _rounded(overlap["recall"] if overlap else ""),
        "overlap_f1": _rounded(overlap["f1"] if overlap else ""),
        "interruption_precision": _rounded(
            interruption["precision"] if interruption else ""
        ),
        "interruption_recall": _rounded(
            interruption["recall"] if interruption else ""
        ),
        "interruption_f1": _rounded(interruption["f1"] if interruption else ""),
        "predicted_turns_path": _display_path(turns_path),
        "claim_level": "automatic_pyannote_real",
        "notes": reason or (
            "Automatic pyannote diarization scored with pyannote.metrics DER/JER; "
            "collar=0.25s, primary DER uses skip_overlap=false."
        ),
    }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    grouped: defaultdict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["dataset_name"]),
                str(row["language"]),
                str(row["model_name"]),
                str(row["metric_status"]),
            )
        ].append(row)
    summaries: list[dict[str, str]] = []
    for (dataset_name, language, model_name, status), group in sorted(grouped.items()):
        ok_group = [row for row in group if row["metric_status"] == "ok"]

        def mean(column: str) -> str:
            values = [
                float(row[column])
                for row in ok_group
                if str(row.get(column, "")).strip()
            ]
            return f"{statistics.fmean(values):.6f}" if values else ""

        summaries.append(
            {
                "dataset_name": dataset_name,
                "language": language,
                "model_name": model_name,
                "metric_status": status,
                "num_clips": str(len(group)),
                "mean_der": mean("der"),
                "mean_jer": mean("jer"),
                "mean_der_skip_overlap": mean("der_skip_overlap"),
                "mean_jer_skip_overlap": mean("jer_skip_overlap"),
                "mean_project_speaker_label_error": mean(
                    "project_speaker_label_error"
                ),
                "mean_overlap_f1": mean("overlap_f1"),
                "mean_rtf": mean("rtf"),
                "notes": (
                    "Means include only metric_status=ok rows; skipped rows "
                    "are counted separately."
                ),
            }
        )
    return summaries


def run_benchmark(
    *,
    manifest: str | Path,
    output: str | Path,
    summary_output: str | Path | None = None,
    predictions_dir: str | Path = "outputs/diarization/pyannote_real",
    model_name: str = DEFAULT_DIARIZATION_MODEL,
    include_single_speaker: bool = False,
    max_clips: int | None = None,
) -> list[dict[str, Any]]:
    manifest_path = resolve_repo_path(str(manifest))
    output_path = resolve_repo_path(str(output))
    prediction_root = resolve_repo_path(str(predictions_dir))
    settings = get_settings()
    pyannote_audio_available = _module_available("pyannote.audio")
    pyannote_metrics_available = _module_available("pyannote.metrics")

    rows: list[dict[str, Any]] = []
    selected: list[dict[str, str]] = []
    for row in _read_manifest(manifest_path):
        if row.get("download_status") not in {"downloaded", "prepared"}:
            continue
        if not include_single_speaker and int(row.get("speaker_count") or 0) < 2:
            continue
        selected.append(row)
        if max_clips is not None and len(selected) >= max_clips:
            break

    if not settings.hf_token:
        reason = "HF_TOKEN is not configured."
    elif not pyannote_audio_available:
        reason = "pyannote.audio is not installed."
    elif not pyannote_metrics_available:
        reason = "pyannote.metrics is not installed."
    else:
        reason = ""

    for row in selected:
        reference = load_reference_evidence(row)
        reference_turns = reference["speaker_turns"]
        reference_events = reference["events"]
        if not reference_turns:
            rows.append(
                _skipped_row(
                    row,
                    model_name=model_name,
                    reason="reference speaker anchors are missing.",
                )
            )
            continue
        if reason:
            rows.append(_skipped_row(row, model_name=model_name, reason=reason))
            continue
        audio_path = resolve_repo_path(row["audio_path"])
        try:
            started = perf_counter()
            diarization = diarize_with_metadata(
                audio_path,
                mock=False,
                hf_token=settings.hf_token,
                model_name=model_name,
                fallback_to_mock=False,
                duration_seconds=float(row.get("duration_seconds") or 0.0),
            )
            runtime_seconds = perf_counter() - started
            predicted_turns = list(diarization["turns"])
            turns_path = prediction_root / f"{row['clip_id']}_pyannote_turns.json"
            _write_json(
                turns_path,
                {
                    "clip_id": row["clip_id"],
                    "model_name": model_name,
                    "runtime_seconds": runtime_seconds,
                    "turns": predicted_turns,
                },
            )
            rows.append(
                _score_prediction(
                    row,
                    model_name=model_name,
                    reference_turns=reference_turns,
                    reference_events=reference_events,
                    predicted_turns=predicted_turns,
                    runtime_seconds=runtime_seconds,
                    turns_path=turns_path,
                )
            )
        except Exception as exc:
            rows.append(_skipped_row(row, model_name=model_name, reason=str(exc)))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=OUTPUT_COLUMNS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    if summary_output is not None:
        summary_path = resolve_repo_path(str(summary_output))
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=SUMMARY_COLUMNS,
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(summarize(rows))
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=Path("outputs/diarization/pyannote_real"),
    )
    parser.add_argument("--model-name", default=DEFAULT_DIARIZATION_MODEL)
    parser.add_argument("--include-single-speaker", action="store_true")
    parser.add_argument("--max-clips", type=int)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        rows = run_benchmark(
            manifest=args.manifest,
            output=args.output,
            summary_output=args.summary_output,
            predictions_dir=args.predictions_dir,
            model_name=args.model_name,
            include_single_speaker=args.include_single_speaker,
            max_clips=args.max_clips,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Pyannote diarization benchmark failed: {exc}", file=sys.stderr)
        return 2
    ok = sum(row["metric_status"] == "ok" for row in rows)
    skipped = sum(row["metric_status"] == "skipped" for row in rows)
    errors = sum(row["metric_status"] == "metric_error" for row in rows)
    print(
        f"Wrote {len(rows)} pyannote benchmark rows: {args.output} "
        f"(ok={ok}, skipped={skipped}, metric_error={errors})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
