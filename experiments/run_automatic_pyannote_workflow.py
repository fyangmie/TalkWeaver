#!/usr/bin/env python3
"""Build automatic TalkWeaver evidence maps from ASR + pyannote outputs."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config import get_settings  # noqa: E402
from backend.conversation_map import save_conversation_map  # noqa: E402
from backend.workflow_variants import build_workflow_variant  # noqa: E402
from experiments.prediction_loader import find_and_load_prediction  # noqa: E402


DETAIL_COLUMNS = [
    "clip_id",
    "dataset_name",
    "language",
    "variant",
    "asr_model",
    "diarization_model",
    "uses_real_asr_prediction",
    "uses_automatic_diarization",
    "num_pyannote_turns",
    "num_anchors",
    "num_speaker_labeled_anchors",
    "num_overlap_anchors",
    "num_events",
    "num_term_candidates",
    "num_correction_audits",
    "num_unsupported_changes",
    "num_needs_review",
    "asr_metric_name",
    "asr_error_rate",
    "conversation_map_path",
    "notes",
]

SUMMARY_COLUMNS = [
    "variant",
    "dataset_name",
    "language",
    "asr_model",
    "diarization_model",
    "num_clips",
    "mean_num_pyannote_turns",
    "mean_num_anchors",
    "mean_num_speaker_labeled_anchors",
    "mean_num_overlap_anchors",
    "mean_num_events",
    "mean_num_term_candidates",
    "mean_num_correction_audits",
    "mean_num_unsupported_changes",
    "mean_num_needs_review",
    "mean_asr_error_rate",
    "notes",
]


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def _repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_pyannote_turns(
    predictions_dir: Path,
    clip_id: str,
) -> tuple[str, list[dict[str, Any]]]:
    path = predictions_dir / f"{clip_id}_pyannote_turns.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    turns = [
        {
            "start": float(turn["start"]),
            "end": float(turn["end"]),
            "speaker": str(turn["speaker"]),
            "confidence": float(turn.get("confidence", 0.0) or 0.0),
            "source": "automatic_pyannote",
        }
        for turn in payload.get("turns", [])
        if float(turn.get("end", 0.0)) > float(turn.get("start", 0.0))
    ]
    return str(payload.get("model_name", "pyannote")), turns


def _speaker_labeled_count(anchors: list[Any]) -> int:
    count = 0
    for anchor in anchors:
        speakers = anchor.speakers or [anchor.speaker]
        if any(speaker not in {"UNKNOWN", "OVERLAP"} for speaker in speakers):
            count += 1
    return count


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["variant"]),
            str(row["dataset_name"]),
            str(row["language"]),
            str(row["asr_model"]),
            str(row["diarization_model"]),
        )
        grouped.setdefault(key, []).append(row)
    summaries: list[dict[str, Any]] = []
    numeric_fields = [
        "num_pyannote_turns",
        "num_anchors",
        "num_speaker_labeled_anchors",
        "num_overlap_anchors",
        "num_events",
        "num_term_candidates",
        "num_correction_audits",
        "num_unsupported_changes",
        "num_needs_review",
        "asr_error_rate",
    ]
    for key, group in sorted(grouped.items()):
        variant, dataset_name, language, asr_model, diarization_model = key
        summary: dict[str, Any] = {
            "variant": variant,
            "dataset_name": dataset_name,
            "language": language,
            "asr_model": asr_model,
            "diarization_model": diarization_model,
            "num_clips": len(group),
            "notes": (
                "Automatic TalkWeaver evidence maps built from fixed real ASR "
                "predictions and automatic pyannote turns; no reference "
                "speaker-time is used in the map construction."
            ),
        }
        for field in numeric_fields:
            values = [float(row[field]) for row in group]
            summary[f"mean_{field}"] = round(statistics.fmean(values), 6)
        summaries.append(summary)
    return summaries


def run_workflow(
    *,
    manifest: str | Path,
    asr_predictions_dir: str | Path,
    pyannote_predictions_dir: str | Path,
    asr_model: str,
    output: str | Path,
    summary_output: str | Path,
    maps_dir: str | Path,
    max_clips: int | None = None,
) -> list[dict[str, Any]]:
    manifest_path = _repo_path(manifest)
    asr_root = _repo_path(asr_predictions_dir)
    pyannote_root = _repo_path(pyannote_predictions_dir)
    output_path = _repo_path(output)
    summary_path = _repo_path(summary_output)
    map_root = _repo_path(maps_dir)
    settings = get_settings()
    rows = _read_manifest(manifest_path)
    if max_clips is not None:
        rows = rows[: max(0, max_clips)]

    details: list[dict[str, Any]] = []
    for row in rows:
        prediction = find_and_load_prediction(asr_root, asr_model, row["clip_id"])
        if prediction is None:
            print(
                f"Skipping {row['clip_id']}: missing {asr_model} prediction.",
                file=sys.stderr,
            )
            continue
        pyannote_path = pyannote_root / f"{row['clip_id']}_pyannote_turns.json"
        if not pyannote_path.is_file():
            print(
                f"Skipping {row['clip_id']}: missing pyannote turns.",
                file=sys.stderr,
            )
            continue
        diarization_model, pyannote_turns = _load_pyannote_turns(
            pyannote_root,
            row["clip_id"],
        )
        conversation_map = build_workflow_variant(
            "full_talkweaver",
            {
                **row,
                "clip_id": row["clip_id"],
                "asr_model": asr_model,
                "asr_prediction_json": _display_path(prediction.source_path),
                "diarization_prediction_json": _display_path(
                    pyannote_root / f"{row['clip_id']}_pyannote_turns.json"
                ),
            },
            prediction.segments,
            pyannote_turns,
            [],
            settings.knowledge_base_dir,
            {"use_api": False},
        )
        conversation_map.metadata.update(
            {
                "variant": "automatic_pyannote_evidence_map",
                "diarization_mode": "pyannote",
                "uses_reference_speaker_time": False,
                "reference_assisted": False,
                "uses_automatic_diarization": True,
                "claim_scope": (
                    "Automatic TalkWeaver evidence map over fixed real ASR "
                    "and automatic pyannote diarization; reference transcript "
                    "is used only for ASR scoring outside map construction."
                ),
            }
        )
        map_path = save_conversation_map(
            conversation_map,
            map_root / "automatic_pyannote_evidence_map",
        )
        details.append(
            {
                "clip_id": row["clip_id"],
                "dataset_name": row["dataset_name"],
                "language": row["language"],
                "variant": "automatic_pyannote_evidence_map",
                "asr_model": asr_model,
                "diarization_model": diarization_model,
                "uses_real_asr_prediction": "true",
                "uses_automatic_diarization": "true",
                "num_pyannote_turns": len(pyannote_turns),
                "num_anchors": len(conversation_map.anchors),
                "num_speaker_labeled_anchors": _speaker_labeled_count(
                    conversation_map.anchors
                ),
                "num_overlap_anchors": sum(
                    anchor.overlap for anchor in conversation_map.anchors
                ),
                "num_events": len(conversation_map.events),
                "num_term_candidates": len(conversation_map.term_rescues),
                "num_correction_audits": len(
                    conversation_map.correction_audits
                ),
                "num_unsupported_changes": sum(
                    len(audit.unsupported_changes)
                    for audit in conversation_map.correction_audits
                ),
                "num_needs_review": sum(
                    anchor.needs_review
                    for anchor in conversation_map.anchors
                ),
                "asr_metric_name": prediction.metric_name,
                "asr_error_rate": round(prediction.error_rate, 6),
                "conversation_map_path": _display_path(map_path),
                "notes": (
                    "Automatic map uses pyannote turns and detected overlap/"
                    "interruption events; no oracle speaker-time turns."
                ),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=DETAIL_COLUMNS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(details)

    summaries = _summarize(details) if details else []
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=SUMMARY_COLUMNS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(summaries)
    return details


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--asr-predictions-dir", type=Path, required=True)
    parser.add_argument("--pyannote-predictions-dir", type=Path, required=True)
    parser.add_argument("--asr-model", default="base")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--maps-dir", type=Path, required=True)
    parser.add_argument("--max-clips", type=int)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        rows = run_workflow(
            manifest=args.manifest,
            asr_predictions_dir=args.asr_predictions_dir,
            pyannote_predictions_dir=args.pyannote_predictions_dir,
            asr_model=args.asr_model,
            output=args.output,
            summary_output=args.summary_output,
            maps_dir=args.maps_dir,
            max_clips=args.max_clips,
        )
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"Automatic pyannote workflow failed: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote {len(rows)} automatic pyannote workflow rows: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
