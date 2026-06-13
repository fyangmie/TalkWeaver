#!/usr/bin/env python3
"""Run TalkWeaver workflow variants over fixed real ASR predictions."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config import get_settings  # noqa: E402
from backend.conversation_map import save_conversation_map  # noqa: E402
from backend.reference_evidence import (  # noqa: E402
    load_reference_evidence,
    resolve_project_path,
)
from backend.workflow_variants import (  # noqa: E402
    VARIANT_NAMES,
    build_workflow_variant,
)
from experiments.metrics.text_metrics import evaluate_text  # noqa: E402
from experiments.metrics.text_normalization import normalize_for_wer  # noqa: E402
from experiments.prediction_loader import (  # noqa: E402
    find_and_load_prediction,
)


OUTPUT_COLUMNS = [
    "clip_id",
    "dataset_name",
    "language",
    "variant",
    "asr_model",
    "uses_real_asr_prediction",
    "uses_reference_speaker_time",
    "uses_overlap_events",
    "uses_term_rescue",
    "uses_correction",
    "num_anchors",
    "num_speaker_labeled_anchors",
    "num_overlap_anchors",
    "num_events",
    "num_term_candidates",
    "num_term_rescues_applied",
    "num_correction_audits",
    "num_unsupported_changes",
    "num_needs_review",
    "conversation_map_path",
    "asr_error_rate",
    "asr_metric_name",
    "corrected_error_rate",
    "corrected_metric_name",
    "anchor_coverage",
    "speaker_evidence_available",
    "overlap_evidence_available",
    "term_reference_available",
    "term_precision",
    "term_recall",
    "notes",
]


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def load_manifest_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _interval_union(
    intervals: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def _duration(intervals: list[tuple[float, float]]) -> float:
    return sum(end - start for start, end in intervals)


def _coverage(
    segments: list[dict[str, Any]],
    anchors: list[Any],
) -> float:
    source = _interval_union(
        [
            (float(segment["start"]), float(segment["end"]))
            for segment in segments
        ]
    )
    anchor_intervals = _interval_union(
        [(float(anchor.start), float(anchor.end)) for anchor in anchors]
    )
    total = _duration(source)
    if total <= 0:
        return 0.0
    covered = 0.0
    for source_start, source_end in source:
        for anchor_start, anchor_end in anchor_intervals:
            covered += max(
                0.0,
                min(source_end, anchor_end)
                - max(source_start, anchor_start),
            )
    return min(1.0, covered / total)


def _transcript_from_anchors(anchors: list[Any]) -> str:
    return " ".join(
        str(anchor.corrected_text or anchor.raw_text).strip()
        for anchor in anchors
        if str(anchor.corrected_text or anchor.raw_text).strip()
    ).strip()


def _has_applied_correction(anchors: list[Any]) -> bool:
    return any(
        str(anchor.corrected_text).strip()
        and normalize_for_wer(anchor.corrected_text)
        != normalize_for_wer(anchor.raw_text)
        for anchor in anchors
    )


def _reference_terms(row: dict[str, str]) -> list[str]:
    path = row.get("terms_path", "").strip()
    if not path:
        return []
    payload = json.loads(
        resolve_project_path(path).read_text(encoding="utf-8")
    )
    if not isinstance(payload, list):
        raise ValueError(f"Term reference must contain a list: {path}")
    terms: list[str] = []
    for item in payload:
        if isinstance(item, str):
            terms.append(item)
        elif isinstance(item, dict):
            value = item.get("canonical") or item.get("term")
            if value:
                terms.append(str(value))
    return list(dict.fromkeys(terms))


def _term_metrics(
    reference_terms: list[str],
    candidates: list[Any],
) -> tuple[float | str, float | str]:
    if not reference_terms:
        return "", ""
    predicted = {
        normalize_for_wer(candidate.canonical)
        for candidate in candidates
        if normalize_for_wer(candidate.canonical)
    }
    reference = {
        normalize_for_wer(term)
        for term in reference_terms
        if normalize_for_wer(term)
    }
    matches = len(predicted & reference)
    precision = matches / len(predicted) if predicted else 0.0
    recall = matches / len(reference) if reference else 0.0
    return round(precision, 6), round(recall, 6)


def _term_rescues_applied(conversation_map: Any) -> int:
    applied: set[tuple[str, str]] = set()
    candidates_by_anchor: dict[str, list[Any]] = {}
    for candidate in conversation_map.term_rescues:
        for anchor_id in candidate.evidence_anchor_ids:
            candidates_by_anchor.setdefault(anchor_id, []).append(candidate)
    for anchor in conversation_map.anchors:
        raw = normalize_for_wer(anchor.raw_text)
        corrected = normalize_for_wer(anchor.corrected_text)
        if not corrected or corrected == raw:
            continue
        for candidate in candidates_by_anchor.get(anchor.anchor_id, []):
            canonical = normalize_for_wer(candidate.canonical)
            if canonical and canonical in corrected and canonical not in raw:
                applied.add((anchor.anchor_id, candidate.canonical))
    return len(applied)


def _variant_notes(
    variant: str,
    *,
    reference_terms: list[str],
    correction_mode: str,
) -> str:
    notes = [
        "Fixed Phase 2C real ASR prediction; no ASR rerun.",
        "Small-subset workflow ablation.",
    ]
    if variant not in {"asr_only", "temporal_anchor_only"}:
        notes.append(
            "Reference speaker-time is oracle evidence, not automatic "
            "diarization."
        )
    if variant in {"term_rescue", "constrained_correction", "full_talkweaver"}:
        notes.append(
            "No technical-term metric available."
            if not reference_terms
            else "Term precision/recall scored against reference terms."
        )
    if variant in {"constrained_correction", "full_talkweaver"}:
        notes.append(f"Correction mode={correction_mode}.")
    return " ".join(notes)


def _parse_variants(values: list[str]) -> list[str]:
    if values == ["all"] or "all" in values:
        return list(VARIANT_NAMES)
    unknown = [value for value in values if value not in VARIANT_NAMES]
    if unknown:
        raise ValueError(
            "Unknown workflow variant(s): " + ", ".join(unknown)
        )
    return list(dict.fromkeys(values))


def run_ablation(
    *,
    manifest: str | Path,
    predictions_dir: str | Path,
    asr_model: str,
    output: str | Path,
    maps_dir: str | Path,
    variants: list[str],
    dataset: str | None = None,
    max_clips: int | None = None,
) -> list[dict[str, Any]]:
    """Run selected workflow variants over fixed real prediction JSON."""

    manifest_path = resolve_project_path(manifest)
    prediction_root = resolve_project_path(predictions_dir)
    output_path = resolve_project_path(output)
    map_root = resolve_project_path(maps_dir)
    selected_variants = _parse_variants(variants)
    rows = load_manifest_rows(manifest_path)
    if dataset:
        rows = [
            row
            for row in rows
            if row.get("dataset_name", "").casefold()
            == dataset.casefold()
        ]
    if max_clips is not None:
        rows = rows[: max(0, max_clips)]
    settings = get_settings()
    results: list[dict[str, Any]] = []

    for row in rows:
        prediction = find_and_load_prediction(
            prediction_root,
            asr_model,
            row["clip_id"],
        )
        if prediction is None:
            print(
                f"Skipping {row['clip_id']}: no {asr_model} real prediction "
                f"under {prediction_root}.",
                file=sys.stderr,
            )
            continue
        if prediction.clip_id != row["clip_id"]:
            raise ValueError(
                f"Prediction clip mismatch for {row['clip_id']}: "
                f"{prediction.clip_id}"
            )
        reference = load_reference_evidence(row)
        reference_terms = _reference_terms(row)
        for variant in selected_variants:
            conversation_map = build_workflow_variant(
                variant,
                {
                    **row,
                    "clip_id": row["clip_id"],
                    "asr_model": asr_model,
                    "asr_prediction_json": _display_path(
                        prediction.source_path
                    ),
                },
                prediction.segments,
                reference["speaker_turns"],
                reference["events"],
                settings.knowledge_base_dir,
                {"use_api": False},
            )
            if conversation_map.metadata.get("is_mock"):
                raise RuntimeError(
                    f"Variant {variant} produced mock metadata for "
                    f"{row['clip_id']}."
                )
            map_path = save_conversation_map(
                conversation_map,
                map_root / variant,
            )
            uses_correction = bool(
                conversation_map.metadata.get("uses_correction")
            )
            if uses_correction and _has_applied_correction(
                conversation_map.anchors
            ):
                corrected_metrics = evaluate_text(
                    prediction.reference_text,
                    _transcript_from_anchors(conversation_map.anchors),
                    row["language"],
                )
            elif uses_correction:
                corrected_metrics = {
                    "metric_name": prediction.metric_name,
                    "error_rate": prediction.error_rate,
                }
            else:
                corrected_metrics = None
            term_precision, term_recall = _term_metrics(
                reference_terms,
                conversation_map.term_rescues,
            )
            speaker_labeled = sum(
                bool(
                    [
                        speaker
                        for speaker in anchor.speakers
                        if speaker not in {"UNKNOWN", "OVERLAP"}
                    ]
                )
                or anchor.speaker not in {"UNKNOWN", "OVERLAP"}
                for anchor in conversation_map.anchors
            )
            results.append(
                {
                    "clip_id": row["clip_id"],
                    "dataset_name": row["dataset_name"],
                    "language": row["language"],
                    "variant": variant,
                    "asr_model": asr_model,
                    "uses_real_asr_prediction": "true",
                    "uses_reference_speaker_time": str(
                        bool(
                            conversation_map.metadata.get(
                                "uses_reference_speaker_time"
                            )
                        )
                    ).lower(),
                    "uses_overlap_events": str(
                        bool(
                            conversation_map.metadata.get(
                                "uses_overlap_events"
                            )
                        )
                    ).lower(),
                    "uses_term_rescue": str(
                        bool(
                            conversation_map.metadata.get(
                                "uses_term_rescue"
                            )
                        )
                    ).lower(),
                    "uses_correction": str(uses_correction).lower(),
                    "num_anchors": len(conversation_map.anchors),
                    "num_speaker_labeled_anchors": speaker_labeled,
                    "num_overlap_anchors": sum(
                        anchor.overlap
                        for anchor in conversation_map.anchors
                    ),
                    "num_events": len(conversation_map.events),
                    "num_term_candidates": len(
                        conversation_map.term_rescues
                    ),
                    "num_term_rescues_applied": (
                        _term_rescues_applied(conversation_map)
                        if uses_correction
                        else 0
                    ),
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
                    "conversation_map_path": _display_path(map_path),
                    "asr_error_rate": round(prediction.error_rate, 6),
                    "asr_metric_name": prediction.metric_name,
                    "corrected_error_rate": (
                        round(
                            float(corrected_metrics["error_rate"]),
                            6,
                        )
                        if corrected_metrics
                        else ""
                    ),
                    "corrected_metric_name": (
                        corrected_metrics["metric_name"]
                        if corrected_metrics
                        else ""
                    ),
                    "anchor_coverage": round(
                        _coverage(
                            prediction.segments,
                            conversation_map.anchors,
                        ),
                        6,
                    ),
                    "speaker_evidence_available": str(
                        bool(reference["speaker_turns"])
                    ).lower(),
                    "overlap_evidence_available": str(
                        any(
                            event.get("type") == "overlap"
                            for event in reference["events"]
                        )
                    ).lower(),
                    "term_reference_available": str(
                        bool(reference_terms)
                    ).lower(),
                    "term_precision": term_precision,
                    "term_recall": term_recall,
                    "notes": _variant_notes(
                        variant,
                        reference_terms=reference_terms,
                        correction_mode=str(
                            conversation_map.metadata.get(
                                "llm_mode",
                                "disabled",
                            )
                        ),
                    ),
                }
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
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--asr-model", default="base")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maps-dir", type=Path, required=True)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["all"],
        help="'all' or one or more named workflow variants.",
    )
    parser.add_argument("--dataset")
    parser.add_argument("--max-clips", type=int)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        rows = run_ablation(
            manifest=args.manifest,
            predictions_dir=args.predictions_dir,
            asr_model=args.asr_model,
            output=args.output,
            maps_dir=args.maps_dir,
            variants=args.variants,
            dataset=args.dataset,
            max_clips=args.max_clips,
        )
    except (
        FileNotFoundError,
        KeyError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"Workflow ablation failed: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote {len(rows)} workflow ablation rows: {args.output}")
    print(
        "Variants="
        + ", ".join(sorted({str(row["variant"]) for row in rows}))
    )
    print(
        f"Clips={len({str(row['clip_id']) for row in rows})}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
