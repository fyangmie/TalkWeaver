#!/usr/bin/env python3
"""Run TalkWeaver's temporal-anchor evidence-grounded core workflow."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.asr import transcribe_with_metadata  # noqa: E402
from backend.asr_prediction import load_asr_prediction_json  # noqa: E402
from backend.config import get_settings  # noqa: E402
from backend.conversation_map import (  # noqa: E402
    build_conversation_map,
    save_conversation_map,
)
from backend.diarization import diarize_with_metadata  # noqa: E402
from backend.llm_config import load_llm_config  # noqa: E402
from backend.reference_evidence import (  # noqa: E402
    load_reference_evidence,
    resolve_project_path,
)

def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"false", "0", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        f"Expected true or false, received {value!r}."
    )


def load_manifest_row(manifest: Path, clip_id: str) -> dict[str, str]:
    with manifest.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("clip_id") == clip_id:
                return row
    raise ValueError(f"clip_id {clip_id!r} was not found in {manifest}.")


def _reference_evidence(
    row: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    evidence = load_reference_evidence(row)
    anchors = evidence["anchors"]
    if not anchors:
        raise RuntimeError(
            f"{row['clip_id']} has no reference anchors for assisted mode."
        )
    return (
        {
            "mode": "reference",
            "segments": evidence["asr_segments"],
            "language": row["language"],
        },
        {
            "mode": "reference",
            "turns": evidence["speaker_turns"],
            "is_mock": False,
        },
        evidence["events"],
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--mock-models", action="store_true")
    parser.add_argument(
        "--asr-source",
        choices=["real", "reference", "prediction-json"],
        default="real",
    )
    parser.add_argument("--asr-prediction-json", type=Path)
    parser.add_argument(
        "--diarization-source",
        choices=["real", "reference", "none", "pyannote"],
        default="real",
    )
    parser.add_argument("--asr-model", default="tiny")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument(
        "--vad-filter",
        type=_parse_bool,
        default=True,
        metavar="{true,false}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/conversation_maps"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        row = load_manifest_row(args.manifest, args.clip_id)
        audio_path = resolve_project_path(row["audio_path"])
        if not audio_path.is_file():
            raise FileNotFoundError(
                f"Local audio is missing: {audio_path}. "
                "Run the Phase 2A-REAL acquisition command first."
            )
        reference_asr = reference_diarization = None
        reference_events: list[dict[str, Any]] = []
        if (
            args.asr_source == "reference"
            or args.diarization_source == "reference"
        ):
            (
                reference_asr,
                reference_diarization,
                reference_events,
            ) = _reference_evidence(row)

        if args.mock_models:
            asr_output = transcribe_with_metadata(
                audio_path,
                mock=True,
                model_size=args.asr_model,
            )
            diarization_output = diarize_with_metadata(
                audio_path,
                mock=True,
                duration_seconds=asr_output.get("duration_seconds"),
            )
            provided_events: list[dict[str, Any]] = []
        else:
            if args.asr_source == "reference":
                asr_output = reference_asr
            elif args.asr_source == "prediction-json":
                if args.asr_prediction_json is None:
                    raise ValueError(
                        "--asr-prediction-json is required when "
                        "--asr-source prediction-json is selected."
                    )
                asr_output = load_asr_prediction_json(
                    resolve_project_path(args.asr_prediction_json)
                )
            else:
                asr_output = transcribe_with_metadata(
                    audio_path,
                    mock=False,
                    model_size=args.asr_model,
                    device=args.device,
                    compute_type=args.compute_type,
                    language=row["language"],
                    vad_filter=args.vad_filter,
                    fallback_to_mock=False,
                )
            if args.diarization_source == "reference":
                diarization_output = reference_diarization
                provided_events = reference_events
            elif args.diarization_source == "none":
                diarization_output = {
                    "mode": "none",
                    "turns": [],
                    "is_mock": False,
                    "fallback_reason": None,
                }
                provided_events = []
            else:
                diarization_output = diarize_with_metadata(
                    audio_path,
                    mock=False,
                    hf_token=get_settings().hf_token,
                    fallback_to_mock=False,
                    duration_seconds=float(row["duration_seconds"]),
                )
                provided_events = []

        settings = get_settings()
        correction_mode = (
            "rule_fallback"
            if settings.use_mock_llm
            else "llm_with_rule_fallback"
        )
        runtime_llm_config = load_llm_config(
            correction_mode=correction_mode,
        )
        conversation_map = build_conversation_map(
            {
                **row,
                "clip_id": args.clip_id,
                "audio_path": row["audio_path"],
                "asr_prediction_json": (
                    str(args.asr_prediction_json)
                    if args.asr_prediction_json is not None
                    else ""
                ),
                "evaluation_scope": "small_subset",
            },
            asr_output,
            diarization_output,
            provided_events,
            settings.knowledge_base_dir,
            {
                "correction_mode": correction_mode,
                "runtime_config": runtime_llm_config,
            },
        )
        output_path = save_conversation_map(
            conversation_map, args.output
        )
    except Exception as exc:
        print(f"TalkWeaver workflow failed: {exc}", file=sys.stderr)
        return 2

    review_count = sum(
        anchor.needs_review for anchor in conversation_map.anchors
    )
    print(f"clip_id={conversation_map.clip_id}")
    print(f"anchors={len(conversation_map.anchors)}")
    print(f"events={len(conversation_map.events)}")
    print(f"term_rescues={len(conversation_map.term_rescues)}")
    print(f"needs_review={review_count}")
    print(f"output={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
