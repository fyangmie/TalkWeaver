#!/usr/bin/env python3
"""Run TalkWeaver's temporal-anchor evidence-grounded core workflow."""

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

from backend.asr import transcribe_with_metadata  # noqa: E402
from backend.config import get_settings  # noqa: E402
from backend.conversation_map import (  # noqa: E402
    build_conversation_map,
    save_conversation_map,
)
from backend.diarization import diarize_with_metadata  # noqa: E402


def _read_json(path: str) -> Any:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def load_manifest_row(manifest: Path, clip_id: str) -> dict[str, str]:
    with manifest.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("clip_id") == clip_id:
                return row
    raise ValueError(f"clip_id {clip_id!r} was not found in {manifest}.")


def _reference_evidence(
    row: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    anchors = _read_json(row["anchors_path"])
    if not anchors:
        raise RuntimeError(
            f"{row['clip_id']} has no reference anchors for assisted mode."
        )
    asr_segments = [
        {
            "start": float(anchor["start"]),
            "end": float(anchor["end"]),
            "text": str(anchor.get("text", "")).strip(),
            "words": [],
        }
        for anchor in anchors
        if str(anchor.get("text", "")).strip()
    ]
    turns = [
        {
            "start": float(anchor["start"]),
            "end": float(anchor["end"]),
            "speaker": str(anchor.get("speaker", "UNKNOWN")),
            "confidence": 1.0,
            "source": "reference_anchor",
        }
        for anchor in anchors
    ]
    events = (
        _read_json(row["events_path"]) if row.get("events_path") else []
    )
    return (
        {
            "mode": "reference",
            "segments": asr_segments,
            "language": row["language"],
        },
        {
            "mode": "reference",
            "turns": turns,
            "is_mock": False,
        },
        events,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--mock-models", action="store_true")
    parser.add_argument(
        "--asr-source",
        choices=["real", "reference"],
        default="real",
    )
    parser.add_argument(
        "--diarization-source",
        choices=["real", "reference"],
        default="real",
    )
    parser.add_argument("--asr-model", default="tiny")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compute-type", default="int8")
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
        audio_path = ROOT / row["audio_path"]
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
            asr_output = (
                reference_asr
                if args.asr_source == "reference"
                else transcribe_with_metadata(
                    audio_path,
                    mock=False,
                    model_size=args.asr_model,
                    device=args.device,
                    compute_type=args.compute_type,
                    language=row["language"],
                    fallback_to_mock=False,
                )
            )
            diarization_output = (
                reference_diarization
                if args.diarization_source == "reference"
                else diarize_with_metadata(
                    audio_path,
                    mock=False,
                    hf_token=get_settings().hf_token,
                    fallback_to_mock=False,
                    duration_seconds=float(row["duration_seconds"]),
                )
            )
            provided_events = reference_events

        settings = get_settings()
        use_api = not settings.use_mock_llm and any(
            (
                settings.openai_api_key,
                settings.deepseek_api_key,
                settings.qwen_api_key,
            )
        )
        conversation_map = build_conversation_map(
            {
                **row,
                "clip_id": args.clip_id,
                "audio_path": row["audio_path"],
            },
            asr_output,
            diarization_output,
            provided_events,
            settings.knowledge_base_dir,
            {
                "use_api": use_api,
                "provider": settings.llm_provider,
                "openai_api_key": settings.openai_api_key,
                "deepseek_api_key": settings.deepseek_api_key,
                "qwen_api_key": settings.qwen_api_key,
                "openai_model": settings.openai_model,
                "deepseek_model": settings.deepseek_model,
                "qwen_model": settings.qwen_model,
                "openai_base_url": settings.openai_base_url,
                "deepseek_base_url": settings.deepseek_base_url,
                "qwen_base_url": settings.qwen_base_url,
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
