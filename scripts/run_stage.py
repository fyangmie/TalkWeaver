#!/usr/bin/env python3
"""Run one TalkWeaver pipeline stage."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.alignment import align_segments
from backend.asr import transcribe_with_metadata
from backend.config import get_settings
from backend.diarization import diarize_with_metadata
from backend.export import (
    export_diarization,
    export_overlap_regions,
    export_raw_transcript,
    export_temporal_anchor_transcript,
    read_json,
)
from backend.overlap import detect_overlap_regions
from backend.preprocessing import preprocess_audio


STAGES = (
    "preprocess",
    "preprocessing",
    "asr",
    "diarization",
    "align",
    "overlap",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument(
        "--denoise",
        action="store_true",
        help="Apply noisereduce when the optional package is installed.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output WAV path for the preprocessing stage.",
    )
    parser.add_argument(
        "--asr-json",
        type=Path,
        help="Existing raw ASR JSON for the align stage.",
    )
    parser.add_argument(
        "--diarization-json",
        type=Path,
        help="Existing speaker-turn JSON for align or overlap.",
    )
    parser.add_argument("--model-size", help="faster-whisper model size.")
    parser.add_argument("--language", help="Optional ASR language code.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--compute-type", default="default")
    return parser


def _asr_stage(args: argparse.Namespace) -> dict[str, Any]:
    settings = get_settings()
    model_size = args.model_size or settings.asr_model_size
    result = transcribe_with_metadata(
        args.audio,
        mock=args.mock,
        model_size=model_size,
        language=args.language,
        device=args.device,
        compute_type=args.compute_type,
        fallback_to_mock=True,
    )
    stem = (
        "mock_asr"
        if args.mock or args.audio is None
        else f"{args.audio.stem}_raw_asr"
    )
    paths = export_raw_transcript(
        settings.output_dir / "transcripts",
        stem,
        result,
    )
    return {
        **result,
        "artifacts": {name: str(path) for name, path in paths.items()},
    }


def _diarization_stage(args: argparse.Namespace) -> dict[str, Any]:
    settings = get_settings()
    result = diarize_with_metadata(
        args.audio,
        mock=args.mock or settings.use_mock_diarization,
        hf_token=settings.hf_token,
        fallback_to_mock=True,
    )
    stem = (
        "mock_diarization"
        if result["mode"].startswith("mock")
        else f"{args.audio.stem}_diarization"
    )
    paths = export_diarization(
        settings.output_dir / "diarization",
        stem,
        result,
    )
    return {
        **result,
        "artifacts": {name: str(path) for name, path in paths.items()},
    }


def _ensure_asr_json(args: argparse.Namespace) -> Path:
    if args.asr_json is not None:
        if not args.asr_json.exists():
            raise FileNotFoundError(f"ASR JSON not found: {args.asr_json}")
        return args.asr_json

    settings = get_settings()
    default_path = settings.output_dir / "transcripts" / "mock_asr.json"
    if args.mock:
        if not default_path.exists():
            payload = _asr_stage(args)
            return Path(payload["artifacts"]["json"])
        return default_path

    if args.audio is None:
        raise ValueError(
            "Provide --asr-json or --audio outside mock alignment mode."
        )
    inferred = (
        settings.output_dir
        / "transcripts"
        / f"{args.audio.stem}_raw_asr.json"
    )
    if not inferred.exists():
        raise FileNotFoundError(
            f"ASR JSON not found: {inferred}. Run the ASR stage first."
        )
    return inferred


def _ensure_diarization_json(
    args: argparse.Namespace,
) -> tuple[Path, str]:
    if args.diarization_json is not None:
        if not args.diarization_json.exists():
            raise FileNotFoundError(
                f"Diarization JSON not found: {args.diarization_json}"
            )
        return args.diarization_json, "artifact"

    settings = get_settings()
    default_path = (
        settings.output_dir / "diarization" / "mock_diarization.json"
    )
    if args.mock:
        if not default_path.exists():
            payload = _diarization_stage(args)
            return Path(payload["artifacts"]["json"]), payload["mode"]
        return default_path, "mock_demo"

    if args.audio is None:
        raise ValueError(
            "Provide --diarization-json or --audio outside mock mode."
        )
    inferred = (
        settings.output_dir
        / "diarization"
        / f"{args.audio.stem}_diarization.json"
    )
    if not inferred.exists():
        payload = _diarization_stage(args)
        return Path(payload["artifacts"]["json"]), payload["mode"]
    return inferred, "artifact"


def _overlap_stage(args: argparse.Namespace) -> dict[str, Any]:
    settings = get_settings()
    diarization_path, mode = _ensure_diarization_json(args)
    speaker_turns = read_json(diarization_path)
    regions = detect_overlap_regions(speaker_turns)
    paths = export_overlap_regions(
        settings.output_dir / "diarization",
        regions,
        mode=mode,
    )
    return {
        "mode": mode,
        "speaker_turns_path": str(diarization_path),
        "regions": regions,
        "artifacts": {name: str(path) for name, path in paths.items()},
    }


def _align_stage(args: argparse.Namespace) -> dict[str, Any]:
    settings = get_settings()
    asr_path = _ensure_asr_json(args)
    diarization_path, mode = _ensure_diarization_json(args)
    asr_segments = read_json(asr_path)
    speaker_turns = read_json(diarization_path)
    overlap_regions = detect_overlap_regions(speaker_turns)
    transcript = align_segments(asr_segments, speaker_turns)

    stem = "mock" if args.mock else asr_path.stem.removesuffix("_raw_asr")
    temporal_paths = export_temporal_anchor_transcript(
        settings.output_dir / "transcripts",
        stem,
        transcript,
        mode=mode,
    )
    overlap_paths = export_overlap_regions(
        settings.output_dir / "diarization",
        overlap_regions,
        mode=mode,
    )
    paths = {
        "temporal_anchor_json": temporal_paths["json"],
        "speaker_transcript_markdown": temporal_paths["markdown"],
        "overlap_regions_json": overlap_paths["json"],
        "overlap_warnings_markdown": overlap_paths["markdown"],
    }
    return {
        "mode": mode,
        "asr_path": str(asr_path),
        "diarization_path": str(diarization_path),
        "segments": transcript,
        "overlap_regions": overlap_regions,
        "artifacts": {name: str(path) for name, path in paths.items()},
    }


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    stage = "preprocess" if args.stage == "preprocessing" else args.stage

    try:
        if stage == "preprocess":
            payload = preprocess_audio(
                args.audio,
                mock=args.mock,
                denoise=args.denoise,
                output_path=args.output,
            )
        elif stage == "asr":
            payload = _asr_stage(args)
        elif stage == "diarization":
            payload = _diarization_stage(args)
        elif stage == "align":
            payload = _align_stage(args)
        else:
            payload = _overlap_stage(args)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"TalkWeaver stage error: {exc}", file=sys.stderr)
        return 2

    print(json.dumps({"stage": stage, "output": payload}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
