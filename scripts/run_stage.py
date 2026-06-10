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

from backend.asr import transcribe_with_metadata
from backend.config import get_settings
from backend.diarization import diarize
from backend.export import export_raw_transcript
from backend.preprocessing import preprocess_audio


STAGES = ("preprocess", "preprocessing", "asr", "diarization")


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
        else:
            payload = diarize(args.audio, mock=args.mock)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"TalkWeaver stage error: {exc}", file=sys.stderr)
        return 2

    print(json.dumps({"stage": stage, "output": payload}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
