#!/usr/bin/env python3
"""Run the TalkWeaver pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, help="Meeting audio path.")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run the deterministic no-GPU demo pipeline.",
    )
    parser.add_argument(
        "--denoise",
        action="store_true",
        help="Apply noisereduce during real preprocessing when installed.",
    )
    parser.add_argument("--model-size", help="faster-whisper model size.")
    parser.add_argument("--language", help="Optional ASR language code.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--compute-type", default="default")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        result = run_pipeline(
            audio_path=args.audio,
            mock=args.mock,
            denoise=args.denoise,
            model_size=args.model_size,
            language=args.language,
            device=args.device,
            compute_type=args.compute_type,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"TalkWeaver error: {exc}", file=sys.stderr)
        return 2

    summary = {
        "mode": result["mode"],
        "asr_segments": len(result["asr_segments"]),
        "temporal_anchor_segments": len(result["temporal_transcript"]),
        "corrected_segments": len(result["transcript"]),
        "overlap_regions": len(result["overlap_regions"]),
        "summary_mode": result["summary"]["mode"],
        "action_items": len(result["summary"]["action_items"]),
        "artifacts": result["artifacts"],
        "warning": result["warning"],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
