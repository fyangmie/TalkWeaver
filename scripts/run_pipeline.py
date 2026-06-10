#!/usr/bin/env python3
"""Run the TalkWeaver pipeline."""

from __future__ import annotations

import argparse
import json
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
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = run_pipeline(audio_path=args.audio, mock=args.mock)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"TalkWeaver error: {exc}", file=sys.stderr)
        return 2

    summary = {
        "mode": result["mode"],
        "segments": len(result["transcript"]),
        "overlap_regions": len(result["overlap_regions"]),
        "artifacts": result["artifacts"],
        "warning": result["warning"],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
