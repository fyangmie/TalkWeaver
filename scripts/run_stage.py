#!/usr/bin/env python3
"""Run one TalkWeaver pipeline stage."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.asr import transcribe
from backend.diarization import diarize
from backend.preprocessing import preprocess_audio


STAGES = ("preprocessing", "asr", "diarization")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--mock", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.stage == "preprocessing":
            payload = preprocess_audio(args.audio, mock=args.mock)
        elif args.stage == "asr":
            payload = transcribe(args.audio, mock=args.mock)
        else:
            payload = diarize(args.audio, mock=args.mock)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"TalkWeaver stage error: {exc}", file=sys.stderr)
        return 2

    print(json.dumps({"stage": args.stage, "output": payload}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
