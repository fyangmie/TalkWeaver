#!/usr/bin/env python3
"""Create the TalkWeaver ablation result scaffold."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
RESULT_PATH = ROOT_DIR / "experiments" / "results" / "ablation_results.csv"

GROUPS = [
    ("A", "Whisper only"),
    ("B", "Preprocessing + Whisper"),
    ("C", "Whisper + diarization + alignment"),
    ("D", "Structured LLM correction"),
    ("E", "Structured LLM correction + RAG glossary"),
    ("F", "Overlap-aware correction"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()
    if not args.mock:
        parser.error(
            "Real ablation requires reference data. Run --mock for the scaffold."
        )

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "group",
                "configuration",
                "result_type",
                "wer",
                "speaker_error",
                "term_error_rate",
                "hallucinated_corrections",
                "latency_seconds",
                "note",
            ],
        )
        writer.writeheader()
        for group, configuration in GROUPS:
            writer.writerow(
                {
                    "group": group,
                    "configuration": configuration,
                    "result_type": "mock_demo_not_measured",
                    "wer": "",
                    "speaker_error": "",
                    "term_error_rate": "",
                    "hallucinated_corrections": "",
                    "latency_seconds": "",
                    "note": "Populate from references; do not cite this row.",
                }
            )

    print(f"Wrote mock ablation scaffold: {RESULT_PATH}")
    print("No performance metrics were fabricated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
