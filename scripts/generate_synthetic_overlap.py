#!/usr/bin/env python3
"""Prepare metadata for a controlled synthetic-overlap sample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    if not args.mock:
        parser.error(
            "Waveform mixing is scheduled for Phase 3. Run with --mock now."
        )

    output = ROOT_DIR / "data" / "synthetic" / "mock_overlap_plan.json"
    payload = {
        "mode": "mock_demo",
        "sample_rate": 16000,
        "speaker_a": {"start": 0.0, "end": 4.0},
        "speaker_b": {"start": 2.5, "end": 6.0},
        "expected_overlap": {"start": 2.5, "end": 4.0},
        "note": "Metadata only; no synthetic audio has been generated.",
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote mock overlap plan: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
