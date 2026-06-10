#!/usr/bin/env python3
"""Report latency and real-time factor from measured durations."""

from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-seconds", type=float, required=True)
    parser.add_argument("--runtime-seconds", type=float, required=True)
    args = parser.parse_args()
    if args.audio_seconds <= 0 or args.runtime_seconds < 0:
        parser.error("Durations must be non-negative and audio must be positive.")
    rtf = args.runtime_seconds / args.audio_seconds
    print(f"LatencySeconds={args.runtime_seconds:.4f}")
    print(f"RealTimeFactor={rtf:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
