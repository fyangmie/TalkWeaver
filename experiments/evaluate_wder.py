#!/usr/bin/env python3
"""Speaker-attribution error placeholder for aligned labels."""

from __future__ import annotations

import argparse


def speaker_label_error(
    reference_labels: list[str],
    hypothesis_labels: list[str],
) -> float:
    """Return positional speaker-label error for equal-length demo sequences."""

    if len(reference_labels) != len(hypothesis_labels):
        raise ValueError("Reference and hypothesis label counts must match.")
    if not reference_labels:
        return 0.0
    errors = sum(
        reference != hypothesis
        for reference, hypothesis in zip(reference_labels, hypothesis_labels)
    )
    return errors / len(reference_labels)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", nargs="+")
    parser.add_argument("--hypothesis", nargs="+")
    args = parser.parse_args()
    if args.reference is None or args.hypothesis is None:
        parser.error("Provide --reference and --hypothesis speaker labels.")
    print(
        "SpeakerLabelError="
        f"{speaker_label_error(args.reference, args.hypothesis):.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
