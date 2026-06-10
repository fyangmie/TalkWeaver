#!/usr/bin/env python3
"""Compute a simplified temporal speaker-attribution error metric."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def speaker_label_error(
    reference_labels: list[str],
    hypothesis_labels: list[str],
) -> float:
    """Return positional speaker-label error for equal-length sequences."""

    if len(reference_labels) != len(hypothesis_labels):
        raise ValueError("Reference and hypothesis label counts must match.")
    if not reference_labels:
        return 0.0
    errors = sum(
        reference != hypothesis
        for reference, hypothesis in zip(reference_labels, hypothesis_labels)
    )
    return errors / len(reference_labels)


def _intersection(
    first: dict[str, Any],
    second: dict[str, Any],
) -> float:
    return max(
        0.0,
        min(float(first["end"]), float(second["end"]))
        - max(float(first["start"]), float(second["start"])),
    )


def _speaker_set(segment: dict[str, Any]) -> frozenset[str]:
    speakers = segment.get("speakers") or []
    if speakers:
        return frozenset(str(speaker) for speaker in speakers)
    speaker = str(segment.get("speaker", "UNKNOWN"))
    return frozenset() if speaker == "UNKNOWN" else frozenset([speaker])


def align_reference_segments(
    reference_segments: list[dict[str, Any]],
    hypothesis_segments: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any] | None]]:
    """Align each reference anchor to the hypothesis with maximum overlap."""

    aligned = []
    for reference in reference_segments:
        candidates = [
            (
                _intersection(reference, hypothesis),
                bool(reference.get("overlap"))
                == bool(hypothesis.get("overlap")),
                -abs(
                    float(reference["start"])
                    - float(hypothesis["start"])
                )
                - abs(
                    float(reference["end"])
                    - float(hypothesis["end"])
                ),
                -index,
                hypothesis,
            )
            for index, hypothesis in enumerate(hypothesis_segments)
        ]
        candidates = [item for item in candidates if item[0] > 0]
        best = max(candidates, default=None, key=lambda item: item[:4])
        aligned.append((reference, None if best is None else best[4]))
    return aligned


def simplified_wder(
    reference_segments: list[dict[str, Any]],
    hypothesis_segments: list[dict[str, Any]],
) -> float:
    """Return duration-weighted speaker error over temporal anchors.

    This project-level approximation compares active speaker sets after
    temporal-overlap alignment. It is not a full DER/WDER implementation and
    does not perform collar handling, word alignment, or global label mapping.
    """

    if not reference_segments:
        return 0.0 if not hypothesis_segments else 1.0
    error_duration = 0.0
    total_duration = 0.0
    for reference, hypothesis in align_reference_segments(
        reference_segments,
        hypothesis_segments,
    ):
        duration = max(
            0.0,
            float(reference["end"]) - float(reference["start"]),
        )
        total_duration += duration
        if hypothesis is None or _speaker_set(reference) != _speaker_set(hypothesis):
            error_duration += duration
    return error_duration / total_duration if total_duration else 0.0


def overlap_flag_error(
    reference_segments: list[dict[str, Any]],
    hypothesis_segments: list[dict[str, Any]],
) -> float:
    """Return the fraction of reference anchors with incorrect overlap flags."""

    if not reference_segments:
        return 0.0
    errors = 0
    for reference, hypothesis in align_reference_segments(
        reference_segments,
        hypothesis_segments,
    ):
        predicted = False if hypothesis is None else bool(hypothesis.get("overlap"))
        errors += predicted != bool(reference.get("overlap"))
    return errors / len(reference_segments)


def load_segments(path: str | Path) -> list[dict[str, Any]]:
    """Load temporal anchors from a JSON list or pipeline manifest."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("transcript", "temporal_transcript", "segments"):
            if isinstance(payload.get(key), list):
                return payload[key]
    raise ValueError("Expected a JSON list of temporal-anchor segments.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference",
        nargs="+",
        required=True,
        help="Reference JSON path or positional speaker labels.",
    )
    parser.add_argument(
        "--hypothesis",
        nargs="+",
        required=True,
        help="Hypothesis JSON path or positional speaker labels.",
    )
    args = parser.parse_args()

    reference_path = Path(args.reference[0])
    hypothesis_path = Path(args.hypothesis[0])
    if (
        len(args.reference) == 1
        and len(args.hypothesis) == 1
        and reference_path.exists()
        and hypothesis_path.exists()
    ):
        reference = load_segments(reference_path)
        hypothesis = load_segments(hypothesis_path)
        print(f"SpeakerErrorOrWDER={simplified_wder(reference, hypothesis):.4f}")
        print(f"OverlapError={overlap_flag_error(reference, hypothesis):.4f}")
        print("MetricType=simplified_temporal_anchor_approximation")
        return 0

    print(
        "SpeakerErrorOrWDER="
        f"{speaker_label_error(args.reference, args.hypothesis):.4f}"
    )
    print("MetricType=positional_label_error")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
