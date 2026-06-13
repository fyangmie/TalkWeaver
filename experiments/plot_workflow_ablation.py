#!/usr/bin/env python3
"""Plot TalkWeaver workflow completeness and review flags."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


VARIANT_ORDER = [
    "asr_only",
    "temporal_anchor_only",
    "reference_speaker_time",
    "overlap_aware",
    "term_rescue",
    "constrained_correction",
    "full_talkweaver",
]


def _load(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Workflow ablation CSV contains no rows.")
    return rows


def _means(
    rows: list[dict[str, str]],
    column: str,
) -> dict[str, float]:
    grouped: defaultdict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(column, "").strip()
        if value:
            grouped[row["variant"]].append(float(value))
    return {
        variant: sum(values) / len(values)
        for variant, values in grouped.items()
        if values
    }


def plot_results(
    input_path: str | Path,
    output_dir: str | Path,
) -> list[Path]:
    """Generate completeness and correction-review charts."""

    import matplotlib.pyplot as plt

    rows = _load(input_path)
    variants = [
        variant
        for variant in VARIANT_ORDER
        if any(row["variant"] == variant for row in rows)
    ]
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    colors = ["#237A8E", "#D46B4C", "#5B7F45"]

    completeness_columns = [
        ("num_anchors", "Anchors"),
        ("num_speaker_labeled_anchors", "Speaker-labeled anchors"),
        ("num_events", "Events"),
    ]
    figure, axis = plt.subplots(figsize=(12, 6.2))
    width = 0.24
    positions = list(range(len(variants)))
    for index, (column, label) in enumerate(completeness_columns):
        means = _means(rows, column)
        offsets = [
            position + (index - 1) * width for position in positions
        ]
        axis.bar(
            offsets,
            [means.get(variant, 0.0) for variant in variants],
            width=width,
            label=label,
            color=colors[index],
        )
    axis.set_xticks(
        positions,
        labels=[variant.replace("_", "\n") for variant in variants],
    )
    axis.set_ylabel("Mean count per real clip")
    axis.set_title(
        "TalkWeaver Evidence Completeness by Workflow Variant"
    )
    axis.grid(axis="y", alpha=0.2)
    axis.legend()
    figure.tight_layout()
    completeness = destination / "workflow_ablation_completeness.png"
    figure.savefig(completeness, dpi=180, bbox_inches="tight")
    plt.close(figure)

    review_columns = [
        ("num_unsupported_changes", "Unsupported changes"),
        ("num_needs_review", "Anchors needing review"),
    ]
    figure, axis = plt.subplots(figsize=(12, 6.2))
    width = 0.34
    for index, (column, label) in enumerate(review_columns):
        means = _means(rows, column)
        offsets = [
            position + (index - 0.5) * width for position in positions
        ]
        axis.bar(
            offsets,
            [means.get(variant, 0.0) for variant in variants],
            width=width,
            label=label,
            color=colors[index],
        )
    axis.set_xticks(
        positions,
        labels=[variant.replace("_", "\n") for variant in variants],
    )
    axis.set_ylabel("Mean count per real clip")
    axis.set_title("TalkWeaver Review and Correction Audit Flags")
    axis.grid(axis="y", alpha=0.2)
    axis.legend()
    figure.tight_layout()
    review = destination / "workflow_ablation_review_flags.png"
    figure.savefig(review, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return [completeness, review]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        outputs = plot_results(args.input, args.output_dir)
    except (FileNotFoundError, ImportError, KeyError, ValueError) as exc:
        print(f"Workflow ablation plotting failed: {exc}")
        return 2
    for output in outputs:
        print(f"Wrote chart: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
