#!/usr/bin/env python3
"""Plot controlled technical-term recovery and correction safety results."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


VARIANT_ORDER = [
    "no_retrieval",
    "exact_glossary",
    "fuzzy",
    "phonetic_like",
    "fused",
    "fused_plus_rule_correction",
    "fused_plus_llm_correction",
]


def _load(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Controlled term rescue CSV contains no rows.")
    return rows


def _mean_by_variant(
    rows: list[dict[str, str]],
    column: str,
) -> dict[str, float]:
    grouped: defaultdict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["variant"]].append(float(row[column]))
    return {
        variant: sum(values) / len(values)
        for variant, values in grouped.items()
    }


def _count_lists_by_variant(
    rows: list[dict[str, str]],
    column: str,
) -> dict[str, int]:
    counts: defaultdict[str, int] = defaultdict(int)
    for row in rows:
        counts[row["variant"]] += len(json.loads(row[column] or "[]"))
    return dict(counts)


def _labels(variants: list[str]) -> list[str]:
    return [variant.replace("_plus_", "\n+ ").replace("_", " ") for variant in variants]


def plot_results(
    input_path: str | Path,
    output_dir: str | Path,
) -> list[Path]:
    """Generate term F1, false-positive, and text-error-delta charts."""

    import matplotlib.pyplot as plt

    rows = _load(input_path)
    variants = [
        variant
        for variant in VARIANT_ORDER
        if any(row["variant"] == variant for row in rows)
    ]
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    colors = ["#4464AD", "#4C956C", "#F4A261", "#8E6C88", "#2A9D8F", "#D1495B", "#6A4C93"]

    f1 = _mean_by_variant(rows, "term_f1")
    figure, axis = plt.subplots(figsize=(12, 6.2))
    axis.bar(
        variants,
        [f1.get(variant, 0.0) for variant in variants],
        color=colors[: len(variants)],
    )
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Mean term F1")
    axis.set_title("Controlled Technical-Term Recovery by Variant")
    axis.set_xticks(range(len(variants)), labels=_labels(variants), rotation=20, ha="right")
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    f1_path = destination / "term_rescue_f1_by_variant.png"
    figure.savefig(f1_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    false_positives = _count_lists_by_variant(rows, "false_positive_terms")
    figure, axis = plt.subplots(figsize=(12, 6.2))
    axis.bar(
        variants,
        [false_positives.get(variant, 0) for variant in variants],
        color=colors[: len(variants)],
    )
    axis.set_ylabel("False-positive retrieved terms")
    axis.set_title("Controlled Retrieval False Positives")
    axis.set_xticks(range(len(variants)), labels=_labels(variants), rotation=20, ha="right")
    axis.grid(axis="y", alpha=0.2)
    false_positive_path = (
        destination / "term_rescue_false_positive_by_variant.png"
    )
    figure.tight_layout()
    figure.savefig(false_positive_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    before = _mean_by_variant(rows, "text_error_before")
    after = _mean_by_variant(rows, "text_error_after")
    delta = {
        variant: before.get(variant, 0.0) - after.get(variant, 0.0)
        for variant in variants
    }
    figure, axis = plt.subplots(figsize=(12, 6.2))
    axis.bar(
        variants,
        [delta[variant] for variant in variants],
        color=colors[: len(variants)],
    )
    axis.axhline(0, color="#333333", linewidth=1)
    axis.set_ylabel("Mean error-rate reduction")
    axis.set_title("Controlled Correction Error Reduction")
    axis.set_xticks(range(len(variants)), labels=_labels(variants), rotation=20, ha="right")
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    delta_path = destination / "term_rescue_error_delta.png"
    figure.savefig(delta_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return [f1_path, false_positive_path, delta_path]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        outputs = plot_results(args.input, args.output_dir)
    except (FileNotFoundError, ImportError, KeyError, ValueError, json.JSONDecodeError) as exc:
        print(f"Controlled term plotting failed: {exc}")
        return 2
    for output in outputs:
        print(f"Wrote chart: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
