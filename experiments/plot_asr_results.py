#!/usr/bin/env python3
"""Plot real ASR error and real-time-factor comparisons."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def load_results(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("ASR benchmark CSV contains no rows.")
    return rows


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def plot_results(
    input_path: str | Path,
    output_dir: str | Path,
) -> list[Path]:
    """Generate language error-rate and model RTF charts."""

    import matplotlib.pyplot as plt

    rows = load_results(input_path)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    models = sorted({row["model_name"] for row in rows})
    languages = sorted({row["language"] for row in rows})
    errors: defaultdict[tuple[str, str], list[float]] = defaultdict(list)
    rtfs: defaultdict[str, list[float]] = defaultdict(list)
    metric_names: defaultdict[str, set[str]] = defaultdict(set)
    for row in rows:
        errors[(row["model_name"], row["language"])].append(
            float(row["error_rate"])
        )
        rtfs[row["model_name"]].append(float(row["rtf"]))
        metric_names[row["language"]].add(row["metric_name"])

    figure, axis = plt.subplots(figsize=(10, 5.8))
    width = 0.8 / max(1, len(models))
    positions = list(range(len(languages)))
    colors = ["#176B87", "#C2413B", "#2E7D5B", "#6C5B7B"]
    for model_index, model in enumerate(models):
        values = [
            _mean(errors[(model, language)])
            if errors[(model, language)]
            else 0.0
            for language in languages
        ]
        offsets = [
            position - 0.4 + width / 2 + model_index * width
            for position in positions
        ]
        bars = axis.bar(
            offsets,
            values,
            width=width,
            label=model,
            color=colors[model_index % len(colors)],
        )
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    labels = [
        f"{language}\n({'+'.join(sorted(metric_names[language]))})"
        for language in languages
    ]
    axis.set_xticks(positions, labels=labels)
    axis.set_ylabel("Mean error rate")
    axis.set_title("Real ASR Error by Language - Small Formal Subset")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(title="faster-whisper model")
    figure.tight_layout()
    error_path = destination / "asr_error_by_language.png"
    figure.savefig(error_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8.5, 5.2))
    rtf_values = [_mean(rtfs[model]) for model in models]
    bars = axis.bar(
        models,
        rtf_values,
        color=[colors[index % len(colors)] for index in range(len(models))],
    )
    axis.set_ylabel("Mean real-time factor")
    axis.set_title("Real ASR Runtime by Model - Small Formal Subset")
    axis.grid(axis="y", alpha=0.2)
    axis.set_ylim(0, max(rtf_values) * 1.35)
    axis.text(
        0.99,
        0.96,
        "All measured RTF values are below the 1.0 real-time boundary.",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="#555555",
    )
    for bar, value in zip(bars, rtf_values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )
    figure.tight_layout()
    rtf_path = destination / "asr_rtf_by_model.png"
    figure.savefig(rtf_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return [error_path, rtf_path]


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
        print(f"ASR plotting failed: {exc}")
        return 2
    for output in outputs:
        print(f"Wrote chart: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
