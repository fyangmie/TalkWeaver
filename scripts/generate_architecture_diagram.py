#!/usr/bin/env python3
"""Generate the TalkWeaver architecture figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT_DIR / "assets" / "architecture.png"

STAGES = [
    ("Meeting\nAudio", "#DCEAF7"),
    ("Preprocess", "#DCEAF7"),
    ("ASR", "#DCEAF7"),
    ("Diarization", "#DCEAF7"),
    ("Alignment", "#DCEAF7"),
    ("Overlap +\nConfidence", "#F8E3A3"),
    ("Temporal-\nAnchor JSON", "#E3E4E8"),
    ("RAG Terms", "#DCEEDC"),
    ("Constrained\nCorrection", "#F4D6D2"),
    ("Review +\nEvaluation", "#E3E4E8"),
]


def main() -> int:
    figure, axis = plt.subplots(figsize=(14, 5.4))
    axis.set_xlim(0, 10)
    axis.set_ylim(0, 4.2)
    axis.axis("off")

    positions = [
        (0.25, 2.65),
        (2.15, 2.65),
        (4.05, 2.65),
        (5.95, 2.65),
        (7.85, 2.65),
        (7.85, 0.75),
        (5.95, 0.75),
        (4.05, 0.75),
        (2.15, 0.75),
        (0.25, 0.75),
    ]

    for index, ((label, color), (x, y)) in enumerate(zip(STAGES, positions)):
        box = FancyBboxPatch(
            (x, y),
            1.55,
            0.78,
            boxstyle="round,pad=0.04,rounding_size=0.08",
            linewidth=1.4,
            edgecolor="#24333F",
            facecolor=color,
        )
        axis.add_patch(box)
        axis.text(
            x + 0.775,
            y + 0.39,
            label,
            ha="center",
            va="center",
            fontsize=10,
            color="#18242C",
        )
        if index < len(positions) - 1:
            next_x, next_y = positions[index + 1]
            if y == next_y and next_x > x:
                start = (x + 1.55, y + 0.39)
                end = (next_x, next_y + 0.39)
            elif y == next_y:
                start = (x, y + 0.39)
                end = (next_x + 1.55, next_y + 0.39)
            elif index == 4:
                start = (x + 0.775, y)
                end = (next_x + 0.775, next_y + 0.78)
            axis.annotate(
                "",
                xy=end,
                xytext=start,
                arrowprops={
                    "arrowstyle": "->",
                    "color": "#52616B",
                    "linewidth": 1.5,
                },
            )

    axis.text(
        0.25,
        3.85,
        "TalkWeaver: overlap-aware multi-speaker ASR research pipeline",
        fontsize=17,
        fontweight="bold",
        color="#18242C",
    )
    axis.text(
        0.25,
        3.55,
        (
            "RAG supports domain-term recovery; diarization, overlap, and "
            "auditable correction remain the core."
        ),
        fontsize=10.5,
        color="#52616B",
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        OUTPUT_PATH,
        dpi=180,
        bbox_inches="tight",
        facecolor="#FFFFFF",
    )
    plt.close(figure)
    print(f"Wrote architecture figure: {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
