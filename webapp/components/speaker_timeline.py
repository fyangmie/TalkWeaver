"""Speaker-turn and overlap timeline visualization."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st


SPEAKER_COLORS = (
    "#176B87",
    "#2E7D5B",
    "#8A5A44",
    "#6C5B7B",
    "#4F6D7A",
)
OVERLAP_COLOR = "#C2413B"


def build_timeline_rows(
    speaker_turns: list[dict[str, Any]],
    overlap_regions: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Normalize diarization turns and overlap intervals for display."""

    rows = [
        {
            "start": float(turn["start"]),
            "end": float(turn["end"]),
            "duration": round(float(turn["end"]) - float(turn["start"]), 3),
            "lane": str(turn["speaker"]),
            "speakers": str(turn["speaker"]),
            "type": "speaker_turn",
        }
        for turn in speaker_turns
    ]
    for region in overlap_regions or []:
        speakers = ", ".join(str(item) for item in region.get("speakers", []))
        rows.append(
            {
                "start": float(region["start"]),
                "end": float(region["end"]),
                "duration": round(
                    float(region["end"]) - float(region["start"]),
                    3,
                ),
                "lane": "OVERLAP",
                "speakers": speakers,
                "type": "overlap",
            }
        )
    return sorted(
        rows,
        key=lambda row: (row["start"], row["type"] == "overlap", row["lane"]),
    )


def render_speaker_timeline(
    speaker_turns: list[dict[str, Any]],
    overlap_regions: list[dict[str, Any]] | None = None,
) -> None:
    """Render speaker turns with overlap on a dedicated review lane."""

    rows = build_timeline_rows(speaker_turns, overlap_regions)
    if not rows:
        st.info("No diarization turns are available.")
        return

    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.info("Install matplotlib for the graphical speaker timeline.")
        return

    lanes = list(dict.fromkeys(row["lane"] for row in rows))
    lane_positions = {lane: index for index, lane in enumerate(lanes)}
    speakers = [lane for lane in lanes if lane != "OVERLAP"]
    speaker_colors = {
        speaker: SPEAKER_COLORS[index % len(SPEAKER_COLORS)]
        for index, speaker in enumerate(speakers)
    }

    figure_height = max(2.2, 0.7 * len(lanes) + 1.2)
    figure, axis = plt.subplots(figsize=(11, figure_height))
    for row in rows:
        color = (
            OVERLAP_COLOR
            if row["type"] == "overlap"
            else speaker_colors[row["lane"]]
        )
        axis.barh(
            lane_positions[row["lane"]],
            row["duration"],
            left=row["start"],
            height=0.52,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            alpha=0.92,
        )

    axis.set_yticks(range(len(lanes)), labels=lanes)
    axis.set_xlabel("Time (seconds)")
    axis.set_xlim(0, max(row["end"] for row in rows) * 1.03)
    axis.grid(axis="x", alpha=0.2)
    axis.invert_yaxis()
    legend = [
        Patch(color=speaker_colors[speaker], label=speaker)
        for speaker in speakers
    ]
    if any(row["type"] == "overlap" for row in rows):
        legend.append(Patch(color=OVERLAP_COLOR, label="Overlap review"))
    axis.legend(handles=legend, loc="upper right", frameon=False, ncol=2)
    figure.tight_layout()
    st.pyplot(figure, width="stretch")
    plt.close(figure)

    table = pd.DataFrame(rows)
    st.dataframe(
        table,
        width="stretch",
        hide_index=True,
        column_config={
            "start": st.column_config.NumberColumn(format="%.2f s"),
            "end": st.column_config.NumberColumn(format="%.2f s"),
            "duration": st.column_config.NumberColumn(format="%.2f s"),
        },
    )
