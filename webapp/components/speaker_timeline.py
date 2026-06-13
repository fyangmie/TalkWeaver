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
REVIEW_COLOR = "#D7A62A"


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


def build_anchor_timeline_rows(
    anchors: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Expand temporal anchors into speaker lanes for investigation views."""

    rows: list[dict[str, Any]] = []
    for anchor in anchors:
        start = float(anchor.get("start", 0.0))
        end = float(anchor.get("end", start))
        speakers = [
            str(item)
            for item in anchor.get("speakers", [])
            if str(item).strip()
        ]
        if not speakers:
            speakers = [str(anchor.get("speaker") or "UNKNOWN")]
        for speaker in speakers:
            rows.append(
                {
                    "anchor_id": str(anchor.get("anchor_id", "")),
                    "start": start,
                    "end": end,
                    "duration": max(0.0, end - start),
                    "lane": speaker,
                    "overlap": bool(anchor.get("overlap")),
                    "needs_review": bool(anchor.get("needs_review")),
                    "confidence": float(anchor.get("confidence", 0.0)),
                    "text": str(
                        anchor.get("corrected_text")
                        or anchor.get("raw_text")
                        or ""
                    ),
                }
            )
    return sorted(rows, key=lambda row: (row["start"], row["lane"]))


def build_anchor_timeline_figure(
    anchors: list[dict[str, Any]],
) -> Any | None:
    """Create a stable matplotlib timeline figure without Streamlit calls."""

    rows = build_anchor_timeline_rows(anchors)
    if not rows:
        return None
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError:
        return None

    lanes = list(dict.fromkeys(row["lane"] for row in rows))
    lane_positions = {lane: index for index, lane in enumerate(lanes)}
    speaker_colors = {
        speaker: SPEAKER_COLORS[index % len(SPEAKER_COLORS)]
        for index, speaker in enumerate(lanes)
    }
    figure_height = max(2.5, 0.72 * len(lanes) + 1.25)
    figure, axis = plt.subplots(figsize=(11.5, figure_height))
    overlap_spans: set[tuple[float, float]] = set()
    for row in rows:
        axis.barh(
            lane_positions[row["lane"]],
            row["duration"],
            left=row["start"],
            height=0.5,
            color=speaker_colors[row["lane"]],
            edgecolor=REVIEW_COLOR if row["needs_review"] else "white",
            linewidth=2.0 if row["needs_review"] else 0.8,
            alpha=0.9,
        )
        if row["overlap"]:
            overlap_spans.add((row["start"], row["end"]))
    for start, end in overlap_spans:
        axis.axvspan(start, end, color=OVERLAP_COLOR, alpha=0.16, hatch="//")

    axis.set_yticks(range(len(lanes)), labels=lanes)
    axis.set_xlabel("Time (seconds)")
    max_end = max(row["end"] for row in rows)
    axis.set_xlim(0, max(max_end * 1.03, 1.0))
    axis.grid(axis="x", color="#D7DEE2", alpha=0.7)
    axis.invert_yaxis()
    legend = [
        Patch(facecolor=speaker_colors[speaker], label=speaker)
        for speaker in lanes
    ]
    if overlap_spans:
        legend.append(Patch(facecolor=OVERLAP_COLOR, alpha=0.3, label="Overlap"))
    if any(row["needs_review"] for row in rows):
        legend.append(Patch(facecolor="white", edgecolor=REVIEW_COLOR, label="Review"))
    axis.legend(handles=legend, loc="upper right", frameon=False, ncol=2)
    figure.tight_layout()
    return figure


def render_anchor_timeline(anchors: list[dict[str, Any]]) -> None:
    """Render temporal anchors as speaker lanes with overlap shading."""

    figure = build_anchor_timeline_figure(anchors)
    if figure is None:
        rows = build_anchor_timeline_rows(anchors)
        if rows:
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            st.info("Install matplotlib for the graphical anchor timeline.")
        else:
            st.info("No temporal anchors are available.")
        return
    st.pyplot(figure, width="stretch")
    try:
        import matplotlib.pyplot as plt

        plt.close(figure)
    except ImportError:
        pass


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
