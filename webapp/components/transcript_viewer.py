"""Reusable raw, temporal-anchor, and correction review components."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st


TEMPORAL_FIELDS = (
    "start",
    "end",
    "speaker",
    "speakers",
    "raw_text",
    "corrected_text",
    "overlap",
    "confidence",
    "retrieved_terms",
)


def temporal_anchor_rows(
    segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return stable table rows containing required temporal-anchor fields."""

    rows: list[dict[str, Any]] = []
    for segment in segments:
        row = {field: segment.get(field) for field in TEMPORAL_FIELDS}
        row["start"] = float(row["start"] or 0.0)
        row["end"] = float(row["end"] or 0.0)
        row["speakers"] = ", ".join(row["speakers"] or [])
        row["retrieved_terms"] = ", ".join(row["retrieved_terms"] or [])
        row["confidence"] = float(row["confidence"] or 0.0)
        rows.append(row)
    return rows


def render_raw_asr(segments: list[dict[str, Any]]) -> None:
    """Render timestamped raw ASR segments and optional word timings."""

    if not segments:
        st.info("No raw ASR transcript is available.")
        return
    for index, segment in enumerate(segments, start=1):
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", 0.0))
        with st.expander(
            f"Segment {index} | {start:.2f}-{end:.2f} s",
            expanded=True,
        ):
            st.write(segment.get("text", ""))
            words = segment.get("words", [])
            if words:
                st.dataframe(
                    pd.DataFrame(words),
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "start": st.column_config.NumberColumn(format="%.2f s"),
                        "end": st.column_config.NumberColumn(format="%.2f s"),
                    },
                )


def render_temporal_anchor_table(
    segments: list[dict[str, Any]],
) -> None:
    """Render every required temporal-anchor transcript field."""

    rows = temporal_anchor_rows(segments)
    if not rows:
        st.info("No speaker-attributed temporal anchors are available.")
        return
    st.dataframe(
        pd.DataFrame(rows),
        width="stretch",
        hide_index=True,
        column_config={
            "start": st.column_config.NumberColumn(format="%.2f s"),
            "end": st.column_config.NumberColumn(format="%.2f s"),
            "overlap": st.column_config.CheckboxColumn(),
            "confidence": st.column_config.ProgressColumn(
                min_value=0.0,
                max_value=1.0,
                format="%.2f",
            ),
        },
    )


def render_transcript_comparison(
    segments: list[dict[str, Any]],
) -> None:
    """Render raw and corrected text side by side with uncertainty cues."""

    if not segments:
        st.info("No corrected transcript is available.")
        return

    for segment in segments:
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", 0.0))
        speaker = str(segment.get("speaker", "UNKNOWN"))
        overlap = bool(segment.get("overlap"))
        heading = f"{start:.2f}-{end:.2f} s | {speaker}"
        if overlap:
            heading += " | OVERLAP REVIEW"

        with st.expander(heading, expanded=True):
            if overlap:
                speakers = ", ".join(segment.get("speakers", []))
                st.warning(
                    "Cross-speech detected. Active speakers: "
                    f"{speakers or 'unknown'}. Correction remains uncertain."
                )
            raw_column, corrected_column = st.columns(2)
            raw_column.markdown("**Raw ASR**")
            raw_column.write(segment.get("raw_text", ""))
            corrected_column.markdown("**Constrained correction**")
            corrected_column.write(
                segment.get("corrected_text")
                or segment.get("raw_text", "")
            )
            st.caption(
                f"Confidence {float(segment.get('confidence', 0.0)):.2f} | "
                f"Terms: {', '.join(segment.get('retrieved_terms', [])) or 'none'} "
                f"| Mode: {segment.get('correction_mode', 'not run')}"
            )
            if segment.get("correction_uncertain"):
                st.warning(segment.get("correction_note", "Review required."))


def render_transcript(segments: list[dict[str, Any]]) -> None:
    """Backward-compatible alias for the correction comparison."""

    render_transcript_comparison(segments)
