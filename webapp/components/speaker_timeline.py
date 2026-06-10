"""Speaker timeline component for temporal-anchor segments."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st


def render_speaker_timeline(segments: list[dict[str, Any]]) -> None:
    """Display segment timing, speakers, and overlap state."""

    if not segments:
        st.info("No speaker segments are available.")
        return
    rows = [
        {
            "start": segment["start"],
            "end": segment["end"],
            "duration": round(segment["end"] - segment["start"], 2),
            "speaker": segment["speaker"],
            "overlap": segment["overlap"],
            "confidence": segment["confidence"],
        }
        for segment in segments
    ]
    st.dataframe(
        pd.DataFrame(rows),
        width="stretch",
        hide_index=True,
        column_config={
            "start": st.column_config.NumberColumn(format="%.2f s"),
            "end": st.column_config.NumberColumn(format="%.2f s"),
            "duration": st.column_config.ProgressColumn(
                min_value=0,
                max_value=max(row["duration"] for row in rows),
                format="%.2f s",
            ),
            "overlap": st.column_config.CheckboxColumn(),
            "confidence": st.column_config.ProgressColumn(
                min_value=0.0,
                max_value=1.0,
                format="%.2f",
            ),
        },
    )
