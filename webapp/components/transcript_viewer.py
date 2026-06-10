"""Transcript comparison component."""

from __future__ import annotations

from typing import Any

import streamlit as st


def render_transcript(segments: list[dict[str, Any]]) -> None:
    """Render raw and corrected temporal-anchor transcript segments."""

    if not segments:
        st.info("Run the mock pipeline to create a transcript.")
        return

    for segment in segments:
        heading = (
            f"{segment['start']:.2f}-{segment['end']:.2f} | "
            f"{segment['speaker']}"
        )
        if segment["overlap"]:
            heading += " | OVERLAP: REVIEW"
        with st.expander(heading, expanded=True):
            raw_column, corrected_column = st.columns(2)
            raw_column.markdown("**Raw ASR**")
            raw_column.write(segment["raw_text"])
            corrected_column.markdown("**Corrected**")
            corrected_column.write(segment["corrected_text"])
            st.caption(
                f"Confidence {segment['confidence']:.2f} | "
                f"Terms: {', '.join(segment['retrieved_terms']) or 'none'}"
            )
