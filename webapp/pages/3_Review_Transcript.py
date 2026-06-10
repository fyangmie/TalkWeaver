"""Speaker attribution, overlap, and correction audit page."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.pipeline import run_pipeline
from webapp.components.project_layout import (
    apply_project_style,
    render_page_header,
    render_project_sidebar,
)
from webapp.components.speaker_timeline import render_speaker_timeline
from webapp.components.transcript_viewer import (
    render_raw_asr,
    render_temporal_anchor_table,
    render_transcript_comparison,
)


st.set_page_config(
    page_title="Transcript Review | TalkWeaver",
    page_icon="TW",
    layout="wide",
)
apply_project_style()
render_project_sidebar("Transcript Review")
render_page_header(
    "Transcript Review",
    "Audit speaker attribution, cross-speech uncertainty, and constrained corrections.",
)

result = st.session_state.get("talkweaver_result")
if result is None:
    st.info("No pipeline result is loaded.")
    if st.button("Run mock pipeline", type="primary"):
        with st.spinner("Generating deterministic review artifacts..."):
            st.session_state["talkweaver_result"] = run_pipeline(mock=True)
        st.rerun()
else:
    overlap_regions = result.get("overlap_regions", [])
    transcript = result.get("transcript", [])
    columns = st.columns(4)
    columns[0].metric("Raw ASR segments", len(result.get("asr_segments", [])))
    columns[1].metric("Temporal anchors", len(transcript))
    columns[2].metric("Speaker turns", len(result.get("speaker_turns", [])))
    columns[3].metric("Overlap regions", len(overlap_regions))

    st.markdown("### Speaker Timeline")
    render_speaker_timeline(
        result.get("speaker_turns", []),
        overlap_regions,
    )

    if overlap_regions:
        st.warning(
            "Overlap intervals are low-confidence review targets. "
            "TalkWeaver does not silently remove their uncertainty."
        )
        st.dataframe(
            pd.DataFrame(overlap_regions),
            width="stretch",
            hide_index=True,
            column_config={
                "start": st.column_config.NumberColumn(format="%.2f s"),
                "end": st.column_config.NumberColumn(format="%.2f s"),
                "duration": st.column_config.NumberColumn(format="%.2f s"),
            },
        )

    raw_tab, attributed_tab, corrected_tab, json_tab = st.tabs(
        [
            "Raw ASR",
            "Speaker-attributed",
            "Raw vs corrected",
            "Temporal-anchor JSON",
        ]
    )
    with raw_tab:
        render_raw_asr(result.get("asr_segments", []))
    with attributed_tab:
        render_temporal_anchor_table(result.get("temporal_transcript", []))
    with corrected_tab:
        render_transcript_comparison(transcript)
    with json_tab:
        st.json(transcript)
