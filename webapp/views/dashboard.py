"""Evidence Dashboard view."""

from __future__ import annotations

from typing import Any

import streamlit as st

from webapp.data_loader import (
    load_asr_summary,
    load_speaker_overlap_baseline,
    load_workflow_ablation,
)
from webapp.detective_ui import (
    CHART_GROUPS,
    page_header,
    render_chart_grid,
    show_frame_warning,
)


def render_dashboard(_: dict[str, Any]) -> None:
    page_header(
        "Evidence Dashboard",
        "Measured ASR results, workflow structure, and controlled correction-safety evidence remain separated by claim type.",
    )
    interpretation_columns = st.columns(3)
    interpretation_columns[0].info(
        "**Accuracy:** base improves FLEURS multilingual accuracy; AMI remains unstable on two short clips."
    )
    interpretation_columns[1].info(
        "**Efficiency:** tiny is faster and is the current low-resource/mobile direction."
    )
    interpretation_columns[2].info(
        "**Evidence:** speaker-time and overlap signals expose who/when/review structure beyond ASR."
    )
    second_row = st.columns(2)
    second_row[0].success(
        "**Controlled term rescue:** fused retrieval gives strong fixture recovery without negative-control replacements."
    )
    second_row[1].warning(
        "**Controlled overlap safety:** overlap awareness increases review and conservative rejection where evidence is weak."
    )

    asr = load_asr_summary()
    workflow = load_workflow_ablation()
    speaker = load_speaker_overlap_baseline()
    for frame in (asr, workflow, speaker):
        show_frame_warning(frame)

    if not asr.empty:
        st.subheader("Real public-data ASR baseline")
        st.dataframe(asr, width="stretch", hide_index=True)
    if not speaker.empty:
        st.subheader("Speaker-time and overlap baseline")
        st.caption(
            "Reference-assisted rows are oracle workflow checks, not automatic diarization scores."
        )
        st.dataframe(speaker, width="stretch", hide_index=True)
    if not workflow.empty:
        st.subheader("TalkWeaver workflow ablation")
        st.dataframe(workflow, width="stretch", hide_index=True)

    for heading, chart_names in CHART_GROUPS.items():
        st.subheader(heading)
        render_chart_grid(chart_names)
