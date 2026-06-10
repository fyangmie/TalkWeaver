"""TalkWeaver Streamlit research review workspace."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.pipeline import run_pipeline
from webapp.components.project_layout import (
    PROJECT_SUBTITLE,
    PROJECT_TITLE,
    apply_project_style,
    render_project_sidebar,
)
from webapp.components.speaker_timeline import render_speaker_timeline
from webapp.components.transcript_viewer import render_transcript_comparison
from webapp.components.waveform_viewer import render_waveform


st.set_page_config(page_title="TalkWeaver", page_icon="TW", layout="wide")
apply_project_style()
render_project_sidebar("Overview")

st.title(PROJECT_TITLE)
st.subheader(PROJECT_SUBTITLE)
st.write(
    "A research-oriented review workspace for who spoke, when they spoke, "
    "where cross-speech occurred, and how constrained correction changed "
    "the ASR evidence."
)

status_column, action_column = st.columns([3, 1])
with status_column:
    st.markdown("### Current Pipeline Result")
    st.write(
        "Use deterministic mock mode for a dependency-free demonstration, "
        "or upload audio and run the real pipeline with labeled fallbacks."
    )
with action_column:
    run_mock = st.button(
        "Run mock pipeline",
        type="primary",
        width="stretch",
    )

if run_mock:
    with st.status("Running the deterministic research pipeline...") as status:
        st.write("Generating ASR, diarization, overlap, and alignment outputs.")
        result = run_pipeline(mock=True)
        st.session_state["talkweaver_result"] = result
        st.session_state["talkweaver_execution_mode"] = "Mock / demo"
        status.update(label="Mock pipeline completed", state="complete")

result = st.session_state.get("talkweaver_result")
if result is None:
    st.info(
        "No result is loaded. Run mock mode above or use the Upload and "
        "Pipeline pages."
    )
    architecture_path = ROOT_DIR / "assets" / "architecture.png"
    if architecture_path.exists():
        st.image(
            architecture_path,
            caption="TalkWeaver research pipeline",
            width="stretch",
        )
    render_waveform(None)
else:
    segments = result.get("transcript", [])
    speakers = {
        speaker
        for segment in segments
        for speaker in segment.get("speakers", [])
    }
    metric_columns = st.columns(5)
    metric_columns[0].metric("Execution", result.get("mode", "unknown"))
    metric_columns[1].metric("ASR segments", len(result.get("asr_segments", [])))
    metric_columns[2].metric("Speakers", len(speakers))
    metric_columns[3].metric(
        "Overlap regions",
        len(result.get("overlap_regions", [])),
    )
    metric_columns[4].metric(
        "Corrected anchors",
        len(segments),
    )

    if str(result.get("mode", "")).startswith("mock"):
        st.warning(
            "This workspace currently shows deterministic mock/demo output, "
            "not measured model performance."
        )
    else:
        st.info(result.get("warning", "Pipeline result loaded."))

    audio_path = result.get("audio_path")
    if audio_path:
        st.markdown("### Audio Evidence")
        render_waveform(audio_path)

    st.markdown("### Speaker and Cross-Speech Timeline")
    render_speaker_timeline(
        result.get("speaker_turns", []),
        result.get("overlap_regions", []),
    )

    st.markdown("### Correction Audit")
    render_transcript_comparison(segments)

    summary = result.get("summary") or {}
    with st.expander("Secondary summary and action items"):
        st.write(summary.get("summary", "No summary is available."))
        action_items = summary.get("action_items", [])
        if not action_items:
            st.caption("No explicit action items were detected.")
        for item in action_items:
            st.write(
                f"- {item['text']} "
                f"({item['speaker']}, {float(item['start']):.2f}s)"
            )
