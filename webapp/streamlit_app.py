"""TalkWeaver Streamlit review workspace."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.pipeline import run_pipeline
from webapp.components.speaker_timeline import render_speaker_timeline
from webapp.components.transcript_viewer import render_transcript
from webapp.components.waveform_viewer import render_waveform_placeholder


st.set_page_config(
    page_title="TalkWeaver",
    page_icon="TW",
    layout="wide",
)

st.title("TalkWeaver")
st.caption(
    "Overlap-aware multi-speaker ASR with diarization-structured correction"
)

status_column, action_column = st.columns([3, 1])
with status_column:
    st.subheader("Pipeline Workspace")
    st.write(
        "Inspect speaker-time segments, overlap uncertainty, glossary terms, "
        "and correction audit trails."
    )
with action_column:
    run_mock = st.button(
        "Run mock pipeline",
        type="primary",
        width="stretch",
    )

if run_mock:
    with st.spinner("Running deterministic mock pipeline..."):
        st.session_state["talkweaver_result"] = run_pipeline(mock=True)

result = st.session_state.get("talkweaver_result")

if result is None:
    st.info(
        "No pipeline result is loaded. Run mock mode here or use the Upload "
        "and Pipeline pages from the sidebar."
    )
    architecture_path = ROOT_DIR / "assets" / "architecture.png"
    if architecture_path.exists():
        st.image(
            architecture_path,
            caption="TalkWeaver research pipeline",
            width="stretch",
        )
    render_waveform_placeholder()
else:
    segments = result["transcript"]
    metric_columns = st.columns(4)
    metric_columns[0].metric("Mode", result["mode"])
    metric_columns[1].metric("Segments", len(segments))
    metric_columns[2].metric(
        "Speakers",
        len({segment["speaker"] for segment in segments}),
    )
    metric_columns[3].metric(
        "Overlap regions",
        len(result["overlap_regions"]),
    )

    st.subheader("Speaker Timeline")
    render_speaker_timeline(segments)

    st.subheader("Transcript Audit")
    render_transcript(segments)

    if result.get("summary") is not None:
        with st.expander("Summary and action items"):
            st.write(result["summary"]["summary"])
            for item in result["summary"]["action_items"]:
                st.write(
                    f"- {item['text']} "
                    f"({item['speaker']}, {item['start']:.2f}s)"
                )

    st.warning(result["warning"])
