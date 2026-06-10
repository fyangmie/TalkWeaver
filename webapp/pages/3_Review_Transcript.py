"""Transcript review page."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from webapp.components.speaker_timeline import render_speaker_timeline
from webapp.components.transcript_viewer import render_transcript


st.set_page_config(page_title="Transcript Review | TalkWeaver", layout="wide")
st.title("Transcript Review")

result = st.session_state.get("talkweaver_result")
if result is None:
    st.info("Run the mock pipeline from the Pipeline page first.")
else:
    st.subheader("Speaker Timeline")
    render_speaker_timeline(result["transcript"])
    st.subheader("Raw and Corrected Transcript")
    render_transcript(result["transcript"])
    st.warning(
        "Overlap segments require human review; uncertainty is never removed "
        "silently."
    )
