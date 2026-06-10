"""Audio upload page."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from webapp.components.audio_player import render_audio_player
from webapp.components.waveform_viewer import render_waveform_placeholder


st.set_page_config(page_title="Upload | TalkWeaver", layout="wide")
st.title("Audio Input")
st.caption("Local files remain in the current Streamlit session.")

uploaded = st.file_uploader(
    "Meeting audio",
    type=["wav", "mp3", "m4a", "flac"],
)
render_audio_player(uploaded)

st.subheader("Signal Preview")
render_waveform_placeholder()

if uploaded is not None:
    st.session_state["uploaded_audio_name"] = uploaded.name
    st.info(
        "The upload is ready for UI review. Real decoding and preprocessing "
        "will be implemented in Phase 2."
    )
