"""Audio upload, persistence, playback, and waveform inspection."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.pipeline import run_pipeline
from webapp.components.audio_player import (
    get_audio_metadata,
    render_audio_metadata,
    render_audio_player,
    save_uploaded_audio,
)
from webapp.components.project_layout import (
    apply_project_style,
    render_page_header,
    render_project_sidebar,
)
from webapp.components.waveform_viewer import render_waveform


st.set_page_config(page_title="Upload | TalkWeaver", page_icon="TW", layout="wide")
apply_project_style()
render_project_sidebar("Audio Upload")
render_page_header(
    "Audio Input",
    "Load meeting audio for preprocessing, ASR, diarization, and overlap analysis.",
)

uploaded = st.file_uploader(
    "Meeting audio",
    type=["wav", "mp3", "m4a", "flac", "ogg"],
    help="The file is saved under outputs/uploads and remains local.",
)

if uploaded is not None:
    try:
        saved_path = save_uploaded_audio(
            uploaded,
            ROOT_DIR / "outputs" / "uploads",
        )
        metadata = get_audio_metadata(saved_path)
    except Exception as exc:
        st.error(f"Unable to save or decode the upload: {exc}")
    else:
        st.session_state["talkweaver_audio_path"] = str(saved_path)
        st.session_state["talkweaver_audio_metadata"] = metadata
        st.session_state["talkweaver_execution_mode"] = "Real audio"
        st.success(f"Audio saved to {saved_path}")

audio_path = st.session_state.get("talkweaver_audio_path")
metadata = st.session_state.get("talkweaver_audio_metadata")

if audio_path:
    st.markdown("### Playback")
    render_audio_player(audio_path)
    if metadata:
        render_audio_metadata(metadata)
    st.markdown("### Signal Preview")
    render_waveform(audio_path)
    st.info(
        "The Pipeline page will convert this recording to normalized mono "
        "16 kHz audio before ASR."
    )
else:
    st.info(
        "No file is loaded. Use the deterministic built-in meeting to review "
        "the full diarization and overlap workflow without audio dependencies."
    )
    if st.button("Load mock/demo meeting", type="primary"):
        with st.spinner("Preparing deterministic mock outputs..."):
            st.session_state["talkweaver_result"] = run_pipeline(mock=True)
            st.session_state["talkweaver_execution_mode"] = "Mock / demo"
        st.success("Mock/demo meeting is ready for review.")
        st.rerun()
