"""Audio upload and playback component."""

from __future__ import annotations

from typing import Any

import streamlit as st


def render_audio_player(uploaded_file: Any | None) -> None:
    """Render an uploaded audio file or a concise empty state."""

    if uploaded_file is None:
        st.info("Upload a WAV, MP3, M4A, or FLAC file to inspect it.")
        return
    st.audio(uploaded_file)
    st.caption(
        f"{uploaded_file.name} | {uploaded_file.size / (1024 * 1024):.2f} MB"
    )
