"""Shared TalkWeaver page framing and research context."""

from __future__ import annotations

import streamlit as st


PROJECT_TITLE = (
    "TalkWeaver: An Overlap-Aware Multi-Speaker ASR System with "
    "Diarization-Structured LLM Correction"
)
PROJECT_SUBTITLE = (
    "RAG-Enhanced Domain Term Recovery for Noisy Meeting Speech"
)
PIPELINE_TEXT = (
    "Audio preprocessing -> ASR -> diarization -> alignment -> overlap "
    "detection -> RAG retrieval -> structured LLM correction -> summary -> "
    "metrics"
)


def apply_project_style() -> None:
    """Apply restrained styling for the research review workspace."""

    st.markdown(
        """
        <style>
        .block-container {padding-top: 2rem; padding-bottom: 3rem;}
        [data-testid="stMetric"] {
            border-top: 3px solid #176B87;
            padding-top: 0.7rem;
        }
        [data-testid="stSidebar"] hr {margin: 1rem 0;}
        .tw-kicker {
            color: #176B87;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0;
            text-transform: uppercase;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_project_sidebar(active_page: str) -> None:
    """Render the project focus and pipeline sequence in the sidebar."""

    with st.sidebar:
        st.markdown("### TalkWeaver")
        st.caption(PROJECT_SUBTITLE)
        st.markdown(f"**Current view:** {active_page}")
        st.divider()
        st.markdown("**Research pipeline**")
        st.write(PIPELINE_TEXT)
        st.divider()
        st.markdown("**Primary focus**")
        st.write(
            "Speaker diarization, cross-speech detection, temporal alignment, "
            "and constrained LLM + ASR correction."
        )
        st.markdown("**Auxiliary module**")
        st.write(
            "Local RAG retrieves domain terms for correction and secondary "
            "meeting understanding."
        )
        st.caption(
            "Mock/demo outputs are deterministic and are never presented as "
            "experimental measurements."
        )


def render_page_header(section: str, description: str) -> None:
    """Render a consistent page title beneath the project identity."""

    st.markdown('<div class="tw-kicker">TalkWeaver research demo</div>', unsafe_allow_html=True)
    st.title(section)
    st.caption(description)
