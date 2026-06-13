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
DETECTIVE_TITLE = "TalkWeaver: AI Meeting Detective"
DETECTIVE_SUBTITLE = (
    "Evidence-grounded conversation maps for chaotic multi-speaker speech"
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


def apply_detective_style() -> None:
    """Apply the investigation-oriented visual language for the v1 app."""

    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1480px;
            padding-top: 1.4rem;
            padding-bottom: 3rem;
        }
        [data-testid="stSidebarNav"] {display: none;}
        [data-testid="stSidebar"] {
            border-right: 1px solid #D9E1E4;
        }
        [data-testid="stMetric"] {
            background: #FFFFFF;
            border: 1px solid #D9E1E4;
            border-left: 4px solid #137C72;
            border-radius: 6px;
            padding: 0.75rem 0.9rem;
            min-height: 108px;
        }
        [data-testid="stMetricValue"] {font-size: 1.55rem;}
        .tw-case-header {
            border-bottom: 1px solid #D9E1E4;
            padding-bottom: 0.8rem;
            margin-bottom: 1rem;
        }
        .tw-kicker {
            color: #137C72;
            font-size: 0.76rem;
            font-weight: 750;
            letter-spacing: 0;
            text-transform: uppercase;
        }
        .tw-evidence-band {
            background: #F3F7F6;
            border-left: 4px solid #137C72;
            border-radius: 4px;
            padding: 0.8rem 1rem;
            margin: 0.7rem 0 1rem 0;
        }
        .tw-warning-band {
            background: #FFF8E7;
            border-left: 4px solid #D7A62A;
            border-radius: 4px;
            padding: 0.8rem 1rem;
            margin: 0.7rem 0 1rem 0;
        }
        .tw-source-real {
            color: #0B665D;
            font-weight: 700;
        }
        .tw-source-controlled {
            color: #9A5A14;
            font-weight: 700;
        }
        .tw-source-oracle {
            color: #9E3D38;
            font-weight: 700;
        }
        .tw-diff-raw {
            border-left: 4px solid #5B6670;
            padding-left: 0.8rem;
        }
        .tw-diff-corrected {
            border-left: 4px solid #137C72;
            padding-left: 0.8rem;
        }
        .tw-diff-panel {
            background: #FFFFFF;
            border-top: 1px solid #D9E1E4;
            border-right: 1px solid #D9E1E4;
            border-bottom: 1px solid #D9E1E4;
            border-radius: 4px;
            min-height: 82px;
            padding: 0.85rem;
            line-height: 1.75;
        }
        .tw-diff-removed {
            background: #FBE1DF;
            color: #8A2E29;
            border-bottom: 2px solid #C2413B;
            padding: 0.08rem 0.18rem;
        }
        .tw-diff-added {
            background: #DDF1EA;
            color: #0B665D;
            border-bottom: 2px solid #137C72;
            padding: 0.08rem 0.18rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_evidence_disclaimer() -> None:
    """Keep result and claim boundaries visible throughout the app."""

    st.markdown(
        """
        <div class="tw-warning-band">
        <strong>Evidence boundary.</strong> This app uses a small formal public
        subset and controlled safety fixtures. Reference speaker-time evidence
        is oracle-assisted, not automatic diarization. Controlled term and
        overlap fixtures are not real-audio generalization claims.
        </div>
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
