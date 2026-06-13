"""TalkWeaver AI Meeting Detective Streamlit entrypoint."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from webapp.components.project_layout import (
    apply_detective_style,
    render_evidence_disclaimer,
)
from webapp.data_loader import (
    conversation_map_label,
    get_best_available_demo_clip,
    list_available_conversation_maps,
    load_conversation_map,
)
from webapp.views.dashboard import render_dashboard
from webapp.views.home import render_crime_scene, render_home
from webapp.views.report import render_export
from webapp.views.safety import (
    render_overlap,
    render_term_rescue,
    render_watchdog,
)
from webapp.views.speakers import render_speaker_cards
from webapp.views.timeline import render_timeline


PAGES = (
    "Home / Project Story",
    "Conversation Crime Scene",
    "Speaker Timeline Detective",
    "Speaker Evidence Cards",
    "Cross-talk and Overlap Warning",
    "Misheard Word Rescue",
    "Hallucination Watchdog",
    "Evidence Dashboard",
    "Export / Report Preview",
)


def _select_conversation_map() -> dict[str, Any]:
    paths = list_available_conversation_maps()
    best = get_best_available_demo_clip()
    if not paths:
        return load_conversation_map(None)
    default_index = paths.index(best) if best in paths else 0
    labels = {path: conversation_map_label(path) for path in paths}
    selected = st.sidebar.selectbox(
        "Investigation artifact",
        paths,
        index=default_index,
        format_func=lambda path: labels[path],
    )
    return load_conversation_map(selected)


def _render_sidebar() -> tuple[str, dict[str, Any]]:
    st.sidebar.markdown("## TalkWeaver")
    st.sidebar.caption("AI Meeting Detective")
    page = st.sidebar.radio("Investigation view", PAGES)
    st.sidebar.divider()
    conversation_map = _select_conversation_map()
    st.sidebar.divider()
    st.sidebar.markdown("**Evidence types**")
    st.sidebar.markdown(
        '<span class="tw-source-real">Real public data</span><br>'
        '<span class="tw-source-oracle">Oracle/reference-assisted</span><br>'
        '<span class="tw-source-controlled">Controlled safety fixture</span>',
        unsafe_allow_html=True,
    )
    st.sidebar.caption(
        "No API key is required. The frontend reads existing local artifacts "
        "and does not execute models."
    )
    return page, conversation_map


def main() -> None:
    st.set_page_config(
        page_title="TalkWeaver: AI Meeting Detective",
        page_icon="TW",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_detective_style()
    page, conversation_map = _render_sidebar()
    render_evidence_disclaimer()

    renderers: dict[str, Callable[[dict[str, Any]], None]] = {
        "Home / Project Story": render_home,
        "Conversation Crime Scene": render_crime_scene,
        "Speaker Timeline Detective": render_timeline,
        "Speaker Evidence Cards": render_speaker_cards,
        "Cross-talk and Overlap Warning": render_overlap,
        "Misheard Word Rescue": render_term_rescue,
        "Hallucination Watchdog": render_watchdog,
        "Evidence Dashboard": render_dashboard,
        "Export / Report Preview": render_export,
    }
    renderers[page](conversation_map)


if __name__ == "__main__":
    main()
