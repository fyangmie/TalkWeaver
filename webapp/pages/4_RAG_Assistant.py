"""Auxiliary domain-term recovery and transcript-grounded QA page."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.pipeline import run_pipeline
from backend.summarizer import answer_question
from webapp.components.project_layout import (
    apply_project_style,
    render_page_header,
    render_project_sidebar,
)


st.set_page_config(
    page_title="Domain Terms | TalkWeaver",
    page_icon="TW",
    layout="wide",
)
apply_project_style()
render_project_sidebar("RAG Domain Recovery")
render_page_header(
    "RAG Domain-Term Recovery",
    "A secondary module for glossary-grounded correction and transcript search.",
)
st.info(
    "RAG is not the primary TalkWeaver task. It retrieves local domain terms "
    "to support ASR correction; speaker attribution and overlap handling "
    "remain the core research focus."
)

result = st.session_state.get("talkweaver_result")
if result is None:
    st.info("No pipeline result is loaded.")
    if st.button("Run mock pipeline", type="primary"):
        with st.spinner("Generating deterministic RAG and correction outputs..."):
            st.session_state["talkweaver_result"] = run_pipeline(mock=True)
        st.rerun()
else:
    transcript = result.get("transcript", [])
    retrieval_rows = [
        {
            "start": segment.get("start"),
            "speaker": segment.get("speaker"),
            "raw_text": segment.get("raw_text"),
            "retrieved_terms": ", ".join(segment.get("retrieved_terms", [])),
            "sources": ", ".join(segment.get("retrieval_sources", [])),
            "overlap": segment.get("overlap", False),
        }
        for segment in transcript
    ]
    st.markdown("### Retrieved Domain Terms")
    st.dataframe(
        pd.DataFrame(retrieval_rows),
        width="stretch",
        hide_index=True,
        column_config={
            "start": st.column_config.NumberColumn(format="%.2f s"),
            "overlap": st.column_config.CheckboxColumn(),
        },
    )

    summary = result.get("summary") or {}
    summary_column, actions_column = st.columns([3, 2])
    with summary_column:
        st.markdown("### Extractive Meeting Summary")
        st.write(summary.get("summary", "No summary is available."))
        st.caption(summary.get("note", ""))
    with actions_column:
        st.markdown("### Sourced Action Items")
        action_items = summary.get("action_items", [])
        if not action_items:
            st.caption("No explicit action items were detected.")
        for index, item in enumerate(action_items):
            st.checkbox(
                (
                    f"{item['text']} "
                    f"({item['speaker']}, {float(item['start']):.2f}s)"
                ),
                value=False,
                key=f"action_item_{index}",
            )

    st.markdown("### Transcript-Grounded QA")
    question = st.text_input(
        "Question",
        placeholder="What should the team compare?",
    )
    if question:
        answer = answer_question(question, transcript)
        st.write(answer["answer"])
        source = answer.get("source")
        if source:
            st.caption(
                f"Source: {source['timestamp']} | {source['speaker']} | "
                f"overlap={str(source['overlap']).lower()}"
            )
        else:
            st.warning("No transcript segment supports an answer.")
