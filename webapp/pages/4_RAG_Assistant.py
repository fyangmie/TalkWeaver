"""Auxiliary domain-term recovery and meeting-understanding page."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.summarizer import answer_question


st.set_page_config(page_title="Domain Terms | TalkWeaver", layout="wide")
st.title("Domain-Term Recovery")
st.caption(
    "RAG is an auxiliary ASR correction module, not the main project topic."
)

result = st.session_state.get("talkweaver_result")
if result is None:
    st.info("Run the mock pipeline from the Pipeline page first.")
else:
    terms = sorted(
        {
            term
            for segment in result["transcript"]
            for term in segment["retrieved_terms"]
        }
    )
    st.subheader("Retrieved Terms")
    st.write(", ".join(terms) or "No glossary terms retrieved.")

    st.subheader("Secondary Meeting Understanding")
    st.write(result["summary"]["summary"])
    for item in result["summary"]["action_items"]:
        st.checkbox(
            (
                f"{item['text']} "
                f"({item['speaker']}, {item['start']:.2f}s)"
            ),
            value=False,
        )

    question = st.text_input("Question about this transcript")
    if question:
        answer = answer_question(question, result["transcript"])
        st.write(answer["answer"])
        if answer["source"]:
            st.caption(
                f"{answer['source']['timestamp']} | "
                f"{answer['source']['speaker']}"
            )
