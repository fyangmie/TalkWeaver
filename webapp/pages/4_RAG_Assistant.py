"""Auxiliary domain-term recovery and meeting-understanding page."""

from __future__ import annotations

import streamlit as st


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
    if result.get("summary") is None:
        st.info(
            "RAG term recovery and meeting understanding are not run in "
            "Phase 3. The current transcript contains empty retrieved-term "
            "and correction fields by design."
        )
    else:
        st.write(result["summary"]["summary"])
        for item in result["summary"]["action_items"]:
            st.checkbox(item, value=False)

    question = st.text_input("Question about this transcript")
    if question:
        st.info(
            "Transcript-grounded QA is a later-phase feature. "
            "No answer is generated in the Phase 1 placeholder."
        )
