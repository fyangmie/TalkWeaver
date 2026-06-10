"""Pipeline controls page."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.pipeline import run_pipeline


st.set_page_config(page_title="Pipeline | TalkWeaver", layout="wide")
st.title("Pipeline")

mode = st.segmented_control(
    "Execution mode",
    options=["Mock / demo", "Real models"],
    default="Mock / demo",
)

st.subheader("Stages")
columns = st.columns(3)
preprocessing = columns[0].checkbox("Preprocessing", value=True)
asr = columns[0].checkbox("ASR", value=True)
diarization = columns[1].checkbox("Diarization", value=True)
overlap = columns[1].checkbox("Overlap detection", value=True)
rag = columns[2].checkbox("RAG term recovery", value=True)
llm = columns[2].checkbox("LLM correction", value=True)

selected = {
    "preprocessing": preprocessing,
    "asr": asr,
    "diarization": diarization,
    "overlap": overlap,
    "rag": rag,
    "llm": llm,
}

if st.button("Run pipeline", type="primary"):
    if mode == "Real models":
        st.error(
            "Use the CLI for real audio execution so an audio path "
            "and pyannote credentials can be supplied."
        )
    elif not all(selected.values()):
        st.warning(
            "The mock orchestrator runs preprocessing, ASR, "
            "diarization, overlap detection, alignment, RAG, and correction "
            "together."
        )
    else:
        with st.spinner("Running pipeline..."):
            st.session_state["talkweaver_result"] = run_pipeline(mock=True)
        st.success("Mock pipeline completed.")

result = st.session_state.get("talkweaver_result")
if result:
    st.subheader("Stage Status")
    for stage in [
        "preprocessing",
        "asr",
        "diarization",
        "alignment",
        "overlap",
        "rag",
        "correction",
        "summary",
    ]:
        st.write(f"`{stage}`: complete (mock/demo)")
    st.json(result["artifacts"])
