"""Pipeline execution controls with dependency-safe fallbacks."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.pipeline import run_pipeline
from webapp.components.project_layout import (
    apply_project_style,
    render_page_header,
    render_project_sidebar,
)


STAGES = (
    ("preprocessing", "Preprocessing"),
    ("asr", "ASR"),
    ("diarization", "Diarization"),
    ("overlap", "Overlap detection"),
    ("rag", "RAG retrieval"),
    ("correction", "LLM correction"),
    ("summary", "Summary"),
)


st.set_page_config(page_title="Pipeline | TalkWeaver", page_icon="TW", layout="wide")
apply_project_style()
render_project_sidebar("Pipeline")
render_page_header(
    "Pipeline",
    "Run the integrated overlap-aware ASR pipeline and inspect its artifacts.",
)

default_mode = st.session_state.get("talkweaver_execution_mode", "Mock / demo")
if default_mode not in ("Mock / demo", "Real audio"):
    default_mode = "Mock / demo"
mode = st.segmented_control(
    "Execution mode",
    options=["Mock / demo", "Real audio"],
    default=default_mode,
)

audio_path = st.session_state.get("talkweaver_audio_path")
if mode == "Real audio":
    if audio_path:
        st.info(f"Input audio: {audio_path}")
    else:
        st.warning("Upload an audio file on the Audio Input page first.")
else:
    st.info(
        "Mock mode requires no GPU, model downloads, HF_TOKEN, or LLM API key."
    )

st.markdown("### Stages")
stage_columns = st.columns(3)
selected: dict[str, bool] = {}
for index, (key, label) in enumerate(STAGES):
    selected[key] = stage_columns[index % 3].toggle(label, value=True)

denoise = st.toggle(
    "Optional denoising",
    value=False,
    disabled=not selected["preprocessing"],
)
st.caption(
    "TalkWeaver preserves the integrated dependency chain. Stage toggles "
    "control the requested review plan; required intermediate artifacts may "
    "still be generated for downstream stages."
)

run_clicked = st.button("Run Pipeline", type="primary")
if run_clicked:
    if mode == "Real audio" and not audio_path:
        st.error("Real mode requires an uploaded audio path.")
    elif not any(selected.values()):
        st.error("Select at least one pipeline stage.")
    else:
        progress = st.progress(0, text="Preparing pipeline inputs")
        try:
            with st.status("Running TalkWeaver pipeline...", expanded=True) as status:
                st.write(
                    "Executing preprocessing, ASR, diarization, alignment, "
                    "overlap detection, retrieval, correction, and summary."
                )
                progress.progress(20, text="Models and fallbacks initialized")
                result = run_pipeline(
                    audio_path=audio_path if mode == "Real audio" else None,
                    mock=mode == "Mock / demo",
                    denoise=denoise,
                )
                progress.progress(100, text="Artifacts exported")
                st.session_state["talkweaver_result"] = result
                st.session_state["talkweaver_execution_mode"] = mode
                st.session_state["talkweaver_stage_selection"] = selected
                status.update(label="Pipeline completed", state="complete")
        except Exception as exc:
            progress.empty()
            st.error(f"Pipeline failed: {exc}")
        else:
            st.success("Pipeline outputs are ready for transcript review.")

result = st.session_state.get("talkweaver_result")
if result:
    st.markdown("### Execution Status")
    asr_mode = result.get("asr", {}).get("mode", "unknown")
    diarization_mode = result.get("diarization", {}).get("mode", "unknown")
    correction_modes = sorted(
        {
            str(segment.get("correction_mode", "unknown"))
            for segment in result.get("transcript", [])
        }
    )
    status_rows = []
    active_selection = st.session_state.get(
        "talkweaver_stage_selection",
        {key: True for key, _label in STAGES},
    )
    for key, label in STAGES:
        detail = "complete"
        if key == "asr":
            detail = asr_mode
        elif key == "diarization":
            detail = diarization_mode
        elif key == "correction":
            detail = ", ".join(correction_modes) or "no segments"
        status_rows.append(
            {
                "stage": label,
                "requested": bool(active_selection.get(key, True)),
                "status": detail,
            }
        )
    st.dataframe(
        pd.DataFrame(status_rows),
        width="stretch",
        hide_index=True,
        column_config={"requested": st.column_config.CheckboxColumn()},
    )
    st.warning(result.get("warning", "Inspect component metadata before evaluation."))

    st.markdown("### Output Artifacts")
    artifact_rows = [
        {"artifact": name, "path": path}
        for name, path in result.get("artifacts", {}).items()
    ]
    st.dataframe(pd.DataFrame(artifact_rows), width="stretch", hide_index=True)
