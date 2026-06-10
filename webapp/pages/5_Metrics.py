"""Reference-backed evaluation and chart dashboard."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from webapp.components.metrics_dashboard import render_metrics_dashboard
from webapp.components.project_layout import (
    apply_project_style,
    render_page_header,
    render_project_sidebar,
)


st.set_page_config(page_title="Metrics | TalkWeaver", page_icon="TW", layout="wide")
apply_project_style()
render_project_sidebar("Metrics")
render_page_header(
    "Experiments and Metrics",
    "Reference-backed evaluation for ASR, speaker attribution, overlap, terms, and latency.",
)
st.write(
    "Real metrics require reference transcripts, reference speaker labels, "
    "and measured runtime. Empty mock scaffold rows are intentionally not "
    "converted into performance claims."
)

render_metrics_dashboard(
    result_directories=[
        ROOT_DIR / "experiments" / "results",
        ROOT_DIR / "outputs" / "metrics",
    ],
    chart_directory=ROOT_DIR / "assets" / "result_charts",
)
