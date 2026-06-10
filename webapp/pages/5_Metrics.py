"""Evaluation dashboard page."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from webapp.components.metrics_dashboard import render_metrics


st.set_page_config(page_title="Metrics | TalkWeaver", layout="wide")
st.title("Experiments and Metrics")
st.caption("Only reference-backed measurements belong in final result charts.")

render_metrics(ROOT_DIR / "experiments" / "results" / "ablation_results.csv")
