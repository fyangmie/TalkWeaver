"""Metrics dashboard component."""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import streamlit as st


def render_metrics(result_path: Path) -> None:
    """Render measured metrics or an explicit unmeasured state."""

    metric_columns = st.columns(4)
    for column, label in zip(
        metric_columns,
        ["WER", "Speaker error", "Term error", "Latency"],
    ):
        column.metric(label, "Not measured")

    if not result_path.exists():
        st.info("Run `python experiments/run_ablation.py --mock` first.")
        return

    with result_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.warning(
        "Rows labeled mock_demo_not_measured are experiment scaffolds, "
        "not performance claims."
    )
