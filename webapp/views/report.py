"""Detective report preview and export view."""

from __future__ import annotations

from typing import Any

import streamlit as st

from webapp.data_loader import discover_charts
from webapp.detective_ui import page_header, require_map
from webapp.report_export import build_detective_report, export_detective_report


def render_export(conversation_map: dict[str, Any]) -> None:
    page_header(
        "Export / Report Preview",
        "Generate a Markdown case report that preserves evidence modes, review flags, term candidates, and correction audits.",
    )
    if not require_map(conversation_map):
        return
    chart_paths = list(discover_charts().values())
    report = build_detective_report(conversation_map, chart_paths)
    preview, source = st.tabs(["Rendered preview", "Markdown source"])
    with preview:
        st.markdown(report)
    with source:
        st.code(report, language="markdown")

    action_columns = st.columns([1, 1, 2])
    if action_columns[0].button(
        "Export local report",
        type="primary",
        width="stretch",
    ):
        path = export_detective_report(conversation_map, chart_paths=chart_paths)
        st.session_state["last_report_path"] = str(path)
    action_columns[1].download_button(
        "Download Markdown",
        data=report,
        file_name=f"{conversation_map.get('clip_id', 'clip')}_detective_report.md",
        mime="text/markdown",
        width="stretch",
    )
    if st.session_state.get("last_report_path"):
        action_columns[2].success(
            f"Saved to `{st.session_state['last_report_path']}`"
        )
