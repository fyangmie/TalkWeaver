"""Speaker Timeline Detective view."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from webapp.components.speaker_timeline import render_anchor_timeline
from webapp.data_loader import ROOT_DIR
from webapp.detective_ui import (
    anchor_table,
    page_header,
    render_public_correction_notice,
    require_map,
)
from webapp.ui_components import render_text_diff


def render_timeline(conversation_map: dict[str, Any]) -> None:
    page_header(
        "Speaker Timeline Detective",
        "Follow timestamped evidence across speakers; overlap is shaded and review flags are outlined.",
    )
    if not require_map(conversation_map):
        return
    render_public_correction_notice(conversation_map)
    anchors = list(conversation_map.get("anchors", []))
    speakers = sorted(
        {
            str(item)
            for anchor in anchors
            for item in (
                anchor.get("speakers")
                or [anchor.get("speaker", "UNKNOWN")]
            )
        }
    )
    filter_columns = st.columns([2, 1, 1])
    selected_speakers = filter_columns[0].multiselect(
        "Speakers",
        speakers,
        default=speakers,
    )
    overlap_only = filter_columns[1].toggle("Overlap only", value=False)
    review_only = filter_columns[2].toggle("Needs review only", value=False)

    filtered = [
        anchor
        for anchor in anchors
        if (
            set(anchor.get("speakers") or [anchor.get("speaker", "UNKNOWN")])
            & set(selected_speakers)
        )
        and (not overlap_only or anchor.get("overlap"))
        and (not review_only or anchor.get("needs_review"))
    ]
    render_anchor_timeline(filtered)
    table = anchor_table(filtered)
    if table.empty:
        st.info("No anchors match the current filters.")
        return
    st.dataframe(
        table[
            [
                "start",
                "end",
                "speaker",
                "speakers",
                "raw_text",
                "corrected_text",
                "overlap",
                "needs_review",
                "confidence",
            ]
        ],
        width="stretch",
        hide_index=True,
        column_config={
            "start": st.column_config.NumberColumn(format="%.2f s"),
            "end": st.column_config.NumberColumn(format="%.2f s"),
            "confidence": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0),
        },
    )

    selected_id = st.selectbox(
        "Inspect one anchor",
        table["anchor_id"].tolist(),
        format_func=lambda value: (
            f"{value} | "
            f"{table.loc[table['anchor_id'] == value, 'start'].iloc[0]:.2f}s"
        ),
    )
    anchor = next(item for item in filtered if item.get("anchor_id") == selected_id)
    render_text_diff(
        anchor.get("raw_text", ""),
        anchor.get("corrected_text", ""),
        corrected_label="Corrected / retained anchor text",
    )
    st.caption(
        f"Overlap={bool(anchor.get('overlap'))} | "
        f"Needs review={bool(anchor.get('needs_review'))} | "
        f"Retrieved terms={', '.join(anchor.get('retrieved_terms', [])) or 'none'}"
    )

    audio_path = conversation_map.get("metadata", {}).get("audio_path")
    if audio_path:
        candidate = (ROOT_DIR / Path(str(audio_path))).resolve()
        if candidate.is_file() and ROOT_DIR in candidate.parents:
            with st.expander("Local audio evidence"):
                st.audio(candidate)
