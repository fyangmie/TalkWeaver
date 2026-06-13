"""Speaker Timeline Detective view."""

from __future__ import annotations

from typing import Any

import streamlit as st

from webapp.components.speaker_timeline import render_anchor_timeline
from webapp.data_loader import (
    get_event_investigation_rows,
    related_anchors_for_event,
)
from webapp.detective_ui import (
    anchor_table,
    page_header,
    render_public_correction_notice,
    require_map,
)
from webapp.ui_components import render_audio_evidence, render_text_diff


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

    events = get_event_investigation_rows(conversation_map)
    event_options: list[dict[str, Any] | None] = [None, *events]
    selected_event = st.selectbox(
        "Jump to event",
        event_options,
        format_func=lambda event: (
            "No event selected"
            if event is None
            else (
                f"{event.get('event_id', 'event')} | "
                f"{float(event.get('start', 0.0)):.2f}-"
                f"{float(event.get('end', 0.0)):.2f}s | "
                f"{event.get('type', 'event')}"
            )
        ),
        key="timeline_event",
    )
    highlighted_ids: set[str] = set()
    if selected_event is not None:
        highlighted_ids = {
            str(anchor.get("anchor_id"))
            for anchor in related_anchors_for_event(
                conversation_map,
                selected_event,
            )
            if anchor.get("anchor_id")
        }

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
    render_anchor_timeline(filtered, highlighted_ids)
    if selected_event is not None:
        st.caption(
            f"Selected event speakers: "
            f"{', '.join(selected_event.get('speakers', [])) or 'unknown'} | "
            f"related anchors: {', '.join(sorted(highlighted_ids)) or 'none'}"
        )
        render_audio_evidence(
            conversation_map,
            selected_event,
            item_type="event",
            label="Selected event audio",
        )
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
    render_audio_evidence(
        conversation_map,
        anchor,
        item_type="anchor",
        label="Selected anchor audio",
    )
