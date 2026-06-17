"""Focused MVP views for the PRD evidence board."""

from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd
import streamlit as st

from webapp.data_loader import (
    load_overlap_safety_summary,
    load_term_rescue_summary,
    load_workflow_ablation,
)
from webapp.detective_ui import (
    anchor_table,
    list_value,
    map_stats,
    number,
    page_header,
    require_map,
    safe_html,
    show_frame_warning,
)
from webapp.ui_components import render_text_diff


STATUS_ORDER = (
    "supported",
    "weakly_supported",
    "unsupported",
    "needs_review",
)
TERM_STATUS_ORDER = (
    "successful_rescue",
    "false_rescue",
    "missed_rescue",
    "needs_review",
)


def _source_label(metadata: dict[str, Any]) -> str:
    source_type = str(metadata.get("source_type", "unknown"))
    if source_type == "synthetic_demo":
        return "synthetic_demo"
    if source_type == "public_dataset":
        return "real_public_data"
    if metadata.get("reference_assisted"):
        return "reference_assisted"
    if metadata.get("is_mock"):
        return "mock"
    return source_type


def _render_source_boundary(conversation_map: dict[str, Any]) -> None:
    metadata = conversation_map.get("metadata", {})
    label = _source_label(metadata)
    if label == "synthetic_demo":
        st.warning(
            "This selected clip is a synthetic focused-MVP demo. It is for "
            "showing overlap, term rescue, and audit behavior, not for real "
            "audio performance claims."
        )
    elif label == "real_public_data":
        st.info(
            "This selected clip is real public data. The committed subset is "
            "small and mostly clean; term-rescue effects may be absent."
        )
    else:
        st.caption(f"Evidence source: {label}")
    boundary = metadata.get("data_boundary") or metadata.get("claim_scope")
    if boundary:
        st.caption(str(boundary))


def _anchor_status(anchor: dict[str, Any]) -> str:
    explicit = str(anchor.get("correction_status", "")).strip()
    if explicit:
        return explicit
    if anchor.get("unsupported_changes"):
        return "unsupported"
    if anchor.get("needs_review"):
        return "needs_review"
    return "supported"


def _audit_status(audit: dict[str, Any]) -> str:
    explicit = str(audit.get("correction_status", "")).strip()
    if explicit:
        return explicit
    if audit.get("unsupported_changes"):
        return "unsupported"
    if audit.get("needs_review"):
        return "needs_review"
    return "supported"


def _anchors_by_id(conversation_map: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(anchor.get("anchor_id")): anchor
        for anchor in conversation_map.get("anchors", [])
        if anchor.get("anchor_id")
    }


def _term_rows(conversation_map: dict[str, Any]) -> list[dict[str, Any]]:
    anchors = _anchors_by_id(conversation_map)
    rows: list[dict[str, Any]] = []
    for candidate in conversation_map.get("term_rescues", []):
        anchor_ids = [str(item) for item in list_value(candidate.get("evidence_anchor_ids"))]
        anchor = anchors.get(anchor_ids[0], {}) if anchor_ids else {}
        rows.append(
            {
                "timestamp": (
                    f"{number(anchor.get('start')):.2f}-"
                    f"{number(anchor.get('end')):.2f}s"
                    if anchor
                    else ""
                ),
                "speaker": anchor.get("speaker", ""),
                "raw_phrase": candidate.get("raw_phrase")
                or ", ".join(list_value(candidate.get("asr_error_forms"))),
                "retrieved_candidate": candidate.get("canonical", ""),
                "corrected_term": candidate.get("corrected_term")
                or candidate.get("canonical", ""),
                "retrieval_score": number(candidate.get("retrieved_score")),
                "source": candidate.get("source", "docs/knowledge_base/domain_terms.md"),
                "status": candidate.get("status", "needs_review"),
                "anchor_ids": ", ".join(anchor_ids),
                "method": candidate.get("retrieval_method", ""),
            }
        )
    return rows


def _audit_rows(conversation_map: dict[str, Any]) -> list[dict[str, Any]]:
    anchors = _anchors_by_id(conversation_map)
    rows: list[dict[str, Any]] = []
    for audit in conversation_map.get("correction_audits", []):
        anchor = anchors.get(str(audit.get("anchor_id")), {})
        rows.append(
            {
                "anchor_id": audit.get("anchor_id", ""),
                "timestamp": (
                    f"{number(anchor.get('start')):.2f}-"
                    f"{number(anchor.get('end')):.2f}s"
                    if anchor
                    else ""
                ),
                "speaker": anchor.get("speaker", ""),
                "overlap": bool(anchor.get("overlap")),
                "raw_text": audit.get("raw_text", anchor.get("raw_text", "")),
                "corrected_text": audit.get(
                    "corrected_text",
                    anchor.get("corrected_text", ""),
                ),
                "status": _audit_status(audit),
                "supported_changes": ", ".join(list_value(audit.get("supported_changes"))),
                "unsupported_changes": ", ".join(
                    list_value(audit.get("unsupported_changes"))
                ),
                "hallucination_risk": audit.get("hallucination_risk", ""),
                "needs_review": bool(audit.get("needs_review")),
                "correction_mode": audit.get("correction_mode", ""),
                "evidence": audit.get("evidence", []),
            }
        )
    return rows


def _status_options(values: list[str], order: tuple[str, ...]) -> list[str]:
    found = set(values)
    ordered = [status for status in order if status in found]
    ordered.extend(sorted(found - set(ordered)))
    return ordered


def render_evidence_timeline(conversation_map: dict[str, Any]) -> None:
    page_header(
        "Evidence Timeline",
        "Timestamped speaker anchors with raw ASR, corrected text, overlap, and review state.",
    )
    if not require_map(conversation_map):
        return
    _render_source_boundary(conversation_map)
    anchors = list(conversation_map.get("anchors", []))
    stats = map_stats(conversation_map)
    metric_columns = st.columns(4)
    metric_columns[0].metric("Segments", stats["anchors"])
    metric_columns[1].metric("Speakers", stats["speakers"])
    metric_columns[2].metric("Overlap events", stats["overlap_events"])
    metric_columns[3].metric("Needs review", stats["needs_review"])

    speakers = sorted(
        {
            str(speaker)
            for anchor in anchors
            for speaker in (
                anchor.get("speakers")
                or ([anchor.get("speaker")] if anchor.get("speaker") else [])
            )
        }
    )
    statuses = _status_options([_anchor_status(anchor) for anchor in anchors], STATUS_ORDER)
    filters = st.columns([2, 2, 1, 1])
    selected_speakers = filters[0].multiselect(
        "Speakers",
        speakers,
        default=speakers,
    )
    selected_statuses = filters[1].multiselect(
        "Correction status",
        statuses,
        default=statuses,
    )
    overlap_only = filters[2].toggle("Overlap", value=False)
    review_only = filters[3].toggle("Review", value=False)

    filtered = [
        anchor
        for anchor in anchors
        if (
            set(anchor.get("speakers") or [anchor.get("speaker", "UNKNOWN")])
            & set(selected_speakers)
        )
        and _anchor_status(anchor) in selected_statuses
        and (not overlap_only or anchor.get("overlap"))
        and (not review_only or anchor.get("needs_review"))
    ]
    table = anchor_table(filtered)
    if table.empty:
        st.info("No timeline segments match the current filters.")
        return
    table["correction_status"] = [
        _anchor_status(anchor) for anchor in filtered
    ]
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
                "correction_status",
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
        "Inspect segment",
        table["anchor_id"].tolist(),
        format_func=lambda value: (
            f"{value} | "
            f"{table.loc[table['anchor_id'] == value, 'start'].iloc[0]:.2f}s"
        ),
    )
    selected = next(anchor for anchor in filtered if anchor.get("anchor_id") == selected_id)
    render_text_diff(selected.get("raw_text", ""), selected.get("corrected_text", ""))
    st.caption(
        f"Status={_anchor_status(selected)} | "
        f"Overlap={bool(selected.get('overlap'))} | "
        f"Needs review={bool(selected.get('needs_review'))} | "
        f"Retrieved terms={', '.join(selected.get('retrieved_terms', [])) or 'none'}"
    )
    with st.expander("Segment evidence JSON"):
        st.json(
            {
                "anchor_id": selected.get("anchor_id"),
                "correction_evidence": selected.get("correction_evidence", []),
                "notes": selected.get("notes", []),
            }
        )


def render_term_rescue_focused(conversation_map: dict[str, Any]) -> None:
    page_header(
        "Term Rescue",
        "Domain-term retrieval candidates with source evidence and correction status.",
    )
    if not require_map(conversation_map):
        return
    _render_source_boundary(conversation_map)
    rows = _term_rows(conversation_map)
    if not rows:
        st.info("No term rescue candidates are available for the selected artifact.")
        return
    status_counts = Counter(row["status"] for row in rows)
    metric_columns = st.columns(4)
    metric_columns[0].metric("Successful", status_counts.get("successful_rescue", 0))
    metric_columns[1].metric("False rescue", status_counts.get("false_rescue", 0))
    metric_columns[2].metric("Missed", status_counts.get("missed_rescue", 0))
    metric_columns[3].metric("Needs review", status_counts.get("needs_review", 0))

    statuses = _status_options([row["status"] for row in rows], TERM_STATUS_ORDER)
    selected_statuses = st.multiselect("Status", statuses, default=statuses)
    frame = pd.DataFrame(
        [row for row in rows if row["status"] in selected_statuses]
    )
    if frame.empty:
        st.info("No term rescues match the current status filter.")
        return
    st.dataframe(
        frame[
            [
                "timestamp",
                "speaker",
                "raw_phrase",
                "retrieved_candidate",
                "corrected_term",
                "retrieval_score",
                "source",
                "status",
            ]
        ],
        width="stretch",
        hide_index=True,
        column_config={
            "retrieval_score": st.column_config.ProgressColumn(
                min_value=0.0,
                max_value=1.0,
            )
        },
    )
    selected = st.selectbox(
        "Inspect rescue",
        frame.to_dict("records"),
        format_func=lambda row: (
            f"{row['raw_phrase']} -> "
            f"{row['corrected_term'] or row['retrieved_candidate']} | {row['status']}"
        ),
    )
    st.markdown(
        f"""
        <div class="tw-evidence-band">
        <strong>{safe_html(selected['raw_phrase'])}</strong>
        -> <strong>{safe_html(selected['corrected_term'] or selected['retrieved_candidate'])}</strong><br>
        Source: <code>{safe_html(selected['source'])}</code><br>
        Method: <code>{safe_html(selected['method'])}</code> |
        Anchor: <code>{safe_html(selected['anchor_ids'])}</code>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_correction_audit(conversation_map: dict[str, Any]) -> None:
    page_header(
        "Correction Audit",
        "Raw-versus-corrected diffs with supported changes, unsupported additions, and review flags.",
    )
    if not require_map(conversation_map):
        return
    _render_source_boundary(conversation_map)
    rows = _audit_rows(conversation_map)
    if not rows:
        st.info("No correction audits are available for the selected artifact.")
        return
    status_counts = Counter(row["status"] for row in rows)
    metric_columns = st.columns(4)
    metric_columns[0].metric("Supported", status_counts.get("supported", 0))
    metric_columns[1].metric("Weak", status_counts.get("weakly_supported", 0))
    metric_columns[2].metric("Unsupported", status_counts.get("unsupported", 0))
    metric_columns[3].metric("Review", status_counts.get("needs_review", 0))

    statuses = _status_options([row["status"] for row in rows], STATUS_ORDER)
    selected_statuses = st.multiselect("Audit status", statuses, default=statuses)
    overlap_only = st.toggle("Overlap-sensitive edits only", value=False)
    filtered = [
        row
        for row in rows
        if row["status"] in selected_statuses
        and (not overlap_only or row["overlap"])
    ]
    if not filtered:
        st.info("No correction audits match the current filters.")
        return
    frame = pd.DataFrame(filtered)
    st.dataframe(
        frame[
            [
                "timestamp",
                "speaker",
                "overlap",
                "raw_text",
                "corrected_text",
                "status",
                "supported_changes",
                "unsupported_changes",
                "hallucination_risk",
                "needs_review",
            ]
        ],
        width="stretch",
        hide_index=True,
    )
    selected = st.selectbox(
        "Inspect audit",
        filtered,
        format_func=lambda row: f"{row['anchor_id']} | {row['status']}",
    )
    if selected["status"] == "unsupported":
        st.error(
            "Unsupported change detected. This proposed correction must stay "
            "in review until evidence is added."
        )
    render_text_diff(selected["raw_text"], selected["corrected_text"])
    with st.expander("Audit evidence JSON"):
        st.json(
            {
                "anchor_id": selected["anchor_id"],
                "status": selected["status"],
                "supported_changes": selected["supported_changes"],
                "unsupported_changes": selected["unsupported_changes"],
                "evidence": selected["evidence"],
                "correction_mode": selected["correction_mode"],
            }
        )


def render_evidence_dashboard_focused(conversation_map: dict[str, Any]) -> None:
    page_header(
        "Evidence Dashboard",
        "Small, claim-scoped metrics for the selected evidence artifact and committed experiments.",
    )
    if not require_map(conversation_map):
        return
    metadata = conversation_map.get("metadata", {})
    _render_source_boundary(conversation_map)
    stats = map_stats(conversation_map)
    term_rows = _term_rows(conversation_map)
    audit_rows = _audit_rows(conversation_map)
    term_success = sum(row["status"] == "successful_rescue" for row in term_rows)
    unsupported = sum(
        len(list_value(row.get("unsupported_changes")))
        for row in audit_rows
    )
    overlap_segments = sum(
        bool(anchor.get("overlap"))
        for anchor in conversation_map.get("anchors", [])
    )
    top = st.columns(5)
    top[0].metric("Term Rescue Count", term_success)
    top[1].metric("Unsupported Edit Count", unsupported)
    top[2].metric("Needs Review Count", stats["needs_review"])
    top[3].metric("Overlap Segment Count", overlap_segments)
    top[4].metric("Data Label", _source_label(metadata))

    st.subheader("Selected artifact configuration")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "clip_id": conversation_map.get("clip_id"),
                    "dataset_name": metadata.get("dataset_name"),
                    "asr_mode": metadata.get("asr_mode"),
                    "diarization_mode": metadata.get("diarization_mode"),
                    "llm_mode": metadata.get("llm_mode"),
                    "claim_scope": metadata.get("claim_scope"),
                }
            ]
        ),
        width="stretch",
        hide_index=True,
    )

    workflow = load_workflow_ablation()
    term_summary = load_term_rescue_summary()
    overlap_summary = load_overlap_safety_summary()
    for frame in (workflow, term_summary, overlap_summary):
        show_frame_warning(frame)
    if not workflow.empty:
        st.subheader("Real public-data workflow boundary")
        columns = [
            column
            for column in (
                "variant",
                "dataset_name",
                "language",
                "num_clips",
                "mean_num_overlap_anchors",
                "mean_num_term_candidates",
                "mean_num_term_rescues_applied",
                "mean_num_unsupported_changes",
                "notes",
            )
            if column in workflow
        ]
        st.dataframe(workflow[columns], width="stretch", hide_index=True)
    if not term_summary.empty:
        st.subheader("Controlled term-rescue fixture boundary")
        st.dataframe(term_summary.head(12), width="stretch", hide_index=True)
    if not overlap_summary.empty:
        st.subheader("Controlled overlap-safety fixture boundary")
        st.dataframe(overlap_summary.head(12), width="stretch", hide_index=True)
