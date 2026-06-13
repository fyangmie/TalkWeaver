"""Home and Conversation Crime Scene views."""

from __future__ import annotations

from typing import Any

import streamlit as st

from webapp.components.project_layout import DETECTIVE_SUBTITLE, DETECTIVE_TITLE
from webapp.data_loader import build_clip_detective_summary
from webapp.detective_ui import (
    map_stats,
    number,
    page_header,
    render_public_correction_notice,
    require_map,
    safe_html,
    source_boundary,
)
from webapp.ui_components import render_audio_evidence


def render_home(conversation_map: dict[str, Any]) -> None:
    page_header(DETECTIVE_TITLE, DETECTIVE_SUBTITLE)
    st.markdown(
        """
        <div class="tw-evidence-band">
        TalkWeaver investigates <strong>who said what, when speakers crossed,
        which terms were misheard, what correction changed, and which regions
        still need human review</strong>. The transcript is evidence, not the
        final verdict.
        </div>
        """,
        unsafe_allow_html=True,
    )
    badge_columns = st.columns(5)
    labels = (
        ("Real ASR", "tiny + base"),
        ("Speaker-time", "oracle labeled"),
        ("Term rescue", "25 fixtures"),
        ("LLM safety", "audited"),
        ("Overlap-aware", "20 fixtures"),
    )
    for column, (label, value) in zip(badge_columns, labels):
        column.metric(label, value)

    left, right = st.columns([1.05, 0.95])
    with left:
        st.subheader("Investigation path")
        st.markdown(
            """
            1. Open a Conversation Crime Scene and inspect its evidence modes.
            2. Trace speaker-attributed temporal anchors.
            3. Review cross-talk before trusting fluent corrections.
            4. Inspect retrieved technical terms and negative controls.
            5. Audit every raw-versus-corrected change.
            """
        )
    with right:
        st.subheader("Current evidence boundary")
        st.markdown(
            """
            - **Real public data:** 17 FLEURS and AMI clips for ASR baselines.
            - **Oracle-assisted evidence:** AMI reference speaker/time labels.
            - **Controlled fixtures:** technical-term and overlap safety tests.
            - **Automatic diarization:** not measured without configured model access.
            """
        )

    if require_map(conversation_map):
        metadata = conversation_map.get("metadata", {})
        st.subheader("Selected investigation")
        source_boundary(metadata)
        columns = st.columns(4)
        columns[0].metric("Clip", conversation_map.get("clip_id", "unknown"))
        columns[1].metric("Dataset", metadata.get("dataset_name", "unknown"))
        columns[2].metric("Language", metadata.get("language", "unknown"))
        columns[3].metric("Workflow", metadata.get("variant", "ConversationMap"))


def render_crime_scene(conversation_map: dict[str, Any]) -> None:
    page_header(
        "Conversation Crime Scene",
        "A compact case file showing the evidence modes, uncertainty, and review burden.",
    )
    if not require_map(conversation_map):
        return
    metadata = conversation_map.get("metadata", {})
    stats = map_stats(conversation_map)
    source_boundary(metadata)
    render_public_correction_notice(conversation_map)
    render_audio_evidence(
        conversation_map,
        label="Full clip audio evidence",
    )

    top = st.columns(6)
    top[0].metric("Clip", conversation_map.get("clip_id", "unknown"))
    top[1].metric("Dataset", metadata.get("dataset_name", "unknown"))
    top[2].metric("Language", metadata.get("language", "unknown"))
    top[3].metric("Anchors", stats["anchors"])
    top[4].metric("Speakers", stats["speakers"])
    top[5].metric("Overlap events", stats["overlap_events"])

    bottom = st.columns(6)
    bottom[0].metric("Needs review", stats["needs_review"])
    bottom[1].metric("Unsupported", stats["unsupported"])
    bottom[2].metric("Term rescues", stats["term_rescues"])
    bottom[3].metric("Correction", metadata.get("llm_mode", "unknown"))
    bottom[4].metric("ASR", metadata.get("asr_mode", "unknown"))
    bottom[5].metric("Diarization", metadata.get("diarization_mode", "unknown"))

    anchors = conversation_map.get("anchors", [])
    uncertain = min(
        anchors,
        key=lambda item: number(item.get("confidence"), 1.0),
    ) if anchors else {}
    overlap_anchors = [item for item in anchors if item.get("overlap")]
    overlap_reviewed = all(item.get("needs_review") for item in overlap_anchors)

    st.subheader("Detective summary")
    detective = build_clip_detective_summary(conversation_map)
    summary_columns = st.columns(2)
    with summary_columns[0]:
        st.markdown("#### What happened?")
        st.write(detective["what_happened"])
        st.markdown("#### Who spoke?")
        st.write(detective["who_spoke"])
        st.markdown("#### Evidence scope")
        st.write(detective["evidence_scope"])
        if uncertain:
            st.markdown(
                f"""
                <div class="tw-warning-band">
                <strong>Most uncertain region:</strong>
                {number(uncertain.get('start')):.2f}-
                {number(uncertain.get('end')):.2f}s,
                {safe_html(uncertain.get('speaker', 'UNKNOWN'))},
                confidence {number(uncertain.get('confidence')):.2f}.<br>
                {safe_html(uncertain.get('raw_text', ''))}
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info("No temporal anchor is present.")
    with summary_columns[1]:
        conservative = stats["unsupported"] == 0 and (
            not overlap_anchors or overlap_reviewed
        )
        st.markdown("#### Where did cross-talk happen?")
        st.write(detective["where_cross_talk"])
        st.markdown("#### What needs review?")
        st.write(detective["what_needs_review"])
        st.markdown("#### Were corrections rejected?")
        st.write(detective["corrections_rejected"])
        st.markdown(
            f"""
            <div class="tw-evidence-band">
            <strong>Cross-talk:</strong> {stats['overlap_events']} event(s),
            {len(overlap_anchors)} overlap anchor(s).<br>
            <strong>Correction posture:</strong>
            {'conservative / review-aware' if conservative else 'requires audit'}.<br>
            <strong>Speaker evidence:</strong>
            {'available' if stats['speakers'] > 0 else 'missing or unknown'}.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("Case metadata and claim scope"):
        st.json(metadata, expanded=False)
