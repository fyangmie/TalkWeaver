"""Cross-talk, term rescue, and Hallucination Watchdog views."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from webapp.data_loader import (
    get_event_investigation_rows,
    get_best_llm_rejection_examples,
    get_best_term_rescue_examples,
    get_correction_diff_examples,
    get_negative_control_examples,
    load_overlap_safety_cases,
    load_overlap_safety_summary,
    load_term_rescue_summary,
)
from webapp.detective_ui import (
    CHART_GROUPS,
    anchor_table,
    list_value,
    map_stats,
    page_header,
    render_chart_grid,
    require_map,
    show_frame_warning,
    truthy,
)
from webapp.ui_components import render_audio_evidence, render_text_diff


def _aggregate(
    frame: pd.DataFrame,
    mean_columns: tuple[str, ...],
    sum_columns: tuple[str, ...],
) -> pd.DataFrame:
    if frame.empty or "variant" not in frame:
        return pd.DataFrame()
    aggregations = {name: "mean" for name in mean_columns if name in frame}
    aggregations.update({name: "sum" for name in sum_columns if name in frame})
    return frame.groupby("variant", as_index=False).agg(aggregations)


def _format_case_label(row: dict[str, Any]) -> str:
    return f"{row.get('case_id', 'case')} | {row.get('variant', 'variant')}"


def _show_audit_metadata(row: dict[str, Any], *, reason_field: str = "notes") -> None:
    unsupported = list_value(row.get("unsupported_changes"))
    details = st.columns(4)
    details[0].metric("Needs review", str(truthy(row.get("needs_review"))))
    details[1].metric(
        "Rejected",
        str(truthy(row.get("correction_rejected"))),
    )
    details[2].metric("API used", str(truthy(row.get("api_used"))))
    details[3].metric("Fallback used", str(truthy(row.get("fallback_used"))))
    st.caption(
        f"Unsupported changes: {', '.join(str(item) for item in unsupported) or 'none'}"
    )
    reason = row.get(reason_field) or row.get("notes")
    if isinstance(reason, str) and reason.strip():
        st.caption(f"Reason: {reason}")


def _select_case(
    label: str,
    frame: pd.DataFrame,
    *,
    key: str,
) -> dict[str, Any] | None:
    if frame.empty:
        st.info(f"No {label.lower()} are available.")
        return None
    records = frame.to_dict("records")
    selected = st.selectbox(
        label,
        records,
        format_func=_format_case_label,
        key=key,
    )
    return selected


def _overlap_case(
    cases: pd.DataFrame,
    *,
    case_id: str,
    variant: str,
) -> dict[str, Any] | None:
    if cases.empty:
        return None
    match = cases[
        cases["case_id"].eq(case_id) & cases["variant"].eq(variant)
    ]
    return match.iloc[0].to_dict() if not match.empty else None


def _render_overlap_case(title: str, row: dict[str, Any] | None, finding: str) -> None:
    st.markdown(f"#### {title}")
    if row is None:
        st.info("Controlled example is not available.")
        return
    st.caption(finding)
    render_text_diff(row.get("raw_asr_text", ""), row.get("corrected_text", ""))
    _show_audit_metadata(row)


def render_overlap(conversation_map: dict[str, Any]) -> None:
    page_header(
        "Cross-talk and Overlap Warning",
        "Overlap is a correction-risk signal: uncertain speech is exposed for review rather than completed fluently.",
    )
    if require_map(conversation_map):
        events = get_event_investigation_rows(conversation_map)
        overlap_anchors = [
            anchor
            for anchor in conversation_map.get("anchors", [])
            if anchor.get("overlap")
        ]
        metrics = st.columns(3)
        metrics[0].metric("Map events", len(events))
        metrics[1].metric("Overlap anchors", len(overlap_anchors))
        metrics[2].metric(
            "Overlap review flags",
            sum(bool(anchor.get("needs_review")) for anchor in overlap_anchors),
        )
        if events:
            st.dataframe(pd.DataFrame(events), width="stretch", hide_index=True)
        else:
            st.info("The selected map has no overlap/interruption event records.")
        if overlap_anchors:
            st.dataframe(
                anchor_table(overlap_anchors)[
                    ["start", "end", "speakers", "raw_text", "confidence", "needs_review"]
                ],
                width="stretch",
                hide_index=True,
            )

        st.subheader("Event Investigation")
        st.markdown(
            """
            **Overlap** means simultaneous speech. **Interruption** is a
            conservative timing candidate where one speaker starts before
            another finishes and may take the floor. The current public
            subset has reference overlap labels but limited human interruption
            labels, so interruption candidates require review.
            """
        )
        event_speakers = sorted(
            {
                str(speaker)
                for event in events
                for speaker in event.get("speakers", [])
            }
        )
        filters = st.columns(3)
        overlap_filter = filters[0].toggle(
            "Overlap only",
            value=False,
            key="event_overlap_only",
        )
        review_filter = filters[1].toggle(
            "Needs review only",
            value=False,
            key="event_review_only",
        )
        speaker_filter = filters[2].multiselect(
            "Event speakers",
            event_speakers,
            default=event_speakers,
            key="event_speaker_filter",
        )
        filtered_events = [
            event
            for event in events
            if (not overlap_filter or event.get("type") == "overlap")
            and (not review_filter or event.get("needs_review"))
            and (
                not event_speakers
                or set(event.get("speakers", [])) & set(speaker_filter)
            )
        ]
        if filtered_events:
            selected_event = st.selectbox(
                "Investigate event",
                filtered_events,
                format_func=lambda event: (
                    f"{event.get('event_id', 'event')} | "
                    f"{float(event.get('start', 0.0)):.2f}-"
                    f"{float(event.get('end', 0.0)):.2f}s | "
                    f"{event.get('type', 'event')}"
                ),
                key="event_investigation",
            )
            event_metrics = st.columns(4)
            event_metrics[0].metric("Type", selected_event.get("type", "event"))
            event_metrics[1].metric(
                "Severity",
                selected_event.get("severity", "unknown"),
            )
            event_metrics[2].metric(
                "Needs review",
                str(bool(selected_event.get("needs_review"))),
            )
            event_metrics[3].metric(
                "Speakers",
                len(selected_event.get("speakers", [])),
            )
            st.write(selected_event.get("description", ""))
            st.caption(
                "Speakers: "
                f"{', '.join(selected_event.get('speakers', [])) or 'unknown'}"
            )
            st.caption(
                "Evidence anchors: "
                f"{', '.join(selected_event.get('evidence_anchor_ids', [])) or 'none'}"
            )
            render_text_diff(
                selected_event.get("related_raw_text", ""),
                selected_event.get("related_corrected_text", ""),
                raw_label="Related raw anchor evidence",
                corrected_label="Related corrected / retained evidence",
            )
            render_audio_evidence(
                conversation_map,
                selected_event,
                item_type="event",
                label="Event audio evidence",
            )
        else:
            st.info("No events match the current filters.")

        interruption_events = [
            event for event in events if event.get("type") == "interruption"
        ]
        if not interruption_events:
            st.info(
                "No human-labeled interruption event is present in this "
                "ConversationMap. The controlled examples below are shown as "
                "interruption-risk safety examples, not public interruption ground truth."
            )

    st.subheader("Controlled Phase 2G safety evidence")
    st.markdown(
        '<span class="tw-source-controlled">Controlled authored text fixtures, not public audio.</span>',
        unsafe_allow_html=True,
    )
    summary = load_overlap_safety_summary()
    show_frame_warning(summary)
    aggregate = _aggregate(
        summary,
        ("safety_pass_rate", "mean_text_error_before", "mean_text_error_after"),
        (
            "unsupported_change_count",
            "invented_content_count",
            "needs_review_count",
            "correction_rejected_count",
        ),
    )
    if not aggregate.empty:
        st.dataframe(aggregate, width="stretch", hide_index=True)

        aware = aggregate[aggregate["variant"].str.startswith("overlap_aware")]
        unaware = aggregate[
            aggregate["variant"].str.startswith("no_overlap_awareness")
        ]
        comparison = st.columns(4)
        comparison[0].metric(
            "Aware safety pass",
            f"{aware['safety_pass_rate'].mean():.2f}" if not aware.empty else "n/a",
        )
        comparison[1].metric(
            "No-awareness pass",
            f"{unaware['safety_pass_rate'].mean():.2f}" if not unaware.empty else "n/a",
        )
        comparison[2].metric(
            "Aware rejections",
            int(aware["correction_rejected_count"].sum()) if not aware.empty else 0,
        )
        comparison[3].metric(
            "Aware review flags",
            int(aware["needs_review_count"].sum()) if not aware.empty else 0,
        )
        st.caption(
            "Accepted invented content: "
            f"{int(aggregate['invented_content_count'].sum())}; "
            "all controlled variants retained speaker attribution."
        )

    cases = load_overlap_safety_cases()
    show_frame_warning(cases)
    if not cases.empty:
        st.subheader("Overlap decision case files")
        tab_mild, tab_heavy, tab_speaker, tab_rack = st.tabs(
            [
                "Mild overlap",
                "Heavy overlap",
                "Ambiguous speakers",
                "Rack negative control",
            ]
        )
        with tab_mild:
            _render_overlap_case(
                "Correction allowed, review retained",
                _overlap_case(
                    cases,
                    case_id="overlap_006",
                    variant="overlap_aware_rule",
                ),
                "Evidence-supported terms can be corrected during mild overlap, but the region remains marked for review.",
            )
        with tab_heavy:
            _render_overlap_case(
                "Unsafe completion rejected",
                _overlap_case(
                    cases,
                    case_id="overlap_009",
                    variant="overlap_aware_llm",
                ),
                "Heavy, incomplete cross-talk keeps the raw fragment and rejects fluent completion.",
            )
        with tab_speaker:
            _render_overlap_case(
                "Speaker attribution remains fixed",
                _overlap_case(
                    cases,
                    case_id="overlap_012",
                    variant="overlap_aware_rule",
                ),
                "TalkWeaver may correct supported terms but does not move words between speakers without evidence.",
            )
        with tab_rack:
            _render_overlap_case(
                "Physical rack stays rack",
                _overlap_case(
                    cases,
                    case_id="overlap_004",
                    variant="overlap_aware_rule",
                ),
                "Ordinary physical-object context blocks the RAG substitution.",
            )
    render_chart_grid(CHART_GROUPS["Controlled overlap safety"])


def render_term_rescue(_: dict[str, Any]) -> None:
    page_header(
        "Misheard Word Rescue",
        "Glossary, fuzzy, and phonetic-like evidence proposes technical terms; retrieval alone is not permission to rewrite.",
    )
    st.markdown(
        '<div class="tw-warning-band"><strong>Controlled fixture view.</strong> '
        "These results use authored technical-term text cases, not measured "
        "public audio.</div>",
        unsafe_allow_html=True,
    )
    summary = load_term_rescue_summary()
    show_frame_warning(summary)
    aggregate = _aggregate(
        summary,
        (
            "mean_term_precision",
            "mean_term_recall",
            "mean_term_f1",
            "mean_text_error_before",
            "mean_text_error_after",
        ),
        (
            "false_positive_count",
            "missed_term_count",
            "unsupported_change_count",
            "needs_review_count",
        ),
    )
    if not aggregate.empty:
        best = aggregate.sort_values("mean_term_f1", ascending=False).iloc[0]
        cards = st.columns(4)
        cards[0].metric("Best variant", best["variant"])
        cards[1].metric("Mean term F1", f"{float(best['mean_term_f1']):.3f}")
        cards[2].metric("False positives", int(best.get("false_positive_count", 0)))
        cards[3].metric("Needs review", int(best.get("needs_review_count", 0)))
    render_chart_grid(CHART_GROUPS["Controlled term rescue"])

    examples = get_best_term_rescue_examples()
    show_frame_warning(examples)
    if examples.empty:
        return
    st.subheader("Rescue Case Files")
    columns = [
        name
        for name in (
            "case_id",
            "raw_asr_text",
            "corrected_text",
            "expected_terms",
            "retrieved_candidates",
            "applied_corrections",
            "variant",
            "term_f1",
            "text_error_before",
            "text_error_after",
            "needs_review",
        )
        if name in examples
    ]
    st.dataframe(examples[columns], width="stretch", hide_index=True)

    selected = _select_case(
        "Controlled Correction Demo",
        examples,
        key="term_rescue_case",
    )
    if selected:
        st.markdown(
            '<span class="tw-source-controlled">Controlled technical-term fixture, not real audio.</span>',
            unsafe_allow_html=True,
        )
        render_text_diff(
            selected.get("raw_asr_text", ""),
            selected.get("corrected_text", ""),
        )
        evidence = st.columns(4)
        evidence[0].metric("Term F1", f"{float(selected.get('term_f1', 0)):.2f}")
        evidence[1].metric(
            "Error before",
            f"{float(selected.get('text_error_before', 0)):.3f}",
        )
        evidence[2].metric(
            "Error after",
            f"{float(selected.get('text_error_after', 0)):.3f}",
        )
        evidence[3].metric(
            "Needs review",
            str(truthy(selected.get("needs_review"))),
        )
        st.caption(
            "Expected terms: "
            f"{', '.join(str(item) for item in list_value(selected.get('expected_terms'))) or 'none'}"
        )
        st.caption(
            "Retrieved candidates: "
            f"{', '.join(str(item) for item in list_value(selected.get('retrieved_candidates'))) or 'none'}"
        )
        st.caption(
            "Applied corrections: "
            f"{', '.join(str(item) for item in list_value(selected.get('applied_corrections'))) or 'none'}"
        )


def _render_watchdog_section(
    title: str,
    explanation: str,
    frame: pd.DataFrame,
    *,
    key: str,
    reason_field: str = "notes",
) -> None:
    st.subheader(title)
    st.write(explanation)
    selected = _select_case(title, frame, key=key)
    if selected is None:
        return
    render_text_diff(
        selected.get("raw_asr_text", ""),
        selected.get("corrected_text", ""),
    )
    _show_audit_metadata(selected, reason_field=reason_field)


def render_watchdog(conversation_map: dict[str, Any]) -> None:
    page_header(
        "Hallucination Watchdog",
        "Raw evidence remains visible, unsupported changes are counted, and risky corrections can be rejected.",
    )
    st.markdown(
        """
        <div class="tw-evidence-band">
        TalkWeaver keeps raw ASR evidence and separates
        <code>corrected_text</code> from <code>raw_text</code>. If a correction
        is unsupported, it is rejected or marked <code>needs_review</code>.
        </div>
        """,
        unsafe_allow_html=True,
    )
    if require_map(conversation_map):
        audits = conversation_map.get("correction_audits", [])
        stats = map_stats(conversation_map)
        cards = st.columns(4)
        cards[0].metric("Map audits", len(audits))
        cards[1].metric("Unsupported", stats["unsupported"])
        cards[2].metric("Review anchors", stats["needs_review"])
        cards[3].metric(
            "API-backed audits",
            sum(bool(item.get("api_used")) for item in audits),
        )
        if audits:
            audit_rows = pd.DataFrame(audits)
            preferred = [
                name
                for name in (
                    "anchor_id",
                    "raw_text",
                    "corrected_text",
                    "supported_changes",
                    "unsupported_changes",
                    "hallucination_risk",
                    "needs_review",
                    "correction_mode",
                    "api_used",
                    "fallback_used",
                )
                if name in audit_rows
            ]
            st.dataframe(audit_rows[preferred], width="stretch", hide_index=True)

    st.markdown(
        '<span class="tw-source-controlled">The case files below are controlled safety fixtures, not public-audio performance.</span>',
        unsafe_allow_html=True,
    )
    _render_watchdog_section(
        "A. Accepted safe corrections",
        "Supported glossary evidence permits a narrow lexical correction.",
        get_correction_diff_examples(),
        key="watchdog_accepted",
    )
    _render_watchdog_section(
        "B. Rejected or review-needed LLM corrections",
        "Strict grounding keeps the raw text when the API output exceeds supported evidence.",
        get_best_llm_rejection_examples(),
        key="watchdog_rejected",
        reason_field="rejection_reason",
    )
    _render_watchdog_section(
        "C. Negative controls not replaced",
        "Common words such as rack and where remain unchanged when context does not support RAG or WER.",
        get_negative_control_examples(),
        key="watchdog_negative",
        reason_field="preservation_reason",
    )
