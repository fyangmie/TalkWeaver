"""Cross-talk, term rescue, and Hallucination Watchdog views."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from webapp.data_loader import (
    load_overlap_safety_results,
    load_overlap_safety_summary,
    load_term_rescue_results,
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


def render_overlap(conversation_map: dict[str, Any]) -> None:
    page_header(
        "Cross-talk and Overlap Warning",
        "Overlap is a correction-risk signal: uncertain speech is exposed for review rather than completed fluently.",
    )
    if require_map(conversation_map):
        events = [
            event
            for event in conversation_map.get("events", [])
            if event.get("type") in {"overlap", "interruption"}
        ]
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
        st.dataframe(aggregate, width="stretch", hide_index=True)
    render_chart_grid(CHART_GROUPS["Controlled term rescue"])

    results = load_term_rescue_results()
    show_frame_warning(results)
    if not results.empty:
        preferred = results[
            results["variant"].isin(
                ["fused_plus_llm_correction", "fused_plus_rule_correction", "fused"]
            )
        ].copy()
        examples = preferred if not preferred.empty else results
        columns = [
            name
            for name in (
                "case_id",
                "variant",
                "raw_asr_text",
                "retrieved_candidates",
                "corrected_text",
                "expected_terms",
                "false_positive_terms",
                "needs_review",
            )
            if name in examples
        ]
        st.subheader("Term rescue evidence examples")
        st.dataframe(examples[columns].head(15), width="stretch", hide_index=True)


def _find_watchdog_examples(
    term_results: pd.DataFrame,
    overlap_results: pd.DataFrame,
) -> dict[str, dict[str, Any] | None]:
    accepted = next(
        (
            row
            for row in term_results.to_dict("records")
            if row.get("corrected_text") != row.get("raw_asr_text")
            and not list_value(row.get("unsupported_changes"))
            and not truthy(row.get("needs_review"))
        ),
        None,
    )
    negative = next(
        (
            row
            for row in term_results.to_dict("records")
            if row.get("corrected_text") == row.get("raw_asr_text")
            and not list_value(row.get("false_positive_terms"))
            and any(
                word in str(row.get("raw_asr_text", "")).lower()
                for word in ("rack", "where")
            )
        ),
        None,
    )
    rejected = next(
        (
            row
            for row in overlap_results.to_dict("records")
            if truthy(row.get("correction_rejected"))
        ),
        None,
    )
    return {"accepted": accepted, "rejected": rejected, "negative": negative}


def _render_example(title: str, row: dict[str, Any] | None, status: str) -> None:
    with st.expander(title, expanded=True):
        if row is None:
            st.info("No matching example is available.")
            return
        st.caption(status)
        raw, corrected = st.columns(2)
        raw.markdown("**Raw**")
        raw.write(row.get("raw_asr_text", row.get("raw_text", "")))
        corrected.markdown("**Corrected / retained**")
        corrected.write(row.get("corrected_text", ""))
        st.caption(
            f"needs_review={truthy(row.get('needs_review'))} | "
            f"unsupported={len(list_value(row.get('unsupported_changes')))} | "
            f"api_used={truthy(row.get('api_used'))}"
        )


def render_watchdog(conversation_map: dict[str, Any]) -> None:
    page_header(
        "Hallucination Watchdog",
        "Raw evidence remains visible, unsupported changes are counted, and risky corrections can be rejected.",
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
        else:
            st.info("The selected map has no CorrectionAudit records.")

    term_results = load_term_rescue_results()
    overlap_results = load_overlap_safety_results()
    examples = _find_watchdog_examples(term_results, overlap_results)
    st.subheader("Controlled watchdog decisions")
    st.markdown(
        '<span class="tw-source-controlled">Controlled safety fixtures; not public-audio performance.</span>',
        unsafe_allow_html=True,
    )
    columns = st.columns(3)
    with columns[0]:
        _render_example(
            "Accepted supported correction",
            examples["accepted"],
            "Evidence-supported lexical edit",
        )
    with columns[1]:
        _render_example(
            "Rejected uncertain correction",
            examples["rejected"],
            "Raw text retained after safety rejection",
        )
    with columns[2]:
        _render_example(
            "Negative control preserved",
            examples["negative"],
            "Common word not rewritten without domain context",
        )
