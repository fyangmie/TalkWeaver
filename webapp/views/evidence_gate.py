"""EvidenceGate trained correction-safety model view."""

from __future__ import annotations

from typing import Any

import streamlit as st

from webapp.data_loader import (
    get_best_evidence_gate_validation,
    get_evidence_gate_validation_examples,
    load_evidence_gate_leakage_audit,
    load_evidence_gate_validation_metrics,
)
from webapp.detective_ui import (
    CHART_GROUPS,
    page_header,
    render_chart_grid,
    show_frame_warning,
)
from webapp.ui_components import render_text_diff


def _setup_instructions() -> None:
    st.code(
        "python experiments/build_evidence_gate_dataset.py\n"
        "python experiments/augment_evidence_gate_examples.py\n"
        "python experiments/audit_evidence_gate_features.py\n"
        "python experiments/build_evidence_gate_heldout.py\n"
        "python experiments/train_evidence_gate.py --feature-set audit_aware\n"
        "python experiments/train_evidence_gate.py --feature-set evidence_only\n"
        "python experiments/train_evidence_gate.py --feature-set risk_only\n"
        "python experiments/evaluate_evidence_gate_heldout.py\n"
        "python experiments/plot_evidence_gate_validation.py",
        language="bash",
    )


def _render_example(title: str, frame: Any, decision: str) -> None:
    st.markdown(f"#### {title}")
    if frame.empty:
        st.info(f"No {decision} example is available.")
        return
    row = frame.iloc[0].to_dict()
    columns = st.columns(4)
    columns[0].metric("Predicted", row.get("predicted_label", decision))
    columns[1].metric(
        "Expected",
        row.get("true_label", decision),
    )
    predicted = str(row.get("predicted_label", decision))
    columns[2].metric(
        "Confidence",
        f"{float(row.get(f'prob_{predicted}', 0.0)):.3f}",
    )
    columns[3].metric(
        "Correct",
        str(predicted == str(row.get("true_label", decision))),
    )
    render_text_diff(
        row.get("raw_text", ""),
        row.get("corrected_text", ""),
    )
    st.caption(
        f"Case `{row.get('case_id', 'unknown')}` · "
        f"template group `{row.get('template_group', 'unknown')}` · "
        f"feature set `{row.get('feature_set', 'unknown')}`"
    )


def render_evidence_gate(_: dict[str, Any]) -> None:
    page_header(
        "EvidenceGate Model",
        "A trained lightweight gate decides whether a proposed correction should be accepted, rejected, or sent for human review.",
    )
    st.markdown(
        '<span class="tw-source-controlled">Controlled / semi-synthetic '
        "correction-safety experiment</span>",
        unsafe_allow_html=True,
    )
    st.error(
        "The initial EvidenceGate result is a policy-distillation sanity "
        "check. Perfect scores likely reflect label-proxy features and should "
        "not be interpreted as real-world generalization."
    )
    st.info(
        "EvidenceGate is TalkWeaver's trained model component. ASR proposes "
        "text, retrieval and an LLM/rule system propose a correction, and "
        "EvidenceGate classifies the audit decision. It is not an ASR or LLM "
        "fine-tune and is not evidence of real-audio generalization."
    )

    metrics = load_evidence_gate_validation_metrics()
    leakage = load_evidence_gate_leakage_audit()
    for frame in (metrics, leakage):
        show_frame_warning(frame)
    best = get_best_evidence_gate_validation()
    if metrics.empty or best is None:
        st.warning(
            "Strict EvidenceGate validation artifacts have not been generated locally."
        )
        _setup_instructions()
        return

    summary = st.columns(5)
    summary[0].metric(
        "Best strict model",
        f"{best['feature_set']} / {best['model_name']}",
    )
    summary[1].metric("Macro F1", f"{float(best['macro_f1']):.3f}")
    summary[2].metric(
        "False accept",
        f"{float(best['false_accept_rate']):.3f}",
    )
    summary[3].metric(
        "Unsafe accept",
        f"{float(best['unsafe_accept_rate']):.3f}",
    )
    summary[4].metric(
        "Review recall",
        f"{float(best['needs_review_recall']):.3f}",
    )

    st.warning(
        "Independent heldout performance is weak, especially for "
        "`needs_review`. This is the current honest result: strict features "
        "do not yet generalize reliably from the Phase 2F/2G templates."
    )

    st.subheader("Three validation meanings")
    definitions = st.columns(3)
    definitions[0].info(
        "**Audit-aware**\n\nUses reference-derived and final audit fields. "
        "It only measures whether a classifier can reproduce the controlled policy."
    )
    definitions[1].success(
        "**Evidence-only**\n\nUses proposal, retrieval, overlap, uncertainty, "
        "language, and model metadata without final audit outcomes."
    )
    definitions[2].warning(
        "**Risk-only**\n\nUses only edit magnitude, overlap, uncertainty, "
        "pre-decision risk, model type, and language."
    )

    st.subheader("Feature leakage audit")
    st.dataframe(
        leakage,
        width="stretch",
        hide_index=True,
    )
    st.subheader("Grouped test versus independent heldout")
    comparison = metrics[
        metrics["split"].isin(["grouped_test", "independent_heldout"])
    ].copy()
    columns = [
        "split",
        "feature_set",
        "model_name",
        "is_baseline",
        "macro_f1",
        "accuracy",
        "false_accept_rate",
        "unsafe_accept_rate",
        "needs_review_recall",
        "reject_recall",
        "accept_precision",
    ]
    st.dataframe(
        comparison[[column for column in columns if column in comparison]],
        width="stretch",
        hide_index=True,
    )
    render_chart_grid(
        CHART_GROUPS["EvidenceGate leakage and strict validation"]
    )

    st.subheader("Independent heldout case files")
    st.caption(
        "These examples intentionally surface errors when available; they are "
        "diagnostic cases, not success-only demos."
    )
    examples = get_evidence_gate_validation_examples()
    tabs = st.tabs(["Accepted", "Rejected", "Needs review"])
    with tabs[0]:
        _render_example(
            "Accepted evidence-supported correction",
            examples["accept"],
            "accept",
        )
    with tabs[1]:
        _render_example(
            "Rejected unsafe correction",
            examples["reject"],
            "reject",
        )
    with tabs[2]:
        _render_example(
            "Correction routed to human review",
            examples["needs_review"],
            "needs_review",
        )
