"""EvidenceGate trained correction-safety model view."""

from __future__ import annotations

from typing import Any

import streamlit as st

from webapp.data_loader import (
    get_best_evidence_gate_model,
    get_evidence_gate_examples,
    load_evidence_gate_evaluation,
    load_evidence_gate_feature_importance,
    load_evidence_gate_metrics,
    load_evidence_gate_split_summary,
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
        "python experiments/train_evidence_gate.py "
        "--input data/controlled_evidence_gate/"
        "evidence_gate_examples_augmented.csv "
        "--output-dir experiments/results/evidence_gate "
        "--models logistic_regression random_forest gradient_boosting "
        "--group-split-column template_group --random-seed 42\n"
        "python experiments/evaluate_evidence_gate.py\n"
        "python experiments/plot_evidence_gate.py",
        language="bash",
    )


def _render_example(title: str, frame: Any, decision: str) -> None:
    st.markdown(f"#### {title}")
    if frame.empty:
        st.info(f"No {decision} example is available.")
        return
    row = frame.iloc[0].to_dict()
    columns = st.columns(4)
    columns[0].metric("Decision", row.get("predicted_label", decision))
    columns[1].metric(
        "Confidence",
        f"{float(row.get(f'prob_{decision}', 0.0)):.3f}",
    )
    columns[2].metric("Source", row.get("source_experiment", "controlled"))
    columns[3].metric(
        "Augmented",
        str(bool(row.get("is_augmented", False))),
    )
    render_text_diff(
        row.get("raw_text", ""),
        row.get("corrected_text", ""),
    )
    st.caption(
        f"Case `{row.get('case_id', 'unknown')}` · "
        f"template group `{row.get('template_group', 'unknown')}` · "
        f"reason: {row.get('label_reason', 'not recorded')}"
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
    st.info(
        "EvidenceGate is TalkWeaver's trained model component. ASR proposes "
        "text, retrieval and an LLM/rule system propose a correction, and "
        "EvidenceGate classifies the audit decision. It is not an ASR or LLM "
        "fine-tune and is not evidence of real-audio generalization."
    )

    metrics = load_evidence_gate_metrics()
    evaluation = load_evidence_gate_evaluation()
    importance = load_evidence_gate_feature_importance()
    split = load_evidence_gate_split_summary()
    for frame in (metrics, evaluation, importance, split):
        show_frame_warning(frame)
    best = get_best_evidence_gate_model()
    if metrics.empty or best is None:
        st.warning("EvidenceGate artifacts have not been generated locally.")
        _setup_instructions()
        return

    summary = st.columns(5)
    summary[0].metric("Best model", best["model_name"])
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
        "Near-perfect controlled scores are expected to be optimistic: labels "
        "and features share explicit safety-audit signals, and augmentation is "
        "rule-generated. The group split prevents case-template leakage, but "
        "external human-audited correction data is still required."
    )

    left, right = st.columns([1.1, 0.9])
    with left:
        st.subheader("Model and baseline comparison")
        comparison = evaluation
        if not comparison.empty and "split" in comparison:
            test = comparison[comparison["split"].eq("test")]
            comparison = test if not test.empty else comparison
        columns = [
            "model_name",
            "is_baseline",
            "macro_f1",
            "false_accept_rate",
            "unsafe_accept_rate",
            "needs_review_recall",
            "reject_recall",
        ]
        st.dataframe(
            comparison[[column for column in columns if column in comparison]],
            width="stretch",
            hide_index=True,
        )
    with right:
        st.subheader("Leakage-controlled split")
        st.dataframe(split, width="stretch", hide_index=True)
        st.caption(
            "All variants and augmentations from one `template_group` stay in "
            "exactly one split."
        )

    st.subheader("What drove the model?")
    best_importance = importance[
        importance["model_name"].eq(best["model_name"])
    ].sort_values("rank")
    st.dataframe(
        best_importance.head(12),
        width="stretch",
        hide_index=True,
    )
    render_chart_grid(CHART_GROUPS["EvidenceGate trained safety model"])

    st.subheader("Decision case files")
    examples = get_evidence_gate_examples()
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
