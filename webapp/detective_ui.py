"""Shared presentation helpers for AI Meeting Detective views."""

from __future__ import annotations

import html
import json
from typing import Any

import pandas as pd
import streamlit as st

from webapp.data_loader import discover_charts, frame_warning


CHART_GROUPS = {
    "ASR evidence": (
        "asr_error_by_language.png",
        "asr_error_by_dataset.png",
        "asr_rtf_by_model.png",
    ),
    "Workflow evidence": (
        "workflow_ablation_completeness.png",
        "workflow_ablation_review_flags.png",
    ),
    "Controlled term rescue": (
        "term_rescue_f1_by_variant.png",
        "term_rescue_false_positive_by_variant.png",
        "term_rescue_error_delta.png",
    ),
    "Controlled overlap safety": (
        "overlap_safety_pass_rate.png",
        "overlap_unsupported_changes.png",
        "overlap_review_flags.png",
        "overlap_error_delta.png",
    ),
    "EvidenceGate trained safety model": (
        "evidence_gate_macro_f1.png",
        "evidence_gate_false_accept_rate.png",
        "evidence_gate_confusion_matrix.png",
        "evidence_gate_feature_importance.png",
        "evidence_gate_class_recall.png",
    ),
}


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def list_value(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return [item.strip() for item in value.split(",") if item.strip()]
        return parsed if isinstance(parsed, list) else [parsed]
    return [value]


def safe_html(value: Any) -> str:
    return html.escape(str(value if value is not None else ""))


def show_frame_warning(frame: pd.DataFrame) -> None:
    warning = frame_warning(frame)
    if warning:
        st.info(warning)


def map_stats(conversation_map: dict[str, Any]) -> dict[str, Any]:
    anchors = conversation_map.get("anchors", [])
    events = conversation_map.get("events", [])
    audits = conversation_map.get("correction_audits", [])
    speaker_ids = {
        speaker
        for anchor in anchors
        for speaker in (
            anchor.get("speakers")
            or ([anchor.get("speaker")] if anchor.get("speaker") else [])
        )
        if speaker not in {"UNKNOWN", "OVERLAP", None, ""}
    }
    anchor_unsupported = sum(
        len(anchor.get("unsupported_changes", [])) for anchor in anchors
    )
    audit_unsupported = sum(
        len(audit.get("unsupported_changes", [])) for audit in audits
    )
    return {
        "anchors": len(anchors),
        "speakers": len(speaker_ids),
        "overlap_events": sum(event.get("type") == "overlap" for event in events),
        "needs_review": sum(bool(anchor.get("needs_review")) for anchor in anchors),
        "unsupported": max(anchor_unsupported, audit_unsupported),
        "term_rescues": len(conversation_map.get("term_rescues", [])),
        "audits": len(audits),
    }


def page_header(title: str, description: str) -> None:
    st.markdown(
        '<div class="tw-kicker">AI Meeting Detective</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="tw-case-header"><h1>{safe_html(title)}</h1></div>',
        unsafe_allow_html=True,
    )
    st.write(description)


def source_boundary(metadata: dict[str, Any]) -> None:
    is_mock = truthy(metadata.get("is_mock"))
    reference = metadata.get("diarization_mode") == "reference" or truthy(
        metadata.get("reference_assisted")
    )
    if is_mock:
        st.warning("Selected map is deterministic mock/demo evidence.")
    elif reference:
        st.markdown(
            '<span class="tw-source-real">Real public ASR artifact</span> + '
            '<span class="tw-source-oracle">reference-assisted speaker/time evidence</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="tw-source-real">Automatic or reference-backed workflow artifact</span>',
            unsafe_allow_html=True,
        )


def public_correction_is_conservative(
    conversation_map: dict[str, Any],
    *,
    threshold: float = 0.8,
) -> bool:
    """Return whether a public clip mostly retained its raw ASR text."""

    metadata = conversation_map.get("metadata", {})
    if metadata.get("source_type") != "public_dataset":
        return False
    anchors = conversation_map.get("anchors", [])
    if not anchors:
        return False
    identical = sum(
        str(anchor.get("raw_text", "")).strip()
        == str(anchor.get("corrected_text", "")).strip()
        for anchor in anchors
    )
    return identical / len(anchors) >= threshold


def render_public_correction_notice(conversation_map: dict[str, Any]) -> None:
    """Explain unchanged public-data text without implying a failed workflow."""

    if public_correction_is_conservative(conversation_map):
        st.info(
            "Correction note: this public-data clip has no annotated "
            "technical-term correction target. TalkWeaver preserved the ASR "
            "text instead of forcing unsupported edits. See Misheard Word "
            "Rescue and Hallucination Watchdog for controlled correction examples."
        )


def render_chart_grid(names: tuple[str, ...], columns: int = 2) -> None:
    charts = discover_charts(names)
    if not charts:
        st.info("The requested charts are not available in `assets/result_charts/`.")
        return
    slots = st.columns(columns)
    for index, name in enumerate(names):
        path = charts.get(name)
        if path is not None:
            slots[index % columns].image(path, caption=name, width="stretch")
        else:
            slots[index % columns].info(f"Missing chart: `{name}`")


def require_map(conversation_map: dict[str, Any]) -> bool:
    warning = conversation_map.get("_warning")
    if warning:
        st.warning(warning)
        st.code(
            "python experiments/run_reference_workflow_maps.py "
            "--manifest data/manifests/formal_eval_real.csv "
            '--dataset "AMI Meeting Corpus" '
            "--output-dir outputs/conversation_maps/reference_assisted_real"
        )
        return False
    return True


def anchor_table(anchors: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for anchor in anchors:
        rows.append(
            {
                "start": number(anchor.get("start")),
                "end": number(anchor.get("end")),
                "speaker": anchor.get("speaker", "UNKNOWN"),
                "speakers": ", ".join(anchor.get("speakers", [])),
                "raw_text": anchor.get("raw_text", ""),
                "corrected_text": anchor.get("corrected_text", ""),
                "overlap": bool(anchor.get("overlap")),
                "needs_review": bool(anchor.get("needs_review")),
                "confidence": number(anchor.get("confidence")),
                "retrieved_terms": ", ".join(anchor.get("retrieved_terms", [])),
                "anchor_id": anchor.get("anchor_id", ""),
            }
        )
    return pd.DataFrame(rows)
