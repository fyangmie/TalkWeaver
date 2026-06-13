"""Markdown detective-report generation from a ConversationMap."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_DIR = ROOT_DIR / "outputs" / "reports"


def _safe_text(value: Any) -> str:
    return str(value if value is not None else "").replace("|", "\\|").replace("\n", " ")


def _count_unsupported(conversation_map: dict[str, Any]) -> int:
    anchors = conversation_map.get("anchors", [])
    audits = conversation_map.get("correction_audits", [])
    anchor_count = sum(len(item.get("unsupported_changes", [])) for item in anchors)
    audit_count = sum(len(item.get("unsupported_changes", [])) for item in audits)
    return max(anchor_count, audit_count)


def build_detective_report(
    conversation_map: dict[str, Any],
    chart_paths: Iterable[str | Path] | None = None,
    *,
    max_anchors: int = 12,
) -> str:
    """Build an evidence-oriented report without mutating the source map."""

    clip_id = str(conversation_map.get("clip_id", "unknown_clip"))
    metadata = conversation_map.get("metadata", {})
    anchors = list(conversation_map.get("anchors", []))
    events = list(conversation_map.get("events", []))
    audits = list(conversation_map.get("correction_audits", []))
    term_rescues = list(conversation_map.get("term_rescues", []))
    review_anchors = [item for item in anchors if item.get("needs_review")]
    overlap_events = [item for item in events if item.get("type") == "overlap"]

    lines = [
        f"# TalkWeaver Detective Report: {_safe_text(clip_id)}",
        "",
        "> Evidence-grounded preview generated from a local ConversationMap. "
        "Reference speaker-time evidence is oracle-assisted when labeled as such.",
        "",
        "## Case Metadata",
        "",
        f"- Dataset/source: {_safe_text(metadata.get('dataset_name', metadata.get('source_type', 'unknown')))}",
        f"- Language: {_safe_text(metadata.get('language', 'unknown'))}",
        f"- Duration: {_safe_text(metadata.get('duration_seconds', 'unknown'))} seconds",
        f"- ASR mode: {_safe_text(metadata.get('asr_mode', 'unknown'))}",
        f"- Diarization mode: {_safe_text(metadata.get('diarization_mode', 'unknown'))}",
        f"- Correction mode: {_safe_text(metadata.get('llm_mode', metadata.get('correction_mode', 'unknown')))}",
        f"- Evaluation scope: {_safe_text(metadata.get('evaluation_scope', 'not specified'))}",
        "",
        "## Evidence Summary",
        "",
        f"- Temporal anchors: {len(anchors)}",
        f"- Conversation events: {len(events)}",
        f"- Overlap events: {len(overlap_events)}",
        f"- Review flags: {len(review_anchors)}",
        f"- Term rescue candidates: {len(term_rescues)}",
        f"- Correction audits: {len(audits)}",
        f"- Unsupported changes: {_count_unsupported(conversation_map)}",
        "",
        "## Temporal Anchors",
        "",
        "| Time | Speaker(s) | Raw evidence | Corrected text | Flags |",
        "| --- | --- | --- | --- | --- |",
    ]
    for anchor in anchors[:max_anchors]:
        speakers = anchor.get("speakers") or [anchor.get("speaker", "UNKNOWN")]
        flags = []
        if anchor.get("overlap"):
            flags.append("overlap")
        if anchor.get("interruption"):
            flags.append("interruption")
        if anchor.get("needs_review"):
            flags.append("needs review")
        lines.append(
            "| "
            f"{float(anchor.get('start', 0.0)):.2f}-{float(anchor.get('end', 0.0)):.2f}s | "
            f"{_safe_text(', '.join(str(item) for item in speakers))} | "
            f"{_safe_text(anchor.get('raw_text', ''))} | "
            f"{_safe_text(anchor.get('corrected_text', ''))} | "
            f"{_safe_text(', '.join(flags) or 'clear')} |"
        )
    if len(anchors) > max_anchors:
        lines.extend(["", f"_Showing {max_anchors} of {len(anchors)} anchors._"])

    lines.extend(["", "## Overlap And Interruption Warnings", ""])
    if events:
        for event in events:
            lines.append(
                f"- **{_safe_text(event.get('type', 'event'))}** "
                f"{float(event.get('start', 0.0)):.2f}-"
                f"{float(event.get('end', 0.0)):.2f}s, "
                f"speakers: {_safe_text(', '.join(event.get('speakers', [])))}. "
                f"{_safe_text(event.get('description', ''))}"
            )
    else:
        lines.append("- No event evidence is present in this ConversationMap.")

    lines.extend(["", "## Review Flags", ""])
    if review_anchors:
        for anchor in review_anchors:
            lines.append(
                f"- `{_safe_text(anchor.get('anchor_id', 'anchor'))}` at "
                f"{float(anchor.get('start', 0.0)):.2f}s: "
                f"{_safe_text('; '.join(anchor.get('notes', [])) or 'manual review requested')}"
            )
    else:
        lines.append("- No anchors are currently marked for review.")

    lines.extend(["", "## Term Rescue Evidence", ""])
    if term_rescues:
        for candidate in term_rescues:
            lines.append(
                f"- **{_safe_text(candidate.get('canonical', 'term'))}** via "
                f"{_safe_text(candidate.get('retrieval_method', 'unknown'))}, "
                f"score {_safe_text(candidate.get('retrieved_score', 'n/a'))}, "
                f"anchors: {_safe_text(', '.join(candidate.get('evidence_anchor_ids', [])))}"
            )
    else:
        lines.append("- No term rescue candidate is attached to this map.")

    lines.extend(["", "## Correction Audit", ""])
    if audits:
        for audit in audits[:max_anchors]:
            decision = "review" if audit.get("needs_review") else "accepted"
            lines.append(
                f"- `{_safe_text(audit.get('anchor_id', 'anchor'))}`: {decision}; "
                f"risk={_safe_text(audit.get('hallucination_risk', 'unknown'))}; "
                f"unsupported={len(audit.get('unsupported_changes', []))}; "
                f"API used={bool(audit.get('api_used', False))}; "
                f"fallback used={bool(audit.get('fallback_used', False))}."
            )
    else:
        lines.append("- This map does not contain correction-audit records.")

    charts = [Path(item) for item in chart_paths or []]
    lines.extend(["", "## Related Experiment Evidence", ""])
    if charts:
        for chart in charts:
            lines.append(f"- `{_safe_text(chart.as_posix())}`")
    else:
        lines.append("- No chart paths were attached to this preview.")

    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "This report separates real public-data artifacts, controlled text "
            "fixtures, reference-assisted oracle evidence, and automatic model "
            "outputs. Controlled fixtures are safety tests, not audio "
            "generalization results.",
            "",
        ]
    )
    return "\n".join(lines)


def export_detective_report(
    conversation_map: dict[str, Any],
    output_dir: str | Path = DEFAULT_REPORT_DIR,
    chart_paths: Iterable[str | Path] | None = None,
) -> Path:
    """Write a Markdown report to a safe clip-derived filename."""

    clip_id = str(conversation_map.get("clip_id", "unknown_clip"))
    safe_id = re.sub(r"[^A-Za-z0-9._-]+", "_", clip_id).strip("._") or "unknown_clip"
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{safe_id}_detective_report.md"
    destination.write_text(
        build_detective_report(conversation_map, chart_paths),
        encoding="utf-8",
    )
    return destination
