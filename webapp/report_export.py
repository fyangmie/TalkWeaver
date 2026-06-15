"""Markdown detective-report generation from a ConversationMap."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

from webapp.data_loader import (
    build_clip_detective_summary,
    build_speaker_evidence_cards,
    get_best_evidence_gate_validation,
    get_event_investigation_rows,
    resolve_local_audio_path,
)

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
    speaker_cards = build_speaker_evidence_cards(conversation_map)
    event_rows = get_event_investigation_rows(conversation_map)
    detective = build_clip_detective_summary(conversation_map)
    audio_path = resolve_local_audio_path(conversation_map)
    review_anchors = [item for item in anchors if item.get("needs_review")]
    overlap_events = [item for item in event_rows if item.get("type") == "overlap"]

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
        f"- Local audio available: {audio_path is not None}",
        f"- Audio metadata path: {_safe_text(metadata.get('audio_path', 'not specified'))}",
        "",
        "## Detective Summary",
        "",
        f"- **What happened?** {_safe_text(detective['what_happened'])}",
        f"- **Who spoke?** {_safe_text(detective['who_spoke'])}",
        f"- **Where did cross-talk happen?** {_safe_text(detective['where_cross_talk'])}",
        f"- **What needs review?** {_safe_text(detective['what_needs_review'])}",
        f"- **Were corrections rejected?** {_safe_text(detective['corrections_rejected'])}",
        f"- **Evidence scope:** {_safe_text(detective['evidence_scope'])}",
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
    evidence_gate = get_best_evidence_gate_validation()
    gate_section = ["## EvidenceGate Safety Model", ""]
    if evidence_gate:
        gate_section.extend(
            [
                f"- Best strict independent-heldout model: {_safe_text(evidence_gate.get('feature_set'))} / {_safe_text(evidence_gate.get('model_name'))}",
                f"- Macro F1: {float(evidence_gate.get('macro_f1', 0.0)):.3f}",
                f"- False-accept rate: {float(evidence_gate.get('false_accept_rate', 0.0)):.3f}",
                f"- Unsafe-accept rate: {float(evidence_gate.get('unsafe_accept_rate', 0.0)):.3f}",
                f"- Needs-review recall: {float(evidence_gate.get('needs_review_recall', 0.0)):.3f}",
                "- Scope: independent controlled text proposals; performance is weak and does not establish real-audio generalization.",
                "",
            ]
        )
    else:
        gate_section.extend(
            [
                "- EvidenceGate results are not available. Regenerate them with the Phase 2H training commands.",
                "",
            ]
        )
    temporal_index = lines.index("## Temporal Anchors")
    lines[temporal_index:temporal_index] = gate_section
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

    lines.extend(
        [
            "",
            "## Event Investigation",
            "",
            "| Event | Type | Time | Speakers | Severity | Review | Evidence anchors | Related evidence | Audio window |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    if event_rows:
        for event in event_rows:
            lines.append(
                "| "
                f"{_safe_text(event.get('event_id', 'event'))} | "
                f"{_safe_text(event.get('type', 'event'))} | "
                f"{float(event.get('start', 0.0)):.2f}-"
                f"{float(event.get('end', 0.0)):.2f}s | "
                f"{_safe_text(', '.join(event.get('speakers', [])))} | "
                f"{_safe_text(event.get('severity', 'unknown'))} | "
                f"{bool(event.get('needs_review'))} | "
                f"{_safe_text(', '.join(event.get('evidence_anchor_ids', [])))} | "
                f"{_safe_text(event.get('related_raw_text', ''))} | "
                f"{float(event.get('audio_window_start', 0.0)):.2f}-"
                f"{float(event.get('audio_window_end', 0.0)):.2f}s |"
            )
    else:
        lines.append(
            "| No event evidence | - | - | - | - | - | - | - | - |"
        )

    lines.extend(["", "## Overlap And Interruption Warnings", ""])
    if event_rows:
        for event in event_rows:
            lines.append(
                f"- **{_safe_text(event.get('type', 'event'))}** "
                f"{float(event.get('start', 0.0)):.2f}-"
                f"{float(event.get('end', 0.0)):.2f}s, "
                f"speakers: {_safe_text(', '.join(event.get('speakers', [])))}. "
                f"{_safe_text(event.get('description', ''))} "
                f"Review={bool(event.get('needs_review'))}."
            )
    else:
        lines.append("- No event evidence is present in this ConversationMap.")

    lines.extend(["", "## Speaker Evidence Cards", ""])
    if speaker_cards:
        for card in speaker_cards:
            lines.extend(
                [
                    f"### {_safe_text(card['speaker_id'])}",
                    "",
                    f"- Speaking time: {float(card['speaking_time_seconds']):.2f}s",
                    f"- Anchors: {card['num_anchors']}",
                    f"- Overlap anchors: {card['num_overlap_anchors']}",
                    f"- Needs-review anchors: {card['needs_review_anchors']}",
                    f"- Summary mode: {_safe_text(card['summary_mode'])}",
                    f"- Top terms: {_safe_text(', '.join(card['top_terms']) or 'none')}",
                    f"- Evidence anchors: {_safe_text(', '.join(card['evidence_anchor_ids']) or 'none')}",
                    f"- Evidence note: {_safe_text(card['stance_summary'])}",
                    "",
                ]
            )
            for quote in card["representative_raw_quotes"][:3]:
                lines.append(
                    f"- Raw quote `{_safe_text(quote.get('anchor_id', 'anchor'))}` "
                    f"({float(quote.get('start', 0.0)):.2f}-"
                    f"{float(quote.get('end', 0.0)):.2f}s): "
                    f"{_safe_text(quote.get('text', ''))}"
                )
            lines.append("")
    else:
        lines.append("- No named speaker evidence is present.")

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
    rescued_anchors = [
        anchor
        for anchor in anchors
        if anchor.get("retrieved_terms")
        or (
            anchor.get("corrected_text")
            and anchor.get("corrected_text") != anchor.get("raw_text")
        )
    ]
    if rescued_anchors:
        lines.append("")
        lines.append("### Anchor-level correction examples")
        lines.append("")
        for anchor in rescued_anchors[:5]:
            lines.append(
                f"- `{_safe_text(anchor.get('anchor_id', 'anchor'))}`: "
                f"raw={_safe_text(anchor.get('raw_text', ''))}; "
                f"corrected={_safe_text(anchor.get('corrected_text', ''))}; "
                f"terms={_safe_text(', '.join(anchor.get('retrieved_terms', [])) or 'none')}."
            )

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
