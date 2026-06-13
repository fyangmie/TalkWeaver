"""Explicit TalkWeaver workflow variants for controlled ablation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from backend.constrained_correction import (
    apply_constrained_correction,
    build_structured_correction_prompt,
)
from backend.conversation_map import (
    build_speaker_cards,
    deduplicate_events,
    reference_events_to_schema,
)
from backend.events import (
    attach_event_evidence,
    detect_interruption_events,
    detect_overlap_events,
)
from backend.schemas import ConversationMap, TemporalAnchor
from backend.summarizer import summarize_segments
from backend.temporal_anchor import build_temporal_anchors
from backend.term_rescue import retrieve_term_candidates


VARIANT_NAMES = (
    "asr_only",
    "temporal_anchor_only",
    "reference_speaker_time",
    "overlap_aware",
    "term_rescue",
    "constrained_correction",
    "full_talkweaver",
)


def _segments_text(segments: list[dict[str, Any]]) -> str:
    return " ".join(
        str(segment.get("text", "")).strip()
        for segment in segments
        if str(segment.get("text", "")).strip()
    ).strip()


def _span(segments: list[dict[str, Any]]) -> tuple[float, float]:
    if not segments:
        return 0.0, 0.0
    return (
        min(float(segment.get("start", 0.0)) for segment in segments),
        max(float(segment.get("end", 0.0)) for segment in segments),
    )


def _metadata(
    clip_metadata: dict[str, Any],
    *,
    variant: str,
    uses_reference_speaker_time: bool,
    uses_overlap_events: bool,
    uses_term_rescue: bool,
    uses_correction: bool,
    correction_mode: str = "disabled",
) -> dict[str, Any]:
    return {
        **clip_metadata,
        "schema_version": "talkweaver.conversation_map.v1",
        "workflow": "talkweaver_workflow_ablation",
        "variant": variant,
        "asr_mode": "real_prediction_json",
        "diarization_mode": (
            "reference" if uses_reference_speaker_time else "none"
        ),
        "llm_mode": correction_mode,
        "uses_real_asr_prediction": True,
        "uses_reference_speaker_time": uses_reference_speaker_time,
        "uses_overlap_events": uses_overlap_events,
        "uses_term_rescue": uses_term_rescue,
        "uses_correction": uses_correction,
        "reference_assisted": uses_reference_speaker_time,
        "is_mock": False,
        "evaluation_scope": "small_subset",
        "claim_scope": (
            "Workflow ablation over fixed real ASR predictions. Reference "
            "speaker-time evidence is oracle evidence, not automatic "
            "diarization performance."
        ),
    }


def _empty_summary() -> dict[str, Any]:
    return {
        "mode": "disabled_for_ablation",
        "summary": "",
        "action_items": [],
        "keywords": [],
        "segment_count": 0,
    }


def _map(
    clip_metadata: dict[str, Any],
    *,
    variant: str,
    anchors: list[TemporalAnchor],
    events: list[Any] | None = None,
    term_rescues: list[Any] | None = None,
    correction_audits: list[Any] | None = None,
    speaker_cards: list[Any] | None = None,
    summary: dict[str, Any] | None = None,
    correction_mode: str = "disabled",
) -> ConversationMap:
    uses_reference = variant not in {
        "asr_only",
        "temporal_anchor_only",
    }
    return ConversationMap(
        clip_id=str(clip_metadata["clip_id"]),
        metadata=_metadata(
            clip_metadata,
            variant=variant,
            uses_reference_speaker_time=uses_reference,
            uses_overlap_events=variant
            in {
                "overlap_aware",
                "term_rescue",
                "constrained_correction",
                "full_talkweaver",
            },
            uses_term_rescue=variant
            in {
                "term_rescue",
                "constrained_correction",
                "full_talkweaver",
            },
            uses_correction=variant
            in {"constrained_correction", "full_talkweaver"},
            correction_mode=correction_mode,
        ),
        anchors=anchors,
        events=list(events or []),
        term_rescues=list(term_rescues or []),
        correction_audits=list(correction_audits or []),
        speaker_cards=list(speaker_cards or []),
        summary=summary if summary is not None else _empty_summary(),
    )


def _build_events(
    clip_id: str,
    speaker_turns: list[dict[str, Any]],
    reference_events: list[dict[str, Any]],
) -> list[Any]:
    events = [
        *reference_events_to_schema(reference_events, clip_id),
        *detect_overlap_events(speaker_turns, clip_id=clip_id),
        *detect_interruption_events(speaker_turns, clip_id=clip_id),
    ]
    return deduplicate_events(events)


def _build_anchors(
    clip_metadata: dict[str, Any],
    asr_segments: list[dict[str, Any]],
    speaker_turns: list[dict[str, Any]],
    events: list[Any],
) -> list[TemporalAnchor]:
    anchors = build_temporal_anchors(
        asr_segments,
        speaker_turns,
        [event.to_dict() for event in events],
        str(clip_metadata["clip_id"]),
        str(clip_metadata.get("language", "unknown")),
    )
    attach_event_evidence(events, anchors)
    for anchor in anchors:
        anchor.interruption = any(
            event.type == "interruption"
            and anchor.anchor_id in event.evidence_anchor_ids
            for event in events
        )
    return anchors


def _single_speaker_view(
    anchors: list[TemporalAnchor],
) -> list[TemporalAnchor]:
    """Remove overlap semantics for the speaker-time-only control."""

    for anchor in anchors:
        if anchor.speaker == "OVERLAP":
            selected = anchor.speakers[0] if anchor.speakers else "UNKNOWN"
            anchor.speaker = selected
            anchor.speakers = [selected] if selected != "UNKNOWN" else []
        anchor.overlap = False
        anchor.interruption = False
        anchor.needs_review = anchor.speaker == "UNKNOWN"
        anchor.notes = [
            note
            for note in anchor.notes
            if "simultaneous speaker evidence" not in note
        ]
    return anchors


def build_asr_only_variant(
    clip_metadata: dict[str, Any],
    asr_segments: list[dict[str, Any]],
    speaker_turns: list[dict[str, Any]] | None = None,
    reference_events: list[dict[str, Any]] | None = None,
    glossary_docs: str | Path | list[str | Path] | None = None,
    llm_config: dict[str, Any] | None = None,
) -> ConversationMap:
    """Build a raw transcript represented by one clip-level span."""

    del speaker_turns, reference_events, glossary_docs, llm_config
    start, end = _span(asr_segments)
    anchor = TemporalAnchor(
        anchor_id=f"{clip_metadata['clip_id']}_anchor_001",
        clip_id=str(clip_metadata["clip_id"]),
        start=start,
        end=end,
        speaker="UNKNOWN",
        speakers=[],
        raw_text=_segments_text(asr_segments),
        language=str(clip_metadata.get("language", "unknown")),
        confidence=0.0,
        asr_confidence=0.0,
        diarization_confidence=0.0,
        needs_review=False,
        notes=["Raw ASR-only control; no speaker/time evidence or audit."],
    )
    return _map(
        clip_metadata,
        variant="asr_only",
        anchors=[anchor],
    )


def build_temporal_anchor_only_variant(
    clip_metadata: dict[str, Any],
    asr_segments: list[dict[str, Any]],
    speaker_turns: list[dict[str, Any]] | None = None,
    reference_events: list[dict[str, Any]] | None = None,
    glossary_docs: str | Path | list[str | Path] | None = None,
    llm_config: dict[str, Any] | None = None,
) -> ConversationMap:
    """Group real ASR words by time while keeping speakers unknown."""

    del speaker_turns, reference_events, glossary_docs, llm_config
    anchors = _build_anchors(clip_metadata, asr_segments, [], [])
    return _map(
        clip_metadata,
        variant="temporal_anchor_only",
        anchors=anchors,
    )


def build_reference_speaker_time_variant(
    clip_metadata: dict[str, Any],
    asr_segments: list[dict[str, Any]],
    speaker_turns: list[dict[str, Any]] | None = None,
    reference_events: list[dict[str, Any]] | None = None,
    glossary_docs: str | Path | list[str | Path] | None = None,
    llm_config: dict[str, Any] | None = None,
) -> ConversationMap:
    """Add oracle speaker-time labels without overlap semantics."""

    del reference_events, glossary_docs, llm_config
    turns = list(speaker_turns or [])
    anchors = _single_speaker_view(
        _build_anchors(clip_metadata, asr_segments, turns, [])
    )
    return _map(
        clip_metadata,
        variant="reference_speaker_time",
        anchors=anchors,
    )


def build_overlap_aware_variant(
    clip_metadata: dict[str, Any],
    asr_segments: list[dict[str, Any]],
    speaker_turns: list[dict[str, Any]] | None = None,
    reference_events: list[dict[str, Any]] | None = None,
    glossary_docs: str | Path | list[str | Path] | None = None,
    llm_config: dict[str, Any] | None = None,
) -> ConversationMap:
    """Add overlap and conservative interruption event evidence."""

    del glossary_docs, llm_config
    turns = list(speaker_turns or [])
    events = _build_events(
        str(clip_metadata["clip_id"]),
        turns,
        list(reference_events or []),
    )
    anchors = _build_anchors(
        clip_metadata,
        asr_segments,
        turns,
        events,
    )
    return _map(
        clip_metadata,
        variant="overlap_aware",
        anchors=anchors,
        events=events,
    )


def build_term_rescue_variant(
    clip_metadata: dict[str, Any],
    asr_segments: list[dict[str, Any]],
    speaker_turns: list[dict[str, Any]] | None = None,
    reference_events: list[dict[str, Any]] | None = None,
    glossary_docs: str | Path | list[str | Path] | None = None,
    llm_config: dict[str, Any] | None = None,
) -> ConversationMap:
    """Retrieve term evidence without changing transcript text."""

    del llm_config
    overlap_map = build_overlap_aware_variant(
        clip_metadata,
        asr_segments,
        speaker_turns,
        reference_events,
    )
    candidates = retrieve_term_candidates(
        overlap_map.anchors,
        glossary_docs=glossary_docs,
    )
    return _map(
        clip_metadata,
        variant="term_rescue",
        anchors=overlap_map.anchors,
        events=overlap_map.events,
        term_rescues=candidates,
    )


def build_constrained_correction_variant(
    clip_metadata: dict[str, Any],
    asr_segments: list[dict[str, Any]],
    speaker_turns: list[dict[str, Any]] | None = None,
    reference_events: list[dict[str, Any]] | None = None,
    glossary_docs: str | Path | list[str | Path] | None = None,
    llm_config: dict[str, Any] | None = None,
) -> ConversationMap:
    """Apply per-anchor evidence-constrained correction and audit."""

    rescue_map = build_term_rescue_variant(
        clip_metadata,
        asr_segments,
        speaker_turns,
        reference_events,
        glossary_docs,
    )
    prompt = build_structured_correction_prompt(
        rescue_map.anchors,
        rescue_map.term_rescues,
        rescue_map.events,
    )
    anchors, audits, mode = apply_constrained_correction(
        rescue_map.anchors,
        rescue_map.term_rescues,
        rescue_map.events,
        llm_config=llm_config,
    )
    result = _map(
        clip_metadata,
        variant="constrained_correction",
        anchors=anchors,
        events=rescue_map.events,
        term_rescues=rescue_map.term_rescues,
        correction_audits=audits,
        correction_mode=mode,
    )
    result.metadata["structured_correction_prompt"] = prompt
    return result


def build_full_talkweaver_variant(
    clip_metadata: dict[str, Any],
    asr_segments: list[dict[str, Any]],
    speaker_turns: list[dict[str, Any]] | None = None,
    reference_events: list[dict[str, Any]] | None = None,
    glossary_docs: str | Path | list[str | Path] | None = None,
    llm_config: dict[str, Any] | None = None,
) -> ConversationMap:
    """Build the full auditable ConversationMap over fixed ASR evidence."""

    corrected = build_constrained_correction_variant(
        clip_metadata,
        asr_segments,
        speaker_turns,
        reference_events,
        glossary_docs,
        llm_config,
    )
    summary = summarize_segments(
        [anchor.to_dict() for anchor in corrected.anchors]
    )
    summary["workflow_note"] = (
        "Evidence-grounded extractive summary; no unsupported stance or "
        "claim generation."
    )
    result = _map(
        clip_metadata,
        variant="full_talkweaver",
        anchors=corrected.anchors,
        events=corrected.events,
        term_rescues=corrected.term_rescues,
        correction_audits=corrected.correction_audits,
        speaker_cards=build_speaker_cards(corrected.anchors),
        summary=summary,
        correction_mode=str(
            corrected.metadata.get("llm_mode", "rule_fallback")
        ),
    )
    result.metadata["structured_correction_prompt"] = corrected.metadata.get(
        "structured_correction_prompt",
        "",
    )
    return result


VARIANT_BUILDERS: dict[str, Callable[..., ConversationMap]] = {
    "asr_only": build_asr_only_variant,
    "temporal_anchor_only": build_temporal_anchor_only_variant,
    "reference_speaker_time": build_reference_speaker_time_variant,
    "overlap_aware": build_overlap_aware_variant,
    "term_rescue": build_term_rescue_variant,
    "constrained_correction": build_constrained_correction_variant,
    "full_talkweaver": build_full_talkweaver_variant,
}


def build_workflow_variant(
    variant: str,
    clip_metadata: dict[str, Any],
    asr_segments: list[dict[str, Any]],
    speaker_turns: list[dict[str, Any]] | None = None,
    reference_events: list[dict[str, Any]] | None = None,
    glossary_docs: str | Path | list[str | Path] | None = None,
    llm_config: dict[str, Any] | None = None,
) -> ConversationMap:
    """Dispatch one named workflow variant."""

    try:
        builder = VARIANT_BUILDERS[variant]
    except KeyError as exc:
        raise ValueError(f"Unknown workflow variant: {variant}") from exc
    return builder(
        clip_metadata,
        asr_segments,
        speaker_turns,
        reference_events,
        glossary_docs,
        llm_config,
    )
