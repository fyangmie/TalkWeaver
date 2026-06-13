"""Build the auditable ConversationMap used by AI Meeting Detective."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from backend.config import ROOT_DIR
from backend.constrained_correction import (
    apply_constrained_correction,
    build_structured_correction_prompt,
)
from backend.events import (
    attach_event_evidence,
    detect_interruption_events,
    detect_overlap_events,
)
from backend.export import write_json
from backend.schemas import (
    ConversationEvent,
    ConversationMap,
    SpeakerCard,
)
from backend.summarizer import summarize_segments
from backend.temporal_anchor import build_temporal_anchors
from backend.term_rescue import retrieve_term_candidates


ACTION_PATTERN = re.compile(
    r"\b(?:we\s+)?(?:should|need to|must|will)\s+(.+?)(?:[.!?]|$)",
    re.IGNORECASE,
)


def _extract_payload(value: Any, key: str) -> tuple[list[dict[str, Any]], str]:
    if isinstance(value, dict):
        return list(value.get(key, [])), str(value.get("mode", "unknown"))
    return list(value or []), "provided"


def _asr_mode_label(mode: str) -> str:
    if mode.startswith("mock"):
        return "mock"
    if mode == "reference":
        return "reference"
    if mode == "real_prediction_json":
        return "real_prediction_json"
    if mode == "faster_whisper":
        return "real"
    return "unknown"


def _diarization_mode_label(mode: str) -> str:
    if mode.startswith("mock"):
        return "mock"
    if mode == "reference":
        return "reference"
    if mode == "none":
        return "none"
    if mode == "pyannote":
        return "pyannote"
    return "unknown"


def _llm_mode_label(mode: str) -> str:
    if mode == "rule_fallback" or "rule_based" in mode:
        return "rule_fallback"
    if mode in {"llm", "llm_with_rule_fallback"}:
        return mode
    if mode.startswith("api_"):
        return "api_llm"
    return mode


def reference_events_to_schema(
    events: list[dict[str, Any]] | None,
    clip_id: str,
) -> list[ConversationEvent]:
    converted: list[ConversationEvent] = []
    for index, event in enumerate(events or [], start=1):
        converted.append(
            ConversationEvent(
                event_id=str(
                    event.get("event_id")
                    or f"{clip_id}_{event.get('type', 'event')}_ref_{index:03d}"
                ),
                clip_id=clip_id,
                type=str(event.get("type", "overlap")),
                start=float(event["start"]),
                end=float(event["end"]),
                speakers=[str(value) for value in event.get("speakers", [])],
                description=str(
                    event.get("description")
                    or "Reference-provided conversation event."
                ),
                severity=str(event.get("severity", "medium")),
                notes=[
                    f"source={event.get('annotation_source', 'provided')}",
                ],
            )
        )
    return converted


def deduplicate_events(
    events: list[ConversationEvent],
) -> list[ConversationEvent]:
    deduplicated: list[ConversationEvent] = []
    seen: set[tuple[str, float, float, tuple[str, ...]]] = set()
    for event in events:
        key = (
            event.type,
            round(event.start, 3),
            round(event.end, 3),
            tuple(sorted(event.speakers)),
        )
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(event)
    return sorted(
        deduplicated,
        key=lambda event: (event.start, event.end, event.type),
    )


def build_speaker_cards(
    anchors: list[Any],
) -> list[SpeakerCard]:
    speaking_time: defaultdict[str, float] = defaultdict(float)
    terms: defaultdict[str, Counter[str]] = defaultdict(Counter)
    claims: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    actions: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    evidence_ids: defaultdict[str, list[str]] = defaultdict(list)

    for anchor in anchors:
        speakers = anchor.speakers or [anchor.speaker]
        valid_speakers = [
            speaker
            for speaker in speakers
            if speaker not in {"UNKNOWN", "OVERLAP"}
        ]
        text = anchor.corrected_text or anchor.raw_text
        for speaker in valid_speakers:
            speaking_time[speaker] += max(0.0, anchor.end - anchor.start)
            terms[speaker].update(anchor.retrieved_terms)
            evidence_ids[speaker].append(anchor.anchor_id)
            if len(claims[speaker]) < 3 and text.strip():
                claims[speaker].append(
                    {
                        "text": text.strip(),
                        "anchor_id": anchor.anchor_id,
                        "start": anchor.start,
                        "end": anchor.end,
                        "mode": "extractive_fallback",
                    }
                )
            for match in ACTION_PATTERN.finditer(text):
                actions[speaker].append(
                    {
                        "text": match.group(1).strip().rstrip(".") + ".",
                        "anchor_id": anchor.anchor_id,
                        "uncertain": anchor.overlap,
                    }
                )

    return [
        SpeakerCard(
            speaker=speaker,
            speaking_time_seconds=round(duration, 3),
            top_terms=[
                term for term, _count in terms[speaker].most_common(5)
            ],
            main_claims=claims[speaker],
            action_items=actions[speaker],
            stance_summary=(
                "Extractive evidence summary only; stance inference is "
                "deferred until a constrained, cited method is implemented."
            ),
            evidence_anchor_ids=list(dict.fromkeys(evidence_ids[speaker])),
        )
        for speaker, duration in sorted(speaking_time.items())
    ]


def save_conversation_map(
    conversation_map: ConversationMap,
    output_dir: str | Path | None = None,
) -> Path:
    directory = (
        Path(output_dir)
        if output_dir is not None
        else ROOT_DIR / "outputs" / "conversation_maps"
    )
    return write_json(
        directory / f"{conversation_map.clip_id}_conversation_map.json",
        conversation_map.to_dict(),
    )


def build_conversation_map(
    clip_metadata: dict[str, Any],
    asr_output: Any,
    diarization_output: Any,
    overlap_events: list[dict[str, Any]] | None,
    glossary_docs: str | Path | list[str | Path] | None,
    llm_config: dict[str, Any] | None,
) -> ConversationMap:
    """Run the temporal-anchor evidence-grounded correction workflow."""

    clip_id = str(clip_metadata["clip_id"])
    language = str(clip_metadata.get("language", "unknown"))
    asr_segments, asr_mode = _extract_payload(asr_output, "segments")
    speaker_turns, diarization_mode = _extract_payload(
        diarization_output, "turns"
    )
    reference_events = reference_events_to_schema(overlap_events, clip_id)
    detected_overlap = detect_overlap_events(
        speaker_turns, clip_id=clip_id
    )
    detected_interruptions = detect_interruption_events(
        speaker_turns, clip_id=clip_id
    )
    events = deduplicate_events(
        [*reference_events, *detected_overlap, *detected_interruptions]
    )
    anchors = build_temporal_anchors(
        asr_segments,
        speaker_turns,
        [event.to_dict() for event in events],
        clip_id,
        language,
    )
    attach_event_evidence(events, anchors)
    for anchor in anchors:
        anchor.interruption = any(
            event.type == "interruption"
            and anchor.anchor_id in event.evidence_anchor_ids
            for event in events
        )
    term_candidates = retrieve_term_candidates(
        anchors,
        glossary_docs=glossary_docs,
    )
    prompt = build_structured_correction_prompt(
        anchors, term_candidates, events
    )
    anchors, audits, correction_mode = apply_constrained_correction(
        anchors,
        term_candidates,
        events,
        llm_config=llm_config,
    )
    segment_dicts = [anchor.to_dict() for anchor in anchors]
    summary = summarize_segments(segment_dicts)
    summary["workflow_note"] = (
        "Evidence-grounded extractive summary; no unsupported stance or "
        "claim generation."
    )
    metadata = {
        **clip_metadata,
        "schema_version": "talkweaver.conversation_map.v1",
        "workflow": "temporal_anchor_evidence_grounded_correction",
        "asr_mode": _asr_mode_label(asr_mode),
        "diarization_mode": _diarization_mode_label(diarization_mode),
        "llm_mode": _llm_mode_label(correction_mode),
        "asr_backend_mode": asr_mode,
        "diarization_backend_mode": diarization_mode,
        "correction_backend_mode": correction_mode,
        "llm_provider": audits[0].llm_provider if audits else "",
        "llm_model": audits[0].llm_model if audits else "",
        "llm_prompt_version": (
            audits[0].prompt_version if audits else ""
        ),
        "llm_temperature": audits[0].temperature if audits else 0.0,
        "llm_api_used": any(audit.api_used for audit in audits),
        "llm_fallback_used": any(
            audit.fallback_used for audit in audits
        ),
        "reference_assisted": (
            asr_mode == "reference" or diarization_mode == "reference"
        ),
        "is_mock": asr_mode.startswith("mock")
        or diarization_mode.startswith("mock"),
        "structured_correction_prompt": prompt,
        "claim_scope": (
            "Paper-inspired proxy workflow. This is not a reproduction of "
            "DiarizationLM, DM-ASR, Diarization-Aware MS-ASR, or TagSpeech."
        ),
    }
    return ConversationMap(
        clip_id=clip_id,
        metadata=metadata,
        anchors=anchors,
        events=events,
        term_rescues=term_candidates,
        correction_audits=audits,
        speaker_cards=build_speaker_cards(anchors),
        summary=summary,
    )
