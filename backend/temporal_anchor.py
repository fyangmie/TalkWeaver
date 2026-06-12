"""Build speaker-time evidence anchors from ASR and diarization outputs."""

from __future__ import annotations

from statistics import mean
from typing import Any

from backend.alignment import align_segments
from backend.schemas import TemporalAnchor


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return vars(value)


def _event_intersects(
    start: float,
    end: float,
    event: Any,
    *,
    event_type: str | None = None,
) -> bool:
    payload = _as_dict(event)
    if event_type and str(payload.get("type")) != event_type:
        return False
    return min(end, float(payload["end"])) > max(
        start, float(payload["start"])
    )


def _event_speakers(
    start: float,
    end: float,
    events: list[Any],
) -> list[str]:
    return sorted(
        {
            str(speaker)
            for event in events
            if _event_intersects(start, end, event, event_type="overlap")
            for speaker in _as_dict(event).get("speakers", [])
        }
    )


def _segment_asr_confidence(
    segment: dict[str, Any],
    start: float,
    end: float,
) -> float:
    probabilities = [
        float(word.get("probability", word.get("confidence", 0.0)))
        for word in segment.get("words", [])
        if word.get("start") is not None
        and word.get("end") is not None
        and min(end, float(word["end"])) > max(start, float(word["start"]))
        and word.get("probability", word.get("confidence")) is not None
    ]
    if probabilities:
        return round(mean(probabilities), 3)
    average_log_probability = segment.get("avg_logprob")
    if average_log_probability is not None:
        return round(max(0.0, min(1.0, 1.0 + float(average_log_probability))), 3)
    return 0.75


def _diarization_confidence(
    speakers: list[str],
    speaker_turns: list[dict[str, Any]],
) -> float:
    values = [
        float(turn.get("confidence", 0.9))
        for turn in speaker_turns
        if str(turn.get("speaker")) in speakers
    ]
    return round(min(values), 3) if values else 0.5


def _source_segment_for_anchor(
    anchor: dict[str, Any],
    asr_segments: list[dict[str, Any]],
) -> dict[str, Any]:
    candidates = [
        segment
        for segment in asr_segments
        if min(float(anchor["end"]), float(segment["end"]))
        > max(float(anchor["start"]), float(segment["start"]))
    ]
    return candidates[0] if candidates else {}


def build_temporal_anchors(
    asr_segments: list[dict[str, Any]],
    speaker_turns: list[dict[str, Any]],
    overlap_events: list[Any] | None,
    clip_id: str,
    language: str,
) -> list[TemporalAnchor]:
    """Build immutable speaker/time anchors around existing v0 alignment."""

    normalized_turns = [
        {
            **_as_dict(turn),
            "speaker": str(_as_dict(turn).get("speaker", "UNKNOWN")),
            "start": float(_as_dict(turn).get("start", 0.0)),
            "end": float(_as_dict(turn).get("end", 0.0)),
        }
        for turn in speaker_turns
    ]
    events = list(overlap_events or [])
    aligned = align_segments(asr_segments, normalized_turns)
    anchors: list[TemporalAnchor] = []
    for index, segment in enumerate(aligned, start=1):
        start = float(segment["start"])
        end = float(segment["end"])
        event_speakers = _event_speakers(start, end, events)
        overlap = bool(segment.get("overlap")) or bool(event_speakers)
        speakers = sorted(
            set(segment.get("speakers", [])) | set(event_speakers)
        )
        speaker = (
            "OVERLAP"
            if overlap and len(speakers) > 1
            else str(segment.get("speaker", "UNKNOWN"))
        )
        interruption = any(
            _event_intersects(start, end, event, event_type="interruption")
            for event in events
        )
        source_segment = _source_segment_for_anchor(segment, asr_segments)
        asr_confidence = _segment_asr_confidence(
            source_segment, start, end
        )
        diarization_confidence = _diarization_confidence(
            speakers or ([speaker] if speaker != "UNKNOWN" else []),
            normalized_turns,
        )
        confidence = min(
            float(segment.get("confidence", 0.5)),
            asr_confidence,
            diarization_confidence,
        )
        if overlap:
            confidence = min(confidence, 0.55)
        notes = []
        if speaker == "UNKNOWN":
            notes.append("No speaker turn contained the aligned midpoint.")
        if overlap:
            notes.append("Anchor intersects simultaneous speaker evidence.")
        anchors.append(
            TemporalAnchor(
                anchor_id=f"{clip_id}_anchor_{index:03d}",
                clip_id=clip_id,
                start=round(start, 3),
                end=round(end, 3),
                speaker=speaker,
                speakers=speakers,
                raw_text=str(segment.get("raw_text", "")).strip(),
                language=language,
                overlap=overlap,
                interruption=interruption,
                confidence=round(confidence, 3),
                asr_confidence=asr_confidence,
                diarization_confidence=diarization_confidence,
                needs_review=overlap or speaker == "UNKNOWN",
                notes=notes,
            )
        )
    return anchors
