"""JSON-serializable evidence schemas for the TalkWeaver core workflow."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


class JsonSerializable:
    """Provide deterministic dictionary and JSON conversion for dataclasses."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=indent,
        )


@dataclass
class ASRWord(JsonSerializable):
    word: str
    start: float
    end: float
    confidence: float = 0.0
    language: str = "unknown"


@dataclass
class SpeakerTurn(JsonSerializable):
    speaker: str
    start: float
    end: float
    confidence: float = 0.0
    source: str = "unknown"


@dataclass
class TemporalAnchor(JsonSerializable):
    anchor_id: str
    clip_id: str
    start: float
    end: float
    speaker: str
    speakers: list[str]
    raw_text: str
    corrected_text: str = ""
    language: str = "unknown"
    overlap: bool = False
    interruption: bool = False
    confidence: float = 0.0
    asr_confidence: float = 0.0
    diarization_confidence: float = 0.0
    retrieved_terms: list[str] = field(default_factory=list)
    correction_evidence: list[dict[str, Any]] = field(default_factory=list)
    unsupported_changes: list[str] = field(default_factory=list)
    needs_review: bool = False
    notes: list[str] = field(default_factory=list)


@dataclass
class ConversationEvent(JsonSerializable):
    event_id: str
    clip_id: str
    type: str
    start: float
    end: float
    speakers: list[str]
    description: str
    evidence_anchor_ids: list[str] = field(default_factory=list)
    severity: str = "low"
    notes: list[str] = field(default_factory=list)


@dataclass
class TermRescueCandidate(JsonSerializable):
    term_id: str
    canonical: str
    spoken_forms: list[str]
    asr_error_forms: list[str]
    retrieved_score: float
    retrieval_method: str
    evidence_anchor_ids: list[str] = field(default_factory=list)


@dataclass
class CorrectionAudit(JsonSerializable):
    anchor_id: str
    raw_text: str
    corrected_text: str
    changed_tokens: list[str] = field(default_factory=list)
    supported_changes: list[str] = field(default_factory=list)
    unsupported_changes: list[str] = field(default_factory=list)
    hallucination_risk: str = "low"
    needs_review: bool = False
    evidence: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class SpeakerCard(JsonSerializable):
    speaker: str
    speaking_time_seconds: float
    top_terms: list[str] = field(default_factory=list)
    main_claims: list[dict[str, Any]] = field(default_factory=list)
    action_items: list[dict[str, Any]] = field(default_factory=list)
    stance_summary: str = ""
    evidence_anchor_ids: list[str] = field(default_factory=list)


@dataclass
class ConversationMap(JsonSerializable):
    clip_id: str
    metadata: dict[str, Any]
    anchors: list[TemporalAnchor] = field(default_factory=list)
    events: list[ConversationEvent] = field(default_factory=list)
    term_rescues: list[TermRescueCandidate] = field(default_factory=list)
    correction_audits: list[CorrectionAudit] = field(default_factory=list)
    speaker_cards: list[SpeakerCard] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
