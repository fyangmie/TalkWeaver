"""Proposal-time safety gate for selective ASR correction.

EccoGate is a deterministic feasibility-pilot policy. It uses only evidence
available before a correction decision: retrieval support, overlap and
speaker uncertainty, and edit risk. It does not inspect pilot gold labels or
call an LLM.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Iterable


DECISIONS = ("accept", "reject", "needs_review")
TOKEN_PATTERN = re.compile(r"\w+(?:[.-]\w+)*", flags=re.UNICODE)
SPEAKER_PATTERN = re.compile(
    r"\b(speaker|participant)\s*([a-z]|\d+)\b",
    flags=re.IGNORECASE,
)

TERM_CONTEXT_CUES = {
    "pyannote": {
        "speaker",
        "diarization",
        "audio",
        "meeting",
        "turn",
    },
    "diarization": {
        "speaker",
        "meeting",
        "turn",
        "segmentation",
        "who spoke",
    },
    "rag": {
        "retrieval",
        "glossary",
        "knowledge",
        "asr",
        "correction",
    },
    "wer": {
        "asr",
        "metric",
        "error rate",
        "transcription",
        "benchmark",
    },
    "der": {
        "diarization",
        "metric",
        "speaker error",
        "benchmark",
    },
    "temporal anchor": {
        "alignment",
        "timestamp",
        "speaker-time",
        "timeline",
        "anchor",
    },
    "tagspeech": {
        "paper",
        "temporal",
        "speech",
        "research",
    },
    "dm-asr": {
        "paper",
        "speaker",
        "asr",
        "research",
    },
    "qwen": {
        "model",
        "llm",
        "api",
        "language model",
    },
}
AMBIGUITY_CUES = (
    "ambiguous",
    "unclear",
    "underspecified",
    "weak",
    "not explicit",
    "not explicitly",
    "not identified",
    "not sufficiently",
    "does not identify",
    "but not ",
    "no model",
    "no speaker",
    "lacks ",
    "without naming",
    "without specifying",
    "without speaker",
    "may be",
    "might be",
    "is likely",
    "is plausible",
)
CONTRADICTION_CUES = (
    "physical",
    "ordinary",
    "greeting",
    "location question",
    "place name",
    "musician",
    "emotional",
    "unrelated",
    "direct message",
    "not a ",
    "no source evidence",
    "no timing evidence",
    "no result",
    "no guarantee",
    "no comparison",
    "no correction is needed",
)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().casefold() in {"1", "true", "yes", "y"}


def _as_terms(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = [part.strip() for part in text.split(",")]
    if isinstance(parsed, str):
        parsed = [parsed]
    if not isinstance(parsed, Iterable):
        return []
    return [str(item).strip() for item in parsed if str(item).strip()]


def _tokens(text: str) -> list[str]:
    return [match.group(0).casefold() for match in TOKEN_PATTERN.finditer(text)]


def _changed_tokens(raw_text: str, corrected_text: str) -> tuple[list[str], list[str]]:
    raw = _tokens(raw_text)
    corrected = _tokens(corrected_text)
    removed: list[str] = []
    added: list[str] = []
    matcher = SequenceMatcher(a=raw, b=corrected)
    for operation, start_a, end_a, start_b, end_b in matcher.get_opcodes():
        if operation in {"replace", "delete"}:
            removed.extend(raw[start_a:end_a])
        if operation in {"replace", "insert"}:
            added.extend(corrected[start_b:end_b])
    return removed, added


def _contains_term(text: str, term: str) -> bool:
    normalized_text = " ".join(_tokens(text))
    normalized_term = " ".join(_tokens(term))
    return bool(normalized_term and normalized_term in normalized_text)


def _term_has_context(term: str, context: str) -> bool:
    normalized_term = term.casefold()
    normalized_context = context.casefold()
    matching_key = next(
        (
            key
            for key in TERM_CONTEXT_CUES
            if key in normalized_term or normalized_term in key
        ),
        "",
    )
    if not matching_key:
        return True
    return any(cue in normalized_context for cue in TERM_CONTEXT_CUES[matching_key])


def _speaker_attribution_changed(raw_text: str, corrected_text: str) -> bool:
    raw_speakers = {
        f"{match.group(1).casefold()} {match.group(2).casefold()}"
        for match in SPEAKER_PATTERN.finditer(raw_text)
    }
    corrected_speakers = {
        f"{match.group(1).casefold()} {match.group(2).casefold()}"
        for match in SPEAKER_PATTERN.finditer(corrected_text)
    }
    return bool(raw_speakers != corrected_speakers and (raw_speakers or corrected_speakers))


@dataclass(frozen=True)
class EccoGatePrediction:
    """One deterministic selective-correction decision."""

    decision: str
    support_score: float
    risk_score: float
    explanation: str

    def to_dict(self) -> dict[str, str | float]:
        return {
            "decision": self.decision,
            "support_score": self.support_score,
            "risk_score": self.risk_score,
            "explanation": self.explanation,
        }


def score_correction_proposal(proposal: dict[str, Any]) -> EccoGatePrediction:
    """Score a correction proposal using pre-decision evidence only."""

    raw_text = str(proposal.get("raw_asr_text", "")).strip()
    corrected_text = str(proposal.get("proposed_corrected_text", "")).strip()
    context = str(proposal.get("context", "")).strip()
    retrieved_terms = _as_terms(proposal.get("retrieved_terms"))
    overlap = _as_bool(proposal.get("overlap_flag"))
    heavy_overlap = _as_bool(proposal.get("heavy_overlap_flag"))
    speaker_ambiguity = _as_bool(proposal.get("speaker_ambiguity_flag"))
    partial_utterance = _as_bool(proposal.get("partial_utterance_flag"))

    if not raw_text or not corrected_text:
        return EccoGatePrediction(
            decision="reject",
            support_score=0.0,
            risk_score=1.0,
            explanation="Reject: empty raw or proposed text is not auditable.",
        )

    raw_tokens = _tokens(raw_text)
    corrected_tokens = _tokens(corrected_text)
    removed, added = _changed_tokens(raw_text, corrected_text)
    no_change = raw_tokens == corrected_tokens
    changed_ratio = (
        (len(removed) + len(added)) / max(len(raw_tokens) + len(corrected_tokens), 1)
    )
    length_growth = max(
        0.0,
        (len(corrected_tokens) - len(raw_tokens)) / max(len(raw_tokens), 1),
    )

    supported_terms = [
        term
        for term in retrieved_terms
        if _contains_term(corrected_text, term) and _term_has_context(term, context)
    ]
    weak_terms = [
        term
        for term in retrieved_terms
        if _contains_term(corrected_text, term) and term not in supported_terms
    ]
    retrieved_tokens = {
        token for term in retrieved_terms for token in _tokens(term)
    }
    raw_token_set = set(raw_tokens)
    added_supported = [
        token
        for token in added
        if token in retrieved_tokens or token in raw_token_set
    ]
    support_ratio = len(added_supported) / len(added) if added else 1.0

    support_score = 0.15
    reasons: list[str] = []
    normalized_context = context.casefold()
    ambiguity_evidence = any(
        cue in normalized_context for cue in AMBIGUITY_CUES
    )
    contradictory_context = any(
        cue in normalized_context for cue in CONTRADICTION_CUES
    )
    if no_change:
        support_score = 0.9
        reasons.append("the proposal preserves the ASR evidence")
    else:
        if supported_terms:
            support_score += min(0.55, 0.35 + 0.1 * len(supported_terms))
            reasons.append(
                "retrieval and context support " + ", ".join(supported_terms)
            )
        if support_ratio >= 0.8:
            support_score += 0.2
            reasons.append("most inserted tokens are grounded")
        elif support_ratio >= 0.5:
            support_score += 0.1
        if not added and removed:
            support_score += 0.05
    if contradictory_context and not no_change:
        support_score -= 0.4
    elif ambiguity_evidence and not no_change:
        support_score -= 0.15

    risk_score = 0.05
    risks: list[str] = []
    if overlap:
        risk_score += 0.12
        risks.append("overlap")
    if heavy_overlap:
        risk_score += 0.28
        risks.append("heavy overlap")
    if speaker_ambiguity:
        risk_score += 0.28
        risks.append("speaker ambiguity")
    if partial_utterance:
        risk_score += 0.2
        risks.append("partial utterance")
    if changed_ratio >= 0.55:
        risk_score += 0.2
        risks.append("large edit")
    elif changed_ratio >= 0.3:
        risk_score += 0.1
    if length_growth >= 0.5:
        risk_score += 0.35
        risks.append("substantial unsupported expansion")
    elif length_growth >= 0.2:
        risk_score += 0.15
    if weak_terms:
        risk_score += 0.25
        risks.append("retrieval term lacks matching context")
    if ambiguity_evidence:
        risk_score += 0.28
        risks.append("context explicitly signals ambiguity")
    if contradictory_context and not no_change:
        risk_score += 0.5
        risks.append("context contradicts the proposed domain edit")
    if added and support_ratio < 0.5:
        risk_score += 0.25
        risks.append("inserted tokens are weakly grounded")
    attribution_changed = _speaker_attribution_changed(raw_text, corrected_text)
    if attribution_changed:
        risk_score += 0.55
        risks.append("speaker attribution changed")

    support_score = round(min(max(support_score, 0.0), 1.0), 3)
    risk_score = round(min(max(risk_score, 0.0), 1.0), 3)
    if no_change:
        decision = "accept"
    elif attribution_changed or risk_score >= 0.78:
        decision = "reject"
    elif support_score < 0.3 and risk_score >= 0.45:
        decision = "reject"
    elif ambiguity_evidence:
        decision = "needs_review"
    elif risk_score >= 0.4 or support_score < 0.58:
        decision = "needs_review"
    else:
        decision = "accept"

    support_text = "; ".join(reasons) if reasons else "no strong support evidence"
    risk_text = ", ".join(dict.fromkeys(risks)) if risks else "low edit risk"
    return EccoGatePrediction(
        decision=decision,
        support_score=support_score,
        risk_score=risk_score,
        explanation=(
            f"{decision}: {support_text}; observed risk: {risk_text}."
        ),
    )
