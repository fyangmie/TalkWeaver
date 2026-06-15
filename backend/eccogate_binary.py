"""Binary proposal-time safety gate for ASR corrections."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from backend.eccogate import score_correction_proposal


WHITESPACE_PATTERN = re.compile(r"\s+")


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().casefold() in {"1", "true", "yes", "y"}


def _normalized_text(value: Any) -> str:
    return WHITESPACE_PATTERN.sub(" ", str(value or "").strip().casefold())


@dataclass(frozen=True)
class BinaryEccoGatePrediction:
    """Binary safety decision with transparent proposal-time scores."""

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


def score_binary_correction(
    proposal: dict[str, Any],
) -> BinaryEccoGatePrediction:
    """Decide whether a proposed correction is safe to apply.

    The gate uses no reference transcript, post-hoc error score, or LLM.
    """

    raw_text = _normalized_text(proposal.get("raw_asr_text"))
    corrected_text = _normalized_text(
        proposal.get("proposed_corrected_text")
    )
    if not raw_text or not corrected_text:
        return BinaryEccoGatePrediction(
            decision="do_not_apply",
            support_score=0.0,
            risk_score=1.0,
            explanation="do_not_apply: empty text cannot be audited safely.",
        )
    if raw_text == corrected_text:
        return BinaryEccoGatePrediction(
            decision="do_not_apply",
            support_score=0.9,
            risk_score=0.05,
            explanation=(
                "do_not_apply: the proposal makes no measurable text change."
            ),
        )

    base = score_correction_proposal(proposal)
    severe_temporal_risk = (
        _as_bool(proposal.get("heavy_overlap_flag"))
        or _as_bool(proposal.get("speaker_ambiguity_flag"))
    )
    partial = _as_bool(proposal.get("partial_utterance_flag"))
    safe = (
        base.decision == "accept"
        and base.support_score >= 0.58
        and base.risk_score < 0.4
        and not severe_temporal_risk
    )
    if partial and base.risk_score >= 0.35:
        safe = False
    decision = "safe_to_apply" if safe else "do_not_apply"
    return BinaryEccoGatePrediction(
        decision=decision,
        support_score=base.support_score,
        risk_score=base.risk_score,
        explanation=(
            f"{decision}: binary mapping of proposal-time evidence. "
            f"{base.explanation}"
        ),
    )
