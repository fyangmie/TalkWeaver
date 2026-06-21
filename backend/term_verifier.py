"""Verification layer for retrieval-based error-word rescue."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

from backend.llm_config import LLMConfig, PROMPT_VERSION
from backend.llm_correction import request_json_completion
from backend.term_rescue import TermMatch


VerifierDecision = Literal["accept", "needs_review", "reject", "no_op"]
VerifierMode = Literal["rule", "llm", "llm_with_rule_fallback"]


@dataclass(frozen=True)
class TermVerification:
    """Decision for one retrieved term candidate."""

    decision: VerifierDecision
    reason: str
    backend: str = "rule"
    api_used: bool = False
    fallback_used: bool = False
    prompt_version: str = PROMPT_VERSION

    @property
    def accepted(self) -> bool:
        return self.decision == "accept"


AMBIGUOUS_CANONICALS = {"RAG", "WER", "DER", "TagSpeech"}
NEGATIVE_CUES = {
    "RAG": (
        "rack on the table",
        "table",
        "microphone",
        "mic",
        "shelf",
        "server rack",
        "metal rack",
        "equipment rack",
        "physical",
    ),
    "WER": (
        "where is",
        "where should",
        "where are",
        "location",
        "place",
        "microphone",
    ),
    "DER": (
        "dear team",
        "dear friend",
        "greeting",
        "letter",
    ),
}
POSITIVE_CUES = {
    "RAG": (
        "retrieval",
        "augmented",
        "generation",
        "benchmark",
        "asr",
        "wer",
        "system",
    ),
    "WER": (
        "word error",
        "error rate",
        "benchmark",
        "asr",
        "metric",
    ),
    "DER": (
        "diarization",
        "speaker",
        "error rate",
        "metric",
    ),
}


def _combined_text(raw_text: str, context: str) -> str:
    return f"{raw_text} {context}".casefold()


def rule_verify_term_match(
    match: TermMatch,
    *,
    raw_text: str,
    context: str = "",
    terms_source: str = "external_or_predefined",
) -> TermVerification:
    """Deterministically decide whether a retrieved term can be applied."""

    canonical = match.canonical
    combined = _combined_text(raw_text, context)
    if any(cue.casefold() in combined for cue in NEGATIVE_CUES.get(canonical, ())):
        return TermVerification(
            decision="reject",
            reason=(
                f"Negative context indicates common-word meaning for {canonical}."
            ),
        )
    if not match.safe_to_apply:
        return TermVerification(
            decision="needs_review",
            reason=match.context_reason or "Candidate lacks enough context.",
        )
    if canonical in AMBIGUOUS_CANONICALS and not any(
        cue.casefold() in combined for cue in POSITIVE_CUES.get(canonical, ())
    ):
        return TermVerification(
            decision="needs_review",
            reason=(
                f"{canonical} is ambiguous and lacks domain context."
            ),
        )
    if terms_source == "oracle_diagnostic":
        return TermVerification(
            decision="accept",
            reason="Accepted as oracle/custom-vocabulary diagnostic candidate.",
        )
    return TermVerification(
        decision="accept",
        reason="Candidate is supported by glossary match and context checks.",
    )


def build_verifier_messages(
    match: TermMatch,
    *,
    raw_text: str,
    context: str,
    terms_source: str,
    language: str = "",
) -> list[dict[str, str]]:
    """Build a constrained verifier prompt for an OpenAI-compatible LLM."""

    system = (
        "You are TalkWeaver's term-rescue verifier. Decide whether one "
        "retrieved candidate may replace a suspected ASR error. Do not rewrite "
        "the whole sentence. Return only JSON with keys decision and reason. "
        "decision must be accept, needs_review, reject, or no_op."
    )
    user = {
        "raw_asr_text": raw_text,
        "language": language or "unknown",
        "context": context,
        "matched_form": match.matched_form,
        "candidate_replacement": match.canonical,
        "retrieval_method": match.retrieval_method,
        "retrieval_score": match.score,
        "terms_source": terms_source,
        "rule_context_reason": match.context_reason,
        "policy": (
            "Accept only when the local context supports the candidate. "
            "Reject common-word meanings such as physical rack -> RAG, "
            "where-location -> WER, or dear-greeting -> DER. Use no_op when "
            "the source is already an acceptable alias, style variant, or "
            "equivalent wording. Use needs_review when plausible but uncertain. "
            "For oracle_diagnostic, the candidate "
            "comes from a curated observed-ASR-failure list; accept plausible "
            "phonetic, named-entity, acronym, or domain-word repairs unless "
            "there is clear negative context. Oracle acceptance is diagnostic "
            "only, not generalization evidence."
        ),
        "language_policy": (
            "Evaluate the candidate in the specified language. For zh-CN, "
            "consider Mandarin homophones, Simplified/Traditional variants, "
            "and Chinese word segmentation errors. For fr, consider accents "
            "and French phonetic confusions. For en, consider names, acronyms, "
            "and domain terms."
        ),
    }
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": json.dumps(user, ensure_ascii=False, sort_keys=True),
        },
    ]


def _parse_llm_decision(payload: dict[str, object]) -> TermVerification:
    raw_decision = str(payload.get("decision", "")).strip().lower()
    decision: VerifierDecision
    if raw_decision in {"accept", "needs_review", "reject", "no_op"}:
        decision = raw_decision  # type: ignore[assignment]
    else:
        raise ValueError("Verifier LLM returned an invalid decision.")
    reason = str(payload.get("reason", "")).strip()
    if not reason:
        raise ValueError("Verifier LLM returned an empty reason.")
    return TermVerification(
        decision=decision,
        reason=reason,
        backend="llm",
        api_used=True,
    )


def verify_term_match(
    match: TermMatch,
    *,
    raw_text: str,
    context: str = "",
    language: str = "",
    terms_source: str = "external_or_predefined",
    verifier: VerifierMode = "rule",
    llm_config: LLMConfig | None = None,
) -> TermVerification:
    """Verify one term candidate using rules or an optional LLM verifier."""

    if verifier == "rule":
        return rule_verify_term_match(
            match,
            raw_text=raw_text,
            context=context,
            terms_source=terms_source,
        )
    if llm_config is None or not llm_config.is_configured:
        if verifier == "llm":
            raise RuntimeError("LLM verifier requested but LLM is not configured.")
        rule_result = rule_verify_term_match(
            match,
            raw_text=raw_text,
            context=context,
            terms_source=terms_source,
        )
        return TermVerification(
            decision=rule_result.decision,
            reason=f"LLM unavailable; rule fallback used. {rule_result.reason}",
            backend="rule_fallback",
            api_used=False,
            fallback_used=True,
            prompt_version=rule_result.prompt_version,
        )
    try:
        payload = request_json_completion(
            llm_config,
            build_verifier_messages(
                match,
                raw_text=raw_text,
                context=context,
                language=language,
                terms_source=terms_source,
            ),
        )
        return _parse_llm_decision(payload)
    except (RuntimeError, ValueError, json.JSONDecodeError) as exc:
        if verifier == "llm":
            raise RuntimeError(f"LLM verifier failed: {exc}") from exc
        rule_result = rule_verify_term_match(
            match,
            raw_text=raw_text,
            context=context,
            terms_source=terms_source,
        )
        return TermVerification(
            decision=rule_result.decision,
            reason=f"LLM verifier failed; rule fallback used. {rule_result.reason}",
            backend="rule_fallback",
            api_used=False,
            fallback_used=True,
            prompt_version=rule_result.prompt_version,
        )
