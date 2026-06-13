"""Evidence-grounded correction and unsupported-change auditing."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

from backend.llm_config import LLMConfig, PROMPT_VERSION
from backend.llm_correction import correct_segments, rule_based_correction
from backend.schemas import (
    ConversationEvent,
    CorrectionAudit,
    TemporalAnchor,
    TermRescueCandidate,
)
from backend.term_rescue import candidates_for_anchor


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9\u4e00-\u9fff]+(?:[.-][A-Za-z0-9]+)*")


def _tokens(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


def build_structured_correction_prompt(
    anchors: list[TemporalAnchor],
    term_candidates: list[TermRescueCandidate],
    events: list[ConversationEvent],
) -> str:
    """Build an auditable DiarizationLM-style proxy prompt."""

    event_by_anchor: dict[str, list[str]] = {}
    for event in events:
        for anchor_id in event.evidence_anchor_ids:
            event_by_anchor.setdefault(anchor_id, []).append(event.type)
    lines = [
        "TalkWeaver evidence-grounded correction task.",
        "Preserve anchor IDs, timestamps, and speaker labels.",
        "Modify corrected_text only. Add no unsupported claims.",
    ]
    for anchor in anchors:
        candidates = candidates_for_anchor(anchor.anchor_id, term_candidates)
        terms = ", ".join(candidate.canonical for candidate in candidates)
        event_types = ", ".join(event_by_anchor.get(anchor.anchor_id, []))
        lines.extend(
            [
                "",
                (
                    f"[{anchor.anchor_id}] {anchor.start:.2f}-{anchor.end:.2f} "
                    f"{anchor.speaker} | speakers={','.join(anchor.speakers) or 'none'} "
                    f"| overlap={str(anchor.overlap).lower()} "
                    f"| interruption={str(anchor.interruption).lower()} "
                    f"| confidence={anchor.confidence:.2f}"
                ),
                f"Raw: {anchor.raw_text}",
                f"Retrieved terms: {terms or 'none'}",
                f"Events: {event_types or 'none'}",
                (
                    "Constraint: conservative correction zone; retain ambiguity."
                    if anchor.overlap
                    else "Constraint: only evidence-supported lexical corrections."
                ),
            ]
        )
    return "\n".join(lines)


def audit_correction(
    anchor: TemporalAnchor,
    corrected_text: str,
    *,
    candidates: list[TermRescueCandidate],
    neighboring_text: str = "",
) -> CorrectionAudit:
    """Audit insertions against raw, retrieval, and neighboring evidence."""

    raw_tokens = _tokens(anchor.raw_text)
    corrected_tokens = _tokens(corrected_text)
    matcher = SequenceMatcher(None, raw_tokens, corrected_tokens)
    changed_tokens: list[str] = []
    inserted_tokens: list[str] = []
    for operation, raw_start, raw_end, corrected_start, corrected_end in matcher.get_opcodes():
        if operation == "equal":
            continue
        removed = raw_tokens[raw_start:raw_end]
        inserted = corrected_tokens[corrected_start:corrected_end]
        changed_tokens.append(
            f"{operation}: {' '.join(removed) or '∅'} -> "
            f"{' '.join(inserted) or '∅'}"
        )
        inserted_tokens.extend(inserted)

    evidence_tokens = set(raw_tokens) | set(_tokens(neighboring_text))
    evidence: list[dict[str, Any]] = [
        {
            "type": "raw_text",
            "anchor_id": anchor.anchor_id,
            "text": anchor.raw_text,
        }
    ]
    for candidate in candidates:
        evidence_tokens.update(_tokens(candidate.canonical))
        evidence_tokens.update(
            token
            for form in candidate.spoken_forms + candidate.asr_error_forms
            for token in _tokens(form)
        )
        evidence.append(
            {
                "type": "term_candidate",
                "term_id": candidate.term_id,
                "canonical": candidate.canonical,
                "score": candidate.retrieved_score,
                "method": candidate.retrieval_method,
            }
        )

    unsupported = sorted(
        {token for token in inserted_tokens if token not in evidence_tokens}
    )
    supported = sorted(
        {token for token in inserted_tokens if token in evidence_tokens}
    )
    raw_length = max(1, len(raw_tokens))
    length_ratio = len(corrected_tokens) / raw_length
    edit_ratio = 1.0 - matcher.ratio()
    large_overlap_change = anchor.overlap and edit_ratio > 0.35
    much_longer = length_ratio > 1.5 and len(corrected_tokens) > len(raw_tokens) + 2
    needs_review = bool(
        unsupported or large_overlap_change or much_longer
    )
    if unsupported or much_longer:
        risk = "high"
    elif large_overlap_change or edit_ratio > 0.4:
        risk = "medium"
    else:
        risk = "low"
    return CorrectionAudit(
        anchor_id=anchor.anchor_id,
        raw_text=anchor.raw_text,
        corrected_text=corrected_text,
        changed_tokens=changed_tokens,
        supported_changes=supported,
        unsupported_changes=unsupported,
        hallucination_risk=risk,
        needs_review=needs_review,
        evidence=evidence,
    )


def apply_constrained_correction(
    anchors: list[TemporalAnchor],
    term_candidates: list[TermRescueCandidate],
    events: list[ConversationEvent],
    *,
    llm_config: dict[str, Any] | None = None,
) -> tuple[list[TemporalAnchor], list[CorrectionAudit], str]:
    """Correct anchors independently and retain an explicit audit trail."""

    config = dict(llm_config or {})
    use_api = bool(config.get("use_api", False))
    correction_mode = str(
        config.get(
            "correction_mode",
            "llm_with_rule_fallback" if use_api else "rule_fallback",
        )
    )
    runtime_config = config.get("runtime_config")
    if runtime_config is not None and not isinstance(
        runtime_config,
        LLMConfig,
    ):
        raise TypeError("runtime_config must be an LLMConfig object.")
    segment_payloads = []
    for anchor in anchors:
        segment_payloads.append(
            {
                **anchor.to_dict(),
                "retrieved_terms": list(anchor.retrieved_terms),
            }
        )
    corrected_segments = correct_segments(
        segment_payloads,
        mock=False,
        provider=str(config.get("provider", "auto")),
        openai_api_key=(
            str(config.get("openai_api_key", "")) if use_api else ""
        ),
        deepseek_api_key=(
            str(config.get("deepseek_api_key", "")) if use_api else ""
        ),
        qwen_api_key=(
            str(config.get("qwen_api_key", "")) if use_api else ""
        ),
        openai_model=str(config.get("openai_model", "gpt-4.1-mini")),
        deepseek_model=str(config.get("deepseek_model", "deepseek-v4-pro")),
        qwen_model=str(config.get("qwen_model", "qwen-plus")),
        openai_base_url=str(
            config.get("openai_base_url", "https://api.openai.com/v1")
        ),
        deepseek_base_url=str(
            config.get("deepseek_base_url", "https://api.deepseek.com")
        ),
        qwen_base_url=str(
            config.get(
                "qwen_base_url",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
        ),
        correction_mode=correction_mode,
        llm_config=runtime_config,
        temperature=float(config.get("temperature", 0.0)),
        timeout_seconds=float(config.get("timeout_seconds", 30.0)),
        prompt_version=str(
            config.get("prompt_version", PROMPT_VERSION)
        ),
    )
    audits: list[CorrectionAudit] = []
    for index, (anchor, corrected) in enumerate(
        zip(anchors, corrected_segments)
    ):
        candidate_list = candidates_for_anchor(
            anchor.anchor_id, term_candidates
        )
        corrected_text = str(
            corrected.get("corrected_text") or anchor.raw_text
        ).strip()
        neighboring_text = " ".join(
            nearby.raw_text
            for nearby in anchors[max(0, index - 1) : index + 2]
            if nearby.anchor_id != anchor.anchor_id
        )
        audit = audit_correction(
            anchor,
            corrected_text,
            candidates=candidate_list,
            neighboring_text=neighboring_text,
        )
        audit.correction_mode = str(
            corrected.get("correction_mode", correction_mode)
        )
        audit.llm_provider = str(corrected.get("llm_provider", ""))
        audit.llm_model = str(corrected.get("llm_model", ""))
        audit.prompt_version = str(
            corrected.get("prompt_version", PROMPT_VERSION)
        )
        audit.temperature = float(
            corrected.get("llm_temperature", 0.0)
        )
        audit.api_used = bool(corrected.get("api_used", False))
        audit.fallback_used = bool(
            corrected.get("fallback_used", False)
        )
        audit.evidence.append(
            {
                "type": "correction_execution",
                "correction_mode": audit.correction_mode,
                "provider": audit.llm_provider,
                "model": audit.llm_model,
                "prompt_version": audit.prompt_version,
                "temperature": audit.temperature,
                "api_used": audit.api_used,
                "fallback_used": audit.fallback_used,
            }
        )
        anchor.corrected_text = (
            anchor.raw_text if audit.unsupported_changes else corrected_text
        )
        if audit.unsupported_changes:
            anchor.notes.append(
                "Unsupported correction was rejected; raw text retained."
            )
        anchor.correction_evidence = audit.evidence
        anchor.unsupported_changes = audit.unsupported_changes
        anchor.needs_review = anchor.needs_review or audit.needs_review
        audits.append(audit)

    modes = {
        str(segment.get("correction_mode", "unknown"))
        for segment in corrected_segments
    }
    correction_mode = ",".join(sorted(modes)) if modes else "no_segments"
    return anchors, audits, correction_mode
