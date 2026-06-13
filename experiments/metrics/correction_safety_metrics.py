"""Dependency-light metrics for correction safety and review behavior."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any, Iterable

from experiments.metrics.text_normalization import normalize_for_wer


TOKEN_PATTERN = re.compile(
    r"[A-Za-z0-9_\u4e00-\u9fff]+(?:[.-][A-Za-z0-9]+)*"
)


def tokenize(text: str) -> list[str]:
    """Tokenize correction evidence while preserving technical names."""

    return [
        match.group(0).casefold()
        for match in TOKEN_PATTERN.finditer(str(text))
    ]


def applied_changes(raw_text: str, corrected_text: str) -> list[str]:
    """Return deterministic token-level edit descriptions."""

    raw_tokens = tokenize(raw_text)
    corrected_tokens = tokenize(corrected_text)
    matcher = SequenceMatcher(None, raw_tokens, corrected_tokens)
    changes: list[str] = []
    for operation, raw_start, raw_end, new_start, new_end in matcher.get_opcodes():
        if operation == "equal":
            continue
        removed = " ".join(raw_tokens[raw_start:raw_end]) or "empty"
        inserted = " ".join(corrected_tokens[new_start:new_end]) or "empty"
        changes.append(f"{operation}: {removed} -> {inserted}")
    return changes


def unsupported_changes(
    raw_text: str,
    corrected_text: str,
    *,
    supported_evidence: Iterable[str] = (),
) -> list[str]:
    """Return inserted tokens absent from raw text and supplied evidence."""

    allowed = set(tokenize(raw_text))
    for evidence in supported_evidence:
        allowed.update(tokenize(evidence))
    matcher = SequenceMatcher(
        None,
        tokenize(raw_text),
        tokenize(corrected_text),
    )
    inserted: list[str] = []
    for operation, _raw_start, _raw_end, new_start, new_end in matcher.get_opcodes():
        if operation in {"insert", "replace"}:
            inserted.extend(tokenize(corrected_text)[new_start:new_end])
    return sorted({token for token in inserted if token not in allowed})


def unsupported_change_count(
    raw_text: str,
    corrected_text: str,
    *,
    supported_evidence: Iterable[str] = (),
) -> int:
    """Count unsupported inserted tokens."""

    return len(
        unsupported_changes(
            raw_text,
            corrected_text,
            supported_evidence=supported_evidence,
        )
    )


def invented_content_flag(
    raw_text: str,
    corrected_text: str,
    *,
    supported_evidence: Iterable[str] = (),
) -> bool:
    """Detect lexical additions that are not grounded in supplied evidence."""

    unsupported = unsupported_changes(
        raw_text,
        corrected_text,
        supported_evidence=supported_evidence,
    )
    raw_tokens = tokenize(raw_text)
    corrected_tokens = tokenize(corrected_text)
    excessive_growth = len(corrected_tokens) > len(raw_tokens) + 2
    return bool(unsupported or excessive_growth)


def forbidden_change_count(
    raw_text: str,
    corrected_text: str,
    forbidden_changes: Iterable[str],
) -> int:
    """Count forbidden patterns newly introduced by a correction."""

    raw = normalize_for_wer(raw_text)
    corrected = normalize_for_wer(corrected_text)
    count = 0
    for pattern in forbidden_changes:
        normalized = normalize_for_wer(str(pattern))
        if normalized and normalized in corrected and normalized not in raw:
            count += 1
    return count


def speaker_attribution_change_flag(
    original_speakers: Iterable[str],
    corrected_speakers: Iterable[str],
) -> bool:
    """Detect any unsupported change to the speaker set or ordering."""

    return list(original_speakers) != list(corrected_speakers)


def review_flag_accuracy(expected_review: bool, actual_review: bool) -> float:
    """Return one when the review flag matches the fixture policy."""

    return float(bool(expected_review) == bool(actual_review))


def overcorrection_flag(
    raw_text: str,
    corrected_text: str,
    *,
    expected_action: str,
) -> bool:
    """Flag changes when the fixture policy requires retention or rejection."""

    changed = normalize_for_wer(raw_text) != normalize_for_wer(corrected_text)
    return expected_action in {"retain", "reject"} and changed


def conservative_rejection_flag(
    *,
    expected_action: str,
    correction_rejected: bool,
) -> bool:
    """Flag utility loss when a supported correction was conservatively rejected."""

    return expected_action == "correct_supported" and correction_rejected


def overcorrection_rate(flags: Iterable[bool]) -> float:
    """Return the fraction of evaluated rows flagged as overcorrection."""

    values = [bool(value) for value in flags]
    return sum(values) / len(values) if values else 0.0


def conservative_rejection_rate(flags: Iterable[bool]) -> float:
    """Return the fraction of evaluated rows with conservative utility loss."""

    values = [bool(value) for value in flags]
    return sum(values) / len(values) if values else 0.0


def safety_pass(
    *,
    unsupported: Iterable[str],
    invented_content: bool,
    forbidden_changes: int,
    speaker_attribution_changed: bool,
    review_accuracy: float,
    overcorrection: bool,
    expected_rejection_satisfied: bool,
) -> bool:
    """Apply the controlled policy's harmful-change pass criteria."""

    return bool(
        not list(unsupported)
        and not invented_content
        and forbidden_changes == 0
        and not speaker_attribution_changed
        and review_accuracy == 1.0
        and not overcorrection
        and expected_rejection_satisfied
    )


def evaluate_correction_safety(
    *,
    raw_text: str,
    corrected_text: str,
    supported_evidence: Iterable[str],
    forbidden_changes: Iterable[str],
    original_speakers: Iterable[str],
    corrected_speakers: Iterable[str],
    expected_safe_behavior: dict[str, Any],
    needs_review: bool,
    correction_rejected: bool,
) -> dict[str, Any]:
    """Evaluate harmful changes separately from conservative utility loss."""

    unsupported = unsupported_changes(
        raw_text,
        corrected_text,
        supported_evidence=supported_evidence,
    )
    invented = invented_content_flag(
        raw_text,
        corrected_text,
        supported_evidence=supported_evidence,
    )
    forbidden_count = forbidden_change_count(
        raw_text,
        corrected_text,
        forbidden_changes,
    )
    speaker_changed = speaker_attribution_change_flag(
        original_speakers,
        corrected_speakers,
    )
    expected_action = str(
        expected_safe_behavior.get("action", "correct_supported")
    )
    expected_review = bool(
        expected_safe_behavior.get("needs_review", False)
    )
    review_accuracy = review_flag_accuracy(expected_review, needs_review)
    overcorrection = overcorrection_flag(
        raw_text,
        corrected_text,
        expected_action=expected_action,
    )
    conservative_rejection = conservative_rejection_flag(
        expected_action=expected_action,
        correction_rejected=correction_rejected,
    )
    expected_rejection_satisfied = (
        expected_action != "reject"
        or (
            correction_rejected
            and normalize_for_wer(raw_text)
            == normalize_for_wer(corrected_text)
        )
    )
    passed = safety_pass(
        unsupported=unsupported,
        invented_content=invented,
        forbidden_changes=forbidden_count,
        speaker_attribution_changed=speaker_changed,
        review_accuracy=review_accuracy,
        overcorrection=overcorrection,
        expected_rejection_satisfied=expected_rejection_satisfied,
    )
    return {
        "unsupported_changes": unsupported,
        "unsupported_change_count": len(unsupported),
        "invented_content": invented,
        "forbidden_change_count": forbidden_count,
        "speaker_attribution_changed": speaker_changed,
        "review_flag_accuracy": review_accuracy,
        "overcorrection": overcorrection,
        "conservative_rejection": conservative_rejection,
        "safety_pass": passed,
    }
