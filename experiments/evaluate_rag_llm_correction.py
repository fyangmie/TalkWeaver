#!/usr/bin/env python3
"""Evaluate evidence-constrained RAG + LLM term correction on ASR rows."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.llm_config import LLMConfig, load_llm_config  # noqa: E402
from backend.llm_correction import request_json_completion  # noqa: E402
from backend.term_rescue import GlossaryEntry, load_reference_glossary  # noqa: E402
from experiments.evaluate_term_rescue_real_audit import (  # noqa: E402
    _term_metrics,
    _terms_present,
)
from experiments.metrics.text_metrics import evaluate_text  # noqa: E402
from experiments.metrics.text_normalization import normalize_for_wer  # noqa: E402


OUTPUT_COLUMNS = [
    "clip_id",
    "dataset_name",
    "language",
    "model_name",
    "wer_before",
    "wer_after",
    "wer_delta",
    "reference_terms",
    "baseline_terms",
    "corrected_terms",
    "missing_before",
    "missing_after",
    "term_recall_before",
    "term_recall_after",
    "term_f1_before",
    "term_f1_after",
    "api_used",
    "llm_provider",
    "llm_model",
    "candidate_corrections",
    "applied_corrections",
    "rejected_corrections",
    "needs_review_corrections",
    "no_op_corrections",
    "reference_text",
    "hypothesis_text",
    "corrected_text",
    "claim_scope",
    "notes",
]
SUMMARY_COLUMNS = [
    "dataset_name",
    "language",
    "model_name",
    "num_rows",
    "mean_wer_before",
    "mean_wer_after",
    "mean_wer_delta",
    "mean_term_recall_before",
    "mean_term_recall_after",
    "mean_term_f1_before",
    "mean_term_f1_after",
    "applied_correction_count",
    "rejected_correction_count",
    "needs_review_count",
    "no_op_count",
    "api_used",
    "llm_provider",
    "llm_model",
    "claim_scope",
]


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[.'-][A-Za-z0-9]+)*")
NUMERIC_UNIT_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(?P<number>\d+(?:[.,]\d+)?)\s+"
    r"(?P<unit>cents?|seems|sense|sent)\s+"
    r"(?P<link>to|a|per)\s+share"
    r"(?![A-Za-z0-9])",
    flags=re.IGNORECASE,
)
NUMERIC_UNIT_CANONICALS = {"cents a share", "cents per share"}
NUMERIC_UNIT_CONTEXT_CUES = {
    "dividend",
    "reinstatement",
    "share",
    "shares",
}
ENTITY_CATEGORIES = {
    "company_name",
    "product_name",
    "business_segment",
    "website",
    "currency",
}
COMMON_ENTITY_SOURCE_FORMS = {
    "u s",
    "us",
    "u.s",
    "u.s.",
    "e u",
    "eu",
    "pc",
    "mobile",
    "it",
    "ai",
}


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _tokens(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


def _numbers(text: str) -> set[str]:
    return {token for token in _tokens(text) if token.isdigit()}


def _normalized_numbers(text: str) -> set[str]:
    return {token.replace(",", "") for token in _numbers(text)}


def _contains_case_insensitive(text: str, phrase: str) -> bool:
    pattern = rf"(?<![A-Za-z0-9]){re.escape(phrase)}(?![A-Za-z0-9])"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def _replace_once(text: str, source: str, replacement: str) -> str:
    pattern = rf"(?<![A-Za-z0-9]){re.escape(source)}(?![A-Za-z0-9])"
    return re.sub(
        pattern,
        replacement,
        text,
        count=1,
        flags=re.IGNORECASE,
    )


def _has_numeric_unit_context(text: str) -> bool:
    normalized = normalize_for_wer(text)
    return any(cue in normalized for cue in NUMERIC_UNIT_CONTEXT_CUES)


def _has_allowed_context(entry: GlossaryEntry, raw_text: str) -> bool:
    normalized = normalize_for_wer(raw_text)
    return any(
        context and normalize_for_wer(context) in normalized
        for context in entry.allowed_contexts
    )


def _is_common_entity_source(source_text: str) -> bool:
    source_norm = normalize_for_wer(source_text)
    compact = source_norm.replace(" ", "")
    common_compact = {
        form.replace(" ", "").replace(".", "")
        for form in COMMON_ENTITY_SOURCE_FORMS
    }
    return (
        source_norm in COMMON_ENTITY_SOURCE_FORMS
        or compact.replace(".", "") in common_compact
    )


def _numeric_unit_no_op(
    *,
    entry: GlossaryEntry,
    source_text: str,
) -> tuple[bool, str]:
    canonical_norm = normalize_for_wer(entry.canonical)
    if canonical_norm not in NUMERIC_UNIT_CANONICALS:
        return False, ""
    match = NUMERIC_UNIT_PATTERN.search(source_text)
    if match is None:
        return False, ""
    unit = match.group("unit").lower()
    link = match.group("link").lower()
    if unit in {"cent", "cents"} and link in {"a", "per"}:
        return (
            True,
            "source_text is already an acceptable numeric-unit expression",
        )
    return False, ""


def _candidate_corrections(
    raw_text: str,
    entries: list[GlossaryEntry],
) -> list[dict[str, str]]:
    """Return glossary-grounded candidate corrections without reference text."""

    candidates: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    entries_by_key = _entry_by_key(entries)
    cents_entry = entries_by_key.get("cents a share")
    if cents_entry is not None and _has_numeric_unit_context(raw_text):
        for match in NUMERIC_UNIT_PATTERN.finditer(raw_text):
            source = match.group(0)
            number = match.group("number")
            replacement = f"{number} cents a share"
            key = (source.casefold(), replacement.casefold(), cents_entry.canonical)
            if key not in seen and normalize_for_wer(source) != normalize_for_wer(replacement):
                seen.add(key)
                candidates.append(
                    {
                        "source_text": source,
                        "replacement_text": replacement,
                        "canonical_term": cents_entry.canonical,
                        "reason": (
                            "numeric-unit pattern near dividend/share context"
                        ),
                    }
                )

    for entry in entries:
        if normalize_for_wer(entry.canonical) in NUMERIC_UNIT_CANONICALS:
            continue
        forms = [
            *entry.asr_error_forms,
            *entry.aliases,
        ]
        for form in forms:
            if not form or not _contains_case_insensitive(raw_text, form):
                continue
            if _already_acceptable_form(entry, form):
                continue
            replacement = entry.canonical
            key = (form.casefold(), replacement.casefold(), entry.canonical)
            if key in seen or normalize_for_wer(form) == normalize_for_wer(replacement):
                continue
            seen.add(key)
            candidates.append(
                {
                    "source_text": form,
                    "replacement_text": replacement,
                    "canonical_term": entry.canonical,
                    "reason": "predefined glossary error form or alias",
                }
            )
    return candidates


def _entry_payload(entries: list[GlossaryEntry]) -> list[dict[str, Any]]:
    return [
        {
            "canonical": entry.canonical,
            "aliases": list(entry.aliases),
            "spoken_forms": list(entry.spoken_forms),
            "asr_error_forms": list(entry.asr_error_forms),
            "category": entry.category,
            "allowed_contexts": list(entry.allowed_contexts),
        }
        for entry in entries
    ]


def _entry_by_key(entries: list[GlossaryEntry]) -> dict[str, GlossaryEntry]:
    return {normalize_for_wer(entry.canonical): entry for entry in entries}


def _allowed_replacement_tokens(
    entry: GlossaryEntry,
    *,
    source_text: str,
) -> set[str]:
    allowed = set(_tokens(source_text))
    allowed.update(_numbers(source_text))
    for form in [
        entry.canonical,
        *entry.aliases,
        *entry.spoken_forms,
    ]:
        allowed.update(_tokens(form))
    return allowed


def _normalized_forms(values: list[str] | tuple[str, ...]) -> set[str]:
    return {normalize_for_wer(value) for value in values if normalize_for_wer(value)}


def _already_acceptable_form(entry: GlossaryEntry, source_text: str) -> bool:
    """Return whether the source is already a non-error glossary form."""

    source_norm = normalize_for_wer(source_text)
    accepted = _normalized_forms(
        [entry.canonical, *entry.aliases, *entry.spoken_forms]
    )
    error_forms = _normalized_forms(entry.asr_error_forms)
    return source_norm in accepted and source_norm not in error_forms


def _validate_numeric_unit_correction(
    *,
    entry: GlossaryEntry,
    source_text: str,
    replacement_text: str,
    raw_text: str,
) -> tuple[bool, str]:
    canonical_norm = normalize_for_wer(entry.canonical)
    if canonical_norm not in NUMERIC_UNIT_CANONICALS:
        return True, "not a numeric-unit correction"
    source_numbers = _normalized_numbers(source_text)
    replacement_numbers = _normalized_numbers(replacement_text)
    if not source_numbers:
        return False, "numeric-unit correction requires an explicit source number"
    if source_numbers != replacement_numbers:
        return False, "numeric-unit correction must preserve source numbers"
    if not _has_numeric_unit_context(raw_text):
        return False, "numeric-unit correction lacks dividend/share context"
    return True, "numeric-unit correction passed context and number checks"


def _validate_correction(
    correction: dict[str, Any],
    *,
    raw_text: str,
    entries_by_key: dict[str, GlossaryEntry],
    gate_version: str = "v2",
) -> tuple[bool, str]:
    decision, reason = _gate_correction(
        correction,
        raw_text=raw_text,
        entries_by_key=entries_by_key,
        gate_version=gate_version,
    )
    return decision == "accept", reason


def _gate_correction(
    correction: dict[str, Any],
    *,
    raw_text: str,
    entries_by_key: dict[str, GlossaryEntry],
    gate_version: str = "v2",
) -> tuple[str, str]:
    """Return the evidence-gate decision for one proposed correction."""

    if gate_version not in {"v2", "v3"}:
        return "reject", f"unsupported gate_version={gate_version}"
    source = str(correction.get("source_text", "")).strip()
    replacement = str(correction.get("replacement_text", "")).strip()
    canonical = str(correction.get("canonical_term", "")).strip()
    if not source or not replacement or not canonical:
        return "reject", "missing source_text, replacement_text, or canonical_term"
    if not _contains_case_insensitive(raw_text, source):
        return "reject", "source_text was not found in raw ASR text"
    entry = entries_by_key.get(normalize_for_wer(canonical))
    if entry is None:
        return "reject", "canonical_term is not in the predefined glossary"
    if normalize_for_wer(source) == normalize_for_wer(replacement):
        return "no_op", "no effective text change"
    numeric_no_op, numeric_reason = _numeric_unit_no_op(
        entry=entry,
        source_text=source,
    )
    if numeric_no_op:
        return "no_op", numeric_reason
    if (
        _already_acceptable_form(entry, source)
        and normalize_for_wer(source) != normalize_for_wer(replacement)
    ):
        return (
            "no_op",
            "source_text is already an accepted glossary form; "
            "canonicalization-only rewrites are treated as no-op",
        )
    source_norm = normalize_for_wer(source)
    error_forms = _normalized_forms(entry.asr_error_forms)
    if gate_version == "v3":
        numeric_unit = normalize_for_wer(entry.canonical) in NUMERIC_UNIT_CANONICALS
        numeric_source = NUMERIC_UNIT_PATTERN.search(source) is not None
        if not numeric_unit and source_norm not in error_forms:
            return (
                "needs_review",
                "v3 requires source_text to match a predefined ASR error form",
            )
        if numeric_unit and not numeric_source and source_norm not in error_forms:
            return (
                "needs_review",
                "v3 numeric-unit correction requires a numeric unit error form",
            )
        if entry.allowed_contexts and not _has_allowed_context(entry, raw_text):
            return (
                "needs_review",
                "v3 requires explicit allowed-context evidence",
            )
    if entry.category in ENTITY_CATEGORIES and _is_common_entity_source(source):
        return (
            "reject",
            "ambiguous common source text cannot be rewritten as an entity",
        )
    if (
        entry.category in ENTITY_CATEGORIES
        and entry.allowed_contexts
        and not _has_allowed_context(entry, raw_text)
    ):
        return (
            "needs_review",
            "entity correction lacks required glossary context",
        )
    replacement_norm = normalize_for_wer(replacement)
    canonical_norm = normalize_for_wer(entry.canonical)
    alias_norms = {
        normalize_for_wer(value)
        for value in [entry.canonical, *entry.aliases, *entry.spoken_forms]
    }
    if not any(value and value in replacement_norm for value in alias_norms):
        return "reject", "replacement_text does not contain the glossary term"
    ok, reason = _validate_numeric_unit_correction(
        entry=entry,
        source_text=source,
        replacement_text=replacement,
        raw_text=raw_text,
    )
    if not ok:
        return "reject", reason
    introduced = set(_tokens(replacement)) - set(_tokens(source))
    allowed = _allowed_replacement_tokens(entry, source_text=source)
    unsupported = sorted(token for token in introduced if token not in allowed)
    if unsupported:
        return "reject", "unsupported replacement tokens: " + ", ".join(unsupported)
    if len(_tokens(replacement)) > len(_tokens(source)) + 3:
        return "reject", "replacement_text is too expansive"
    if canonical_norm not in replacement_norm and replacement_norm not in canonical_norm:
        return "reject", "replacement is not a short canonical-term correction"
    return "accept", f"accepted by {gate_version} evidence gate"


def _build_messages(
    *,
    row: dict[str, str],
    entries: list[GlossaryEntry],
    candidates: list[dict[str, str]],
    gate_version: str = "v2",
) -> list[dict[str, str]]:
    payload = {
        "task": "Propose evidence-constrained RAG term corrections only.",
        "gate_version": gate_version,
        "raw_asr_text": row.get("hypothesis_text", ""),
        "language": row.get("language", "en"),
        "dataset_name": row.get("dataset_name", ""),
        "glossary": _entry_payload(entries),
        "retrieved_candidate_corrections": candidates,
        "rules": [
            "Return JSON only.",
            "Do not use or infer the reference transcript.",
            "Prefer corrections from retrieved_candidate_corrections; only add a new correction when it is directly supported by the glossary and raw ASR context.",
            "Only correct short ASR spans that are supported by the glossary and local context.",
            "source_text must appear exactly in raw_asr_text.",
            "replacement_text must be the canonical glossary term or a short phrase that preserves raw numbers and inserts a glossary term.",
            "For numeric-unit corrections such as cents a share, preserve the original number and require dividend/share context.",
            "Use no_op for equivalent style variants such as cents per share -> cents a share.",
            "Do not rewrite common geopolitical or ordinary tokens such as U.S., EU, PC, or mobile into company names or tickers.",
            "Do not rewrite style, grammar, punctuation, or filler words unless part of the glossary-term correction.",
            "If a correction is plausible but not clearly supported, put it in needs_review instead of corrections.",
            "For gate_version v3, only propose a correction when source_text is a predefined ASR error form or a numeric-unit error pattern with explicit allowed-context evidence.",
        ],
        "response_schema": {
            "corrections": [
                {
                    "source_text": "exact phrase from raw_asr_text",
                    "replacement_text": "grounded replacement",
                    "canonical_term": "one canonical glossary term",
                    "reason": "short evidence reason",
                }
            ],
            "needs_review": [
                {
                    "source_text": "phrase",
                    "candidate_term": "term",
                    "reason": "why not safe",
                }
            ],
            "no_op": [
                {
                    "source_text": "phrase",
                    "candidate_term": "term",
                    "reason": "already acceptable or equivalent",
                }
            ],
        },
    }
    return [
        {
            "role": "system",
            "content": (
                "You are TalkWeaver's conservative RAG correction verifier. "
                "You propose only short, auditable term substitutions."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False),
        },
    ]


def _dedupe_decision_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for record in records:
        source = str(record.get("source_text", "")).casefold()
        term = str(
            record.get(
                "canonical_term",
                record.get("candidate_term", record.get("replacement_text", "")),
            )
        ).casefold()
        key = (source, term)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)
    return deduped


def _apply_llm_corrections(
    *,
    raw_text: str,
    payload: dict[str, Any],
    entries_by_key: dict[str, GlossaryEntry],
    gate_version: str = "v2",
) -> tuple[
    str,
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    corrected = raw_text
    applied: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    needs_review: list[dict[str, Any]] = []
    no_op: list[dict[str, Any]] = []
    corrections = payload.get("corrections", [])
    if not isinstance(corrections, list):
        corrections = []
    raw_needs_review = payload.get("needs_review", [])
    if isinstance(raw_needs_review, list):
        needs_review = [
            item
            for item in raw_needs_review
            if isinstance(item, dict)
        ]
    raw_no_op = payload.get("no_op", [])
    if isinstance(raw_no_op, list):
        no_op = [
            item
            for item in raw_no_op
            if isinstance(item, dict)
        ]
    for item in corrections:
        if not isinstance(item, dict):
            rejected.append({"item": item, "reason": "correction is not an object"})
            continue
        decision, reason = _gate_correction(
            item,
            raw_text=corrected,
            entries_by_key=entries_by_key,
            gate_version=gate_version,
        )
        record = dict(item)
        record["validation"] = reason
        record["gate_decision"] = decision
        if decision == "reject":
            rejected.append(record)
            continue
        if decision == "needs_review":
            needs_review.append(record)
            continue
        if decision == "no_op":
            no_op.append(record)
            continue
        corrected = _replace_once(
            corrected,
            str(item["source_text"]).strip(),
            str(item["replacement_text"]).strip(),
        )
        applied.append(record)
    return (
        corrected,
        _dedupe_decision_records(applied),
        _dedupe_decision_records(rejected),
        _dedupe_decision_records(needs_review),
        _dedupe_decision_records(no_op),
    )


def evaluate_rows(
    *,
    input_path: Path,
    glossary_path: Path,
    output_path: Path,
    summary_output_path: Path | None = None,
    markdown_output_path: Path | None = None,
    llm_config: LLMConfig,
    gate_version: str = "v2",
) -> list[dict[str, str]]:
    entries = load_reference_glossary(glossary_path)
    entries_by_key = _entry_by_key(entries)
    with input_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    results: list[dict[str, str]] = []
    for row in rows:
        candidates = _candidate_corrections(row["hypothesis_text"], entries)
        api_used = bool(candidates)
        if api_used:
            response = request_json_completion(
                llm_config,
                _build_messages(
                    row=row,
                    entries=entries,
                    candidates=candidates,
                    gate_version=gate_version,
                ),
            )
        else:
            response = {"corrections": [], "needs_review": []}
        corrected_text, applied, rejected, needs_review, no_op = _apply_llm_corrections(
            raw_text=row["hypothesis_text"],
            payload=response,
            entries_by_key=entries_by_key,
            gate_version=gate_version,
        )
        before_score = evaluate_text(
            row["reference_text"],
            row["hypothesis_text"],
            row.get("language"),
        )
        after_score = evaluate_text(
            row["reference_text"],
            corrected_text,
            row.get("language"),
        )
        reference_terms = _terms_present(
            entries,
            row["reference_text"],
            row.get("language", ""),
        )
        baseline_terms = _terms_present(
            entries,
            row["hypothesis_text"],
            row.get("language", ""),
        )
        corrected_terms = _terms_present(
            entries,
            corrected_text,
            row.get("language", ""),
        )
        before_terms = _term_metrics(reference_terms, baseline_terms)
        after_terms = _term_metrics(reference_terms, corrected_terms)
        results.append(
            {
                "clip_id": row.get("clip_id", ""),
                "dataset_name": row.get("dataset_name", ""),
                "language": row.get("language", ""),
                "model_name": row.get("model_name", ""),
                "wer_before": f"{float(before_score['error_rate']):.6f}",
                "wer_after": f"{float(after_score['error_rate']):.6f}",
                "wer_delta": f"{float(after_score['error_rate']) - float(before_score['error_rate']):.6f}",
                "reference_terms": _json(reference_terms),
                "baseline_terms": _json(baseline_terms),
                "corrected_terms": _json(corrected_terms),
                "missing_before": _json(before_terms["missing"]),
                "missing_after": _json(after_terms["missing"]),
                "term_recall_before": f"{before_terms['recall']:.6f}",
                "term_recall_after": f"{after_terms['recall']:.6f}",
                "term_f1_before": f"{before_terms['f1']:.6f}",
                "term_f1_after": f"{after_terms['f1']:.6f}",
                "api_used": str(api_used).lower(),
                "llm_provider": llm_config.provider,
                "llm_model": llm_config.model,
                "candidate_corrections": _json(candidates),
                "applied_corrections": _json(applied),
                "rejected_corrections": _json(rejected),
                "needs_review_corrections": _json(needs_review),
                "no_op_corrections": _json(no_op),
                "reference_text": row["reference_text"],
                "hypothesis_text": row["hypothesis_text"],
                "corrected_text": corrected_text,
                "claim_scope": (
                    f"External/predefined RAG glossary plus constrained LLM "
                    f"substitution using evidence gate {gate_version}. No "
                    "reference text is sent to the LLM."
                ),
                "notes": (
                    "Diagnostic Earnings-22 experiment; "
                    f"{gate_version} evidence gate accepts only glossary-grounded short "
                    "substitutions, rejects common-token entity rewrites, "
                    "and records equivalent wording as no_op; rows without "
                    "retrieved candidates skip the LLM call."
                ),
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=OUTPUT_COLUMNS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(results)
    if summary_output_path is not None:
        summary_rows = summarize_rows(results)
        summary_output_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=SUMMARY_COLUMNS,
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(summary_rows)
    if markdown_output_path is not None:
        write_markdown_report(markdown_output_path, rows=results)
    return results


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _load_json_list(value: str) -> list[Any]:
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return []
    return payload if isinstance(payload, list) else []


def summarize_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: defaultdict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["dataset_name"],
                row["language"],
                row["model_name"],
            )
        ].append(row)
    summaries: list[dict[str, str]] = []
    for (dataset_name, language, model_name), group in sorted(grouped.items()):
        wer_before = [float(row["wer_before"]) for row in group]
        wer_after = [float(row["wer_after"]) for row in group]
        applied_count = sum(
            len(_load_json_list(row["applied_corrections"]))
            for row in group
        )
        rejected_count = sum(
            len(_load_json_list(row["rejected_corrections"]))
            for row in group
        )
        review_count = sum(
            len(_load_json_list(row["needs_review_corrections"]))
            for row in group
        )
        no_op_count = sum(
            len(_load_json_list(row.get("no_op_corrections", "[]")))
            for row in group
        )
        summaries.append(
            {
                "dataset_name": dataset_name,
                "language": language,
                "model_name": model_name,
                "num_rows": str(len(group)),
                "mean_wer_before": f"{_mean(wer_before):.6f}",
                "mean_wer_after": f"{_mean(wer_after):.6f}",
                "mean_wer_delta": f"{_mean(wer_after) - _mean(wer_before):.6f}",
                "mean_term_recall_before": f"{_mean([float(row['term_recall_before']) for row in group]):.6f}",
                "mean_term_recall_after": f"{_mean([float(row['term_recall_after']) for row in group]):.6f}",
                "mean_term_f1_before": f"{_mean([float(row['term_f1_before']) for row in group]):.6f}",
                "mean_term_f1_after": f"{_mean([float(row['term_f1_after']) for row in group]):.6f}",
                "applied_correction_count": str(applied_count),
                "rejected_correction_count": str(rejected_count),
                "needs_review_count": str(review_count),
                "no_op_count": str(no_op_count),
                "api_used": str(any(row["api_used"] == "true" for row in group)).lower(),
                "llm_provider": group[0]["llm_provider"],
                "llm_model": group[0]["llm_model"],
                "claim_scope": group[0]["claim_scope"],
            }
        )
    return summaries


def _markdown_table(rows: list[dict[str, str]], columns: list[str]) -> str:
    if not rows:
        return "_No rows._\n"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(str(row.get(column, "")).replace("|", "\\|") for column in columns)
            + " |"
        )
    return "\n".join(lines) + "\n"


def _compact_corrections(value: str) -> str:
    corrections = _load_json_list(value)
    compact: list[str] = []
    for correction in corrections:
        if not isinstance(correction, dict):
            continue
        source = correction.get("source_text", "")
        replacement = correction.get("replacement_text", "")
        canonical = correction.get("canonical_term", "")
        if source or replacement:
            compact.append(f"{source} -> {replacement} ({canonical})")
    return "; ".join(compact)


def write_markdown_report(path: Path, *, rows: list[dict[str, str]]) -> None:
    summaries = summarize_rows(rows)
    example_rows = [
        {
            "clip_id": row["clip_id"],
            "model_name": row["model_name"],
            "wer_before": row["wer_before"],
            "wer_after": row["wer_after"],
            "term_recall_before": row["term_recall_before"],
            "term_recall_after": row["term_recall_after"],
            "applied": _compact_corrections(row["applied_corrections"]),
            "rejected_count": str(len(_load_json_list(row["rejected_corrections"]))),
            "no_op_count": str(len(_load_json_list(row.get("no_op_corrections", "[]")))),
        }
        for row in rows
    ]
    content = [
        "# Earnings-22 Finance RAG + LLM Correction",
        "",
        "This diagnostic experiment uses a predefined finance glossary and a conservative LLM verifier. The reference transcript is used only for scoring, not as prompt input.",
        "",
        "## Summary",
        "",
        _markdown_table(
            summaries,
            [
                "model_name",
                "num_rows",
                "mean_wer_before",
                "mean_wer_after",
                "mean_wer_delta",
                "mean_term_recall_before",
                "mean_term_recall_after",
                "applied_correction_count",
                "rejected_correction_count",
                "needs_review_count",
                "no_op_count",
            ],
        ),
        "## Data Scope",
        "",
        f"- Evaluated rows: {len(rows)} ASR rows across {len({row['clip_id'] for row in rows})} audio slice(s).",
        "- This is a diagnostic multi-file subset, not a final held-out benchmark.",
        "- The RAG glossary is treated as external/context knowledge; reference transcripts are used only for scoring.",
        "- ASR-specific error forms should be validated on additional held-out Earnings-22 files before making a final generalization claim.",
        "",
        "## Correction Examples",
        "",
        _markdown_table(
            example_rows,
            [
                "clip_id",
                "model_name",
                "wer_before",
                "wer_after",
                "term_recall_before",
                "term_recall_after",
                "applied",
                "rejected_count",
                "no_op_count",
            ],
        ),
        "## Interpretation",
        "",
        "- The intended win condition is better finance-term recall without unsupported rewrites.",
        "- Numeric-unit corrections must preserve the original number and require dividend/share context.",
        "- Equivalent wording such as cents per share is recorded as no_op rather than applied as a rescue.",
        "- Common tokens such as U.S. are rejected as entity rewrites unless represented by a non-ambiguous source form.",
        "- Style-only differences and filler cleanup are not counted as the core RAG contribution.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(content), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("experiments/results/asr_benchmark_earnings22_rag_smoke_combined.csv"),
    )
    parser.add_argument(
        "--glossary",
        type=Path,
        default=Path("data/controlled_terms/earnings22_finance_terms.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/results/earnings22_finance_rag_llm.csv"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("experiments/results/earnings22_finance_rag_llm_summary.csv"),
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("docs/earnings22_finance_rag_error_analysis.md"),
    )
    parser.add_argument(
        "--gate-version",
        choices=("v2", "v3"),
        default="v2",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_llm_config(correction_mode="llm")
    rows = evaluate_rows(
        input_path=args.input,
        glossary_path=args.glossary,
        output_path=args.output,
        summary_output_path=args.summary_output,
        markdown_output_path=args.markdown_output,
        llm_config=config,
        gate_version=args.gate_version,
    )
    print(
        f"Wrote {len(rows)} RAG+LLM correction rows to {args.output}. "
        f"provider={config.provider} model={config.model}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
