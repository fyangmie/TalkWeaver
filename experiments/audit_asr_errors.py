#!/usr/bin/env python3
"""Audit real ASR result rows for likely failure modes.

The audit is intentionally diagnostic: it identifies candidate error types
that should be inspected before claiming that a downstream correction module
improves real ASR performance.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from experiments.evaluate_terms import DOMAIN_TERMS  # noqa: E402
from experiments.metrics.text_normalization import (  # noqa: E402
    DISFLUENCY_TOKENS,
    is_mandarin_language,
    normalize_for_wer,
)


OUTPUT_COLUMNS = [
    "clip_id",
    "dataset_name",
    "language",
    "model_name",
    "metric_name",
    "error_rate",
    "cleaned_metric_name",
    "cleaned_error_rate",
    "reference_text",
    "hypothesis_text",
    "suspected_error_types",
    "candidate_terms_or_entities",
    "missing_reference_tokens",
    "term_eval_candidate",
    "notes",
]

COMMON_SENTENCE_STARTS = {
    "a",
    "an",
    "and",
    "but",
    "de",
    "for",
    "i",
    "il",
    "in",
    "it",
    "maintenant",
    "okay",
    "right",
    "so",
    "the",
    "then",
    "this",
    "um",
    "we",
}
NUMBER_WORDS = {
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "hundred",
    "thousand",
    "million",
    "billion",
}
STOP_TOKENS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "for",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "we",
    "with",
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _json_list(values: list[str]) -> str:
    return json.dumps(values, ensure_ascii=False)


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _contains_phrase(text: str, phrase: str, *, language: str) -> bool:
    if not phrase:
        return False
    if is_mandarin_language(language) or any(
        "\u4e00" <= character <= "\u9fff" for character in phrase
    ):
        return phrase in text
    normalized = normalize_for_wer(text)
    target = normalize_for_wer(phrase)
    if not target:
        return False
    return re.search(
        rf"(?<![a-z0-9]){re.escape(target)}(?![a-z0-9])",
        normalized,
    ) is not None


def _extract_glossary_terms(reference: str, language: str) -> list[str]:
    return [
        term
        for term in DOMAIN_TERMS
        if _contains_phrase(reference, term, language=language)
    ]


def _extract_acronyms(reference: str) -> list[str]:
    return _dedupe(
        re.findall(
            r"\b[A-Z]{2,}(?:[./-][A-Z0-9]+)*\b",
            reference,
        )
    )


def _extract_named_entities(reference: str) -> list[str]:
    pattern = re.compile(
        r"\b(?:[A-ZÀ-ÖØ-Þ][a-zÀ-ÖØ-öø-ÿ'’-]+|[A-Z]{2,})"
        r"(?:\s+(?:[A-ZÀ-ÖØ-Þ][a-zÀ-ÖØ-öø-ÿ'’-]+|[A-Z]{2,}))*\b"
    )
    candidates: list[str] = []
    for match in pattern.finditer(reference):
        value = " ".join(match.group(0).split())
        tokens = value.split()
        if normalize_for_wer(value) in DISFLUENCY_TOKENS:
            continue
        if len(tokens) == 1 and tokens[0].casefold() in COMMON_SENTENCE_STARTS:
            continue
        if len(tokens) == 1 and len(tokens[0]) <= 2:
            continue
        candidates.append(value)
    return _dedupe(candidates)


def _extract_numbers(text: str) -> list[str]:
    normalized = normalize_for_wer(text)
    digits = re.findall(r"\b\d+(?:[.,]\d+)?\b", normalized)
    words = [
        token
        for token in normalized.split()
        if token in NUMBER_WORDS
    ]
    return _dedupe([*digits, *words])


def _missing_tokens(reference: str, hypothesis: str, language: str) -> list[str]:
    if is_mandarin_language(language):
        missing = [
            character
            for character in reference
            if "\u4e00" <= character <= "\u9fff" and character not in hypothesis
        ]
        return _dedupe(missing)[:20]
    reference_tokens = normalize_for_wer(reference).split()
    hypothesis_tokens = set(normalize_for_wer(hypothesis).split())
    missing: list[str] = []
    for token in reference_tokens:
        if (
            token not in hypothesis_tokens
            and token not in STOP_TOKENS
            and len(token) > 2
        ):
            missing.append(token)
    return _dedupe(missing)[:20]


def _ratio(hypothesis: str, reference: str, language: str) -> float:
    if is_mandarin_language(language):
        reference_count = max(
            1,
            sum("\u4e00" <= character <= "\u9fff" for character in reference),
        )
        hypothesis_count = sum(
            "\u4e00" <= character <= "\u9fff" for character in hypothesis
        )
        return hypothesis_count / reference_count
    reference_tokens = normalize_for_wer(reference).split()
    hypothesis_tokens = normalize_for_wer(hypothesis).split()
    return len(hypothesis_tokens) / max(1, len(reference_tokens))


def audit_row(row: dict[str, str]) -> dict[str, str]:
    """Return one diagnostic audit row for an ASR benchmark result."""

    reference = row.get("reference_text", "")
    hypothesis = row.get("hypothesis_text", "")
    language = row.get("language", "")
    dataset = row.get("dataset_name", "")
    error_rate = _safe_float(row.get("error_rate"))
    cleaned_error_rate = _safe_float(row.get("cleaned_error_rate"), default=-1.0)

    glossary_terms = _extract_glossary_terms(reference, language)
    acronyms = [
        value
        for value in _extract_acronyms(reference)
        if not _contains_phrase(hypothesis, value, language=language)
    ]
    named_entities = [
        value
        for value in _extract_named_entities(reference)
        if not _contains_phrase(hypothesis, value, language=language)
    ]
    missing_glossary = [
        value
        for value in glossary_terms
        if not _contains_phrase(hypothesis, value, language=language)
    ]
    reference_numbers = _extract_numbers(reference)
    hypothesis_numbers = _extract_numbers(hypothesis)
    missing = _missing_tokens(reference, hypothesis, language)
    length_ratio = _ratio(hypothesis, reference, language)

    error_types: list[str] = []
    notes: list[str] = []
    if missing_glossary:
        error_types.append("professional_term_or_glossary")
    if acronyms:
        error_types.append("acronym_or_short_code")
    if named_entities:
        error_types.append("proper_noun_or_named_entity")
    if reference_numbers != hypothesis_numbers:
        error_types.append("number_or_quantity")
    if is_mandarin_language(language) and error_rate >= 0.2:
        error_types.append("mandarin_low_frequency_or_script")
    if "ami" in dataset.casefold() or set(normalize_for_wer(reference).split()) & DISFLUENCY_TOKENS:
        error_types.append("meeting_disfluency_or_truncation")
    if length_ratio < 0.72 and error_rate >= 0.2:
        error_types.append("truncation_or_omission")
        notes.append(f"hypothesis/reference length ratio={length_ratio:.2f}")
    if cleaned_error_rate >= 0 and cleaned_error_rate < error_rate:
        notes.append("cleaned meeting metric is lower after disfluency removal")
    if not error_types:
        error_types.append("general_asr_error")

    candidates = _dedupe([*missing_glossary, *acronyms, *named_entities])
    term_eval_candidate = bool(missing_glossary or acronyms or named_entities)
    return {
        "clip_id": row.get("clip_id", ""),
        "dataset_name": dataset,
        "language": language,
        "model_name": row.get("model_name", ""),
        "metric_name": row.get("metric_name", ""),
        "error_rate": row.get("error_rate", ""),
        "cleaned_metric_name": row.get("cleaned_metric_name", ""),
        "cleaned_error_rate": row.get("cleaned_error_rate", ""),
        "reference_text": reference,
        "hypothesis_text": hypothesis,
        "suspected_error_types": _json_list(_dedupe(error_types)),
        "candidate_terms_or_entities": _json_list(candidates),
        "missing_reference_tokens": _json_list(missing),
        "term_eval_candidate": str(term_eval_candidate).lower(),
        "notes": "; ".join(notes),
    }


def audit_asr_errors(
    input_path: str | Path,
    output_path: str | Path,
    *,
    min_error_rate: float = 0.0,
    top_n: int | None = None,
) -> list[dict[str, str]]:
    """Audit an ASR benchmark CSV and write diagnostic rows."""

    source = Path(input_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    with source.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("ASR benchmark CSV contains no rows.")
    required = {"clip_id", "reference_text", "hypothesis_text", "error_rate"}
    missing_columns = sorted(required - set(rows[0]))
    if missing_columns:
        raise KeyError(
            f"ASR benchmark CSV is missing: {', '.join(missing_columns)}"
        )

    selected = [
        row
        for row in rows
        if _safe_float(row.get("error_rate")) >= min_error_rate
    ]
    selected.sort(
        key=lambda row: (
            _safe_float(row.get("error_rate")),
            row.get("clip_id", ""),
            row.get("model_name", ""),
        ),
        reverse=True,
    )
    if top_n is not None:
        selected = selected[:top_n]
    audited = [audit_row(row) for row in selected]

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(audited)
    return audited


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("experiments/results/asr_benchmark_real.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/results/asr_error_audit_real.csv"),
    )
    parser.add_argument(
        "--min-error-rate",
        type=float,
        default=0.0,
        help="Only audit rows with at least this ASR error rate.",
    )
    parser.add_argument("--top-n", type=int)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        rows = audit_asr_errors(
            args.input,
            args.output,
            min_error_rate=args.min_error_rate,
            top_n=args.top_n,
        )
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"ASR error audit failed: {exc}", file=sys.stderr)
        return 2
    term_candidates = sum(row["term_eval_candidate"] == "true" for row in rows)
    print(
        f"Wrote {len(rows)} ASR audit rows to {args.output} "
        f"({term_candidates} candidate term/entity rows)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
