#!/usr/bin/env python3
"""Classify ASR token errors into actionable failure categories."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.metrics.text_normalization import (  # noqa: E402
    DISFLUENCY_TOKENS,
    normalize_for_cleaned_wer,
    normalize_for_wer,
)


DETAIL_COLUMNS = [
    "clip_id",
    "dataset_name",
    "language",
    "model_name",
    "metric_name",
    "error_rate",
    "cleaned_metric_name",
    "cleaned_error_rate",
    "operation",
    "error_category",
    "impact",
    "reference_span",
    "hypothesis_span",
    "reference_tokens",
    "hypothesis_tokens",
    "reason",
    "recommendation",
]

SUMMARY_COLUMNS = [
    "dataset_name",
    "language",
    "model_name",
    "error_category",
    "count",
    "share",
    "mean_wer",
    "mean_cleaned_wer",
    "example_reference",
    "example_hypothesis",
    "recommendation",
]

MODEL_COLUMNS = [
    "dataset_name",
    "language",
    "model_name",
    "rows",
    "mean_wer",
    "mean_cleaned_wer",
    "strict_cleaned_gap",
    "top_error_categories",
]


FUNCTION_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "but",
    "by",
    "for",
    "from",
    "has",
    "have",
    "he",
    "her",
    "him",
    "i",
    "in",
    "is",
    "it",
    "its",
    "me",
    "my",
    "now",
    "of",
    "on",
    "or",
    "our",
    "she",
    "so",
    "that",
    "the",
    "their",
    "there",
    "these",
    "they",
    "this",
    "to",
    "we",
    "with",
    "you",
    "your",
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
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
    "hundred",
    "thousand",
    "million",
    "billion",
}
MONTHS = {
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
}
UNIT_TERMS = {
    "%",
    "basis",
    "bps",
    "cent",
    "cents",
    "percent",
    "percentage",
    "per",
    "rate",
    "rates",
    "share",
    "shares",
}
FINANCE_TERMS = {
    "aspen",
    "brand",
    "brands",
    "constant",
    "covid",
    "debt",
    "digit",
    "dividend",
    "earnings",
    "exchange",
    "growth",
    "leverage",
    "oncology",
    "operation",
    "operations",
    "organic",
    "regional",
    "reported",
    "revenue",
    "segmental",
    "sterile",
}
STYLE_PAIRS = {
    ("gonna", "going to"),
    ("cuz", "because"),
    ("double-digit", "double digit"),
    ("year-on-year", "year on year"),
}


@dataclass(frozen=True)
class ClassifiedError:
    clip_id: str
    dataset_name: str
    language: str
    model_name: str
    metric_name: str
    error_rate: str
    cleaned_metric_name: str
    cleaned_error_rate: str
    operation: str
    error_category: str
    impact: str
    reference_span: str
    hypothesis_span: str
    reference_tokens: list[str]
    hypothesis_tokens: list[str]
    reason: str
    recommendation: str

    def as_row(self) -> dict[str, str]:
        return {
            "clip_id": self.clip_id,
            "dataset_name": self.dataset_name,
            "language": self.language,
            "model_name": self.model_name,
            "metric_name": self.metric_name,
            "error_rate": self.error_rate,
            "cleaned_metric_name": self.cleaned_metric_name,
            "cleaned_error_rate": self.cleaned_error_rate,
            "operation": self.operation,
            "error_category": self.error_category,
            "impact": self.impact,
            "reference_span": self.reference_span,
            "hypothesis_span": self.hypothesis_span,
            "reference_tokens": json.dumps(self.reference_tokens, ensure_ascii=False),
            "hypothesis_tokens": json.dumps(self.hypothesis_tokens, ensure_ascii=False),
            "reason": self.reason,
            "recommendation": self.recommendation,
        }


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in {None, ""}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _cleaned_or_error(row: dict[str, str]) -> float:
    cleaned = row.get("cleaned_error_rate")
    if cleaned not in {None, ""}:
        return _safe_float(cleaned)
    return _safe_float(row.get("error_rate"))


def _tokens(text: str) -> list[str]:
    return normalize_for_wer(text).split()


def _span(tokens: list[str]) -> str:
    return " ".join(tokens)


def _contains_number(tokens: list[str]) -> bool:
    return any(token.isdigit() or token in NUMBER_WORDS for token in tokens)


def _contains_unit(tokens: list[str]) -> bool:
    return any(token in UNIT_TERMS for token in tokens)


def _contains_date(tokens: list[str]) -> bool:
    return any(token in MONTHS for token in tokens) or any(
        token.isdigit() and len(token) == 4 for token in tokens
    )


def _contains_finance_term(tokens: list[str]) -> bool:
    return any(token in FINANCE_TERMS for token in tokens)


def _only_function_words(tokens: list[str]) -> bool:
    return bool(tokens) and all(token in FUNCTION_WORDS for token in tokens)


def _contains_disfluency(tokens: list[str]) -> bool:
    return any(token in DISFLUENCY_TOKENS for token in tokens)


def _is_repeated_word_deletion(reference: list[str], hypothesis: list[str]) -> bool:
    if hypothesis:
        return False
    return len(reference) > 1 and len(set(reference)) == 1


def _is_style_pair(reference: list[str], hypothesis: list[str]) -> bool:
    left = _span(reference)
    right = _span(hypothesis)
    return (left, right) in STYLE_PAIRS or (right, left) in STYLE_PAIRS


def classify_token_error(
    operation: str,
    reference_tokens: list[str],
    hypothesis_tokens: list[str],
    *,
    reference_context: list[str] | None = None,
    hypothesis_context: list[str] | None = None,
) -> tuple[str, str, str, str]:
    """Return category, impact, reason, and recommendation."""

    actual = [*reference_tokens, *hypothesis_tokens]
    context = [
        *(reference_context if reference_context is not None else reference_tokens),
        *(hypothesis_context if hypothesis_context is not None else hypothesis_tokens),
    ]
    if _contains_date(context) and _contains_number(context):
        return (
            "date_time_error",
            "meaning_likely_changed",
            "date or year words were substituted, deleted, or inserted",
            "Use stronger ASR or audio-backed verification; RAG alone is weak here.",
        )
    if _contains_number(context) and _contains_unit(context):
        return (
            "number_unit_error",
            "meaning_likely_changed",
            "number or measurement-unit wording changed",
            "Add numeric-unit candidate rules and verify against local context/audio.",
        )
    if _contains_finance_term(actual):
        return (
            "domain_term_error",
            "meaning_may_change",
            "finance/company/domain term was altered or missed",
            "Use a predeclared finance glossary plus conservative LLM verifier.",
        )
    if (
        _contains_disfluency(actual)
        or _is_repeated_word_deletion(reference_tokens, hypothesis_tokens)
        or _is_style_pair(reference_tokens, hypothesis_tokens)
    ):
        return (
            "disfluency_or_style_error",
            "mostly_style_or_reference_policy",
            "filler, contraction, repeated word, or spelling style changed",
            "Report cleaned WER separately; do not optimize RAG for this category.",
        )
    if _only_function_words(actual):
        return (
            "function_word_error",
            "meaning_may_change",
            "short pronoun/preposition/function-word change",
            "Use stronger ASR/context; glossary RAG should usually avoid this.",
        )
    if operation in {"delete", "insert"}:
        return (
            "omission_or_insertion",
            "meaning_may_change",
            "content was omitted or inserted",
            "Inspect audio and segment boundaries; consider VAD/chunking changes.",
        )
    return (
        "semantic_word_error",
        "meaning_may_change",
        "content word substitution not covered by specialized categories",
        "Collect more examples, then decide whether ASR API or targeted RAG helps.",
    )


def classify_row(row: dict[str, str]) -> list[ClassifiedError]:
    reference = row.get("normalized_reference") or normalize_for_wer(
        row.get("reference_text", "")
    )
    hypothesis = row.get("normalized_hypothesis") or normalize_for_wer(
        row.get("hypothesis_text", "")
    )
    reference_tokens = reference.split()
    hypothesis_tokens = hypothesis.split()
    matcher = SequenceMatcher(
        None,
        reference_tokens,
        hypothesis_tokens,
        autojunk=False,
    )
    errors: list[ClassifiedError] = []
    for operation, ref_start, ref_end, hyp_start, hyp_end in matcher.get_opcodes():
        if operation == "equal":
            continue
        ref_span = reference_tokens[ref_start:ref_end]
        hyp_span = hypothesis_tokens[hyp_start:hyp_end]
        ref_context = reference_tokens[
            max(0, ref_start - 2) : min(len(reference_tokens), ref_end + 2)
        ]
        hyp_context = hypothesis_tokens[
            max(0, hyp_start - 2) : min(len(hypothesis_tokens), hyp_end + 2)
        ]
        category, impact, reason, recommendation = classify_token_error(
            operation,
            ref_span,
            hyp_span,
            reference_context=ref_context,
            hypothesis_context=hyp_context,
        )
        errors.append(
            ClassifiedError(
                clip_id=row.get("clip_id", ""),
                dataset_name=row.get("dataset_name", ""),
                language=row.get("language", ""),
                model_name=row.get("model_name", ""),
                metric_name=row.get("metric_name", ""),
                error_rate=row.get("error_rate", ""),
                cleaned_metric_name=row.get("cleaned_metric_name", ""),
                cleaned_error_rate=row.get("cleaned_error_rate", ""),
                operation=operation,
                error_category=category,
                impact=impact,
                reference_span=_span(ref_span),
                hypothesis_span=_span(hyp_span),
                reference_tokens=ref_span,
                hypothesis_tokens=hyp_span,
                reason=reason,
                recommendation=recommendation,
            )
        )
    return errors


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize_errors(
    source_rows: list[dict[str, str]],
    errors: list[ClassifiedError],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    row_groups: defaultdict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in source_rows:
        row_groups[
            (
                row.get("dataset_name", ""),
                row.get("language", ""),
                row.get("model_name", ""),
            )
        ].append(row)

    error_groups: defaultdict[tuple[str, str, str, str], list[ClassifiedError]] = defaultdict(list)
    for error in errors:
        error_groups[
            (
                error.dataset_name,
                error.language,
                error.model_name,
                error.error_category,
            )
        ].append(error)

    summary_rows: list[dict[str, str]] = []
    for key, group in sorted(error_groups.items()):
        dataset_name, language, model_name, category = key
        model_rows = row_groups[(dataset_name, language, model_name)]
        total_errors = sum(
            len(error_group)
            for group_key, error_group in error_groups.items()
            if group_key[:3] == key[:3]
        )
        example = group[0]
        summary_rows.append(
            {
                "dataset_name": dataset_name,
                "language": language,
                "model_name": model_name,
                "error_category": category,
                "count": str(len(group)),
                "share": f"{(len(group) / total_errors if total_errors else 0.0):.6f}",
                "mean_wer": f"{_mean([_safe_float(row.get('error_rate')) for row in model_rows]):.6f}",
                "mean_cleaned_wer": f"{_mean([_cleaned_or_error(row) for row in model_rows]):.6f}",
                "example_reference": example.reference_span,
                "example_hypothesis": example.hypothesis_span,
                "recommendation": example.recommendation,
            }
        )

    model_rows_out: list[dict[str, str]] = []
    for key, rows in sorted(row_groups.items()):
        dataset_name, language, model_name = key
        model_errors = [
            error
            for error in errors
            if (
                error.dataset_name,
                error.language,
                error.model_name,
            )
            == key
        ]
        counts = Counter(error.error_category for error in model_errors)
        mean_wer = _mean([_safe_float(row.get("error_rate")) for row in rows])
        mean_cleaned = _mean([_cleaned_or_error(row) for row in rows])
        model_rows_out.append(
            {
                "dataset_name": dataset_name,
                "language": language,
                "model_name": model_name,
                "rows": str(len(rows)),
                "mean_wer": f"{mean_wer:.6f}",
                "mean_cleaned_wer": f"{mean_cleaned:.6f}",
                "strict_cleaned_gap": f"{(mean_wer - mean_cleaned):.6f}",
                "top_error_categories": json.dumps(
                    counts.most_common(5),
                    ensure_ascii=False,
                ),
            }
        )
    return summary_rows, model_rows_out


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


def write_markdown_report(
    path: Path,
    *,
    detail_rows: list[dict[str, str]],
    summary_rows: list[dict[str, str]],
    model_rows: list[dict[str, str]],
) -> None:
    top_examples = detail_rows[:12]
    content = [
        "# Baseline ASR Error Pattern Analysis",
        "",
        "This report separates strict WER from actionable error types. "
        "It is diagnostic and should not be presented as a final benchmark claim.",
        "",
        "## Model Summary",
        "",
        _markdown_table(
            model_rows,
            [
                "dataset_name",
                "language",
                "model_name",
                "rows",
                "mean_wer",
                "mean_cleaned_wer",
                "strict_cleaned_gap",
                "top_error_categories",
            ],
        ),
        "## Error Categories",
        "",
        _markdown_table(
            summary_rows,
            [
                "dataset_name",
                "language",
                "model_name",
                "error_category",
                "count",
                "share",
                "example_reference",
                "example_hypothesis",
                "recommendation",
            ],
        ),
        "## Representative Errors",
        "",
        _markdown_table(
            top_examples,
            [
                "clip_id",
                "model_name",
                "error_category",
                "reference_span",
                "hypothesis_span",
                "impact",
            ],
        ),
        "## Interpretation",
        "",
        "- RAG is most defensible for `domain_term_error` and some `number_unit_error` cases.",
        "- A text-only LLM can judge whether a correction is safe, but it cannot hear the audio.",
        "- `disfluency_or_style_error` should be handled with cleaned WER, not with aggressive correction.",
        "- Persistent date, number, and short-word errors are candidates for a stronger ASR baseline or audio-backed API comparison.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(content), encoding="utf-8")


def classify_asr_error_patterns(
    *,
    input_path: str | Path | list[str | Path],
    output_path: str | Path,
    summary_output_path: str | Path,
    model_summary_output_path: str | Path | None = None,
    markdown_output_path: str | Path | None = None,
) -> list[ClassifiedError]:
    input_paths = (
        [input_path]
        if isinstance(input_path, (str, Path))
        else list(input_path)
    )
    source_rows: list[dict[str, str]] = []
    for input_item in input_paths:
        source = Path(input_item)
        with source.open(encoding="utf-8", newline="") as handle:
            source_rows.extend(csv.DictReader(handle))
    if not source_rows:
        raise ValueError("ASR benchmark CSV contains no rows.")
    errors = [
        error
        for row in source_rows
        for error in classify_row(row)
    ]
    detail_rows = [error.as_row() for error in errors]
    summary_rows, model_rows = summarize_errors(source_rows, errors)

    _write_csv(Path(output_path), DETAIL_COLUMNS, detail_rows)
    _write_csv(Path(summary_output_path), SUMMARY_COLUMNS, summary_rows)
    if model_summary_output_path is not None:
        _write_csv(Path(model_summary_output_path), MODEL_COLUMNS, model_rows)
    if markdown_output_path is not None:
        write_markdown_report(
            Path(markdown_output_path),
            detail_rows=detail_rows,
            summary_rows=summary_rows,
            model_rows=model_rows,
        )
    return errors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        default=[
            Path("experiments/results/asr_benchmark_real.csv"),
            Path("experiments/results/asr_benchmark_ami_no_vad_real.csv"),
            Path("experiments/results/asr_benchmark_earnings22_rag_smoke_combined.csv"),
        ],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/results/asr_error_patterns_real_combined.csv"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("experiments/results/asr_error_patterns_real_combined_summary.csv"),
    )
    parser.add_argument(
        "--model-summary-output",
        type=Path,
        default=Path("experiments/results/asr_error_patterns_real_combined_model_summary.csv"),
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("docs/baseline_error_analysis_real_combined.md"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        errors = classify_asr_error_patterns(
            input_path=args.input,
            output_path=args.output,
            summary_output_path=args.summary_output,
            model_summary_output_path=args.model_summary_output,
            markdown_output_path=args.markdown_output,
        )
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"ASR error pattern classification failed: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote {len(errors)} classified ASR error spans to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
