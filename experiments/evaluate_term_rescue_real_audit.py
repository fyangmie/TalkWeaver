#!/usr/bin/env python3
"""Evaluate glossary/RAG-style term rescue on real ASR benchmark rows."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.term_rescue import (  # noqa: E402
    GlossaryEntry,
    load_reference_glossary,
    retrieve_controlled_matches,
)
from backend.llm_config import LLMConfig, load_llm_config  # noqa: E402
from backend.term_verifier import verify_term_match  # noqa: E402
from experiments.metrics.text_normalization import (  # noqa: E402
    is_mandarin_language,
    normalize_for_wer,
)


RESULT_COLUMNS = [
    "clip_id",
    "dataset_name",
    "language",
    "model_name",
    "terms_source",
    "glossary_path",
    "metric_name",
    "error_rate",
    "reference_terms",
    "baseline_terms",
    "rescued_terms",
    "missing_before",
    "missing_after",
    "unexpected_before",
    "unexpected_after",
    "baseline_term_error_rate",
    "rescued_term_error_rate",
    "term_precision_before",
    "term_recall_before",
    "term_f1_before",
    "term_precision_after",
    "term_recall_after",
    "term_f1_after",
    "reference_text",
    "hypothesis_text",
    "corrected_text",
    "verifier",
    "verifier_decisions",
    "verifier_accept_count",
    "verifier_needs_review_count",
    "verifier_reject_count",
    "verifier_api_used_count",
    "verifier_fallback_used_count",
    "retrieved_candidates",
    "applied_corrections",
    "needs_review_count",
    "rejected_candidates",
    "claim_scope",
    "notes",
]
SUMMARY_COLUMNS = [
    "terms_source",
    "dataset_name",
    "language",
    "model_name",
    "num_rows",
    "total_reference_terms",
    "missing_terms_before",
    "missing_terms_after",
    "rescued_missing_terms",
    "mean_term_error_before",
    "mean_term_error_after",
    "mean_recall_before",
    "mean_recall_after",
    "mean_f1_before",
    "mean_f1_after",
    "verifier",
    "verifier_accept_count",
    "verifier_needs_review_count",
    "verifier_reject_count",
    "verifier_api_used_count",
    "verifier_fallback_used_count",
    "claim_scope",
    "notes",
]


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _dedupe(values: Iterable[str]) -> list[str]:
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


def _replace_phrase(text: str, phrase: str, replacement: str, *, language: str) -> str:
    if not phrase or not replacement:
        return text
    if is_mandarin_language(language) or any(
        "\u4e00" <= character <= "\u9fff" for character in phrase
    ):
        return text.replace(phrase, replacement)
    return re.sub(
        rf"(?<![A-Za-z0-9]){re.escape(phrase)}(?![A-Za-z0-9])",
        replacement,
        text,
        count=1,
        flags=re.IGNORECASE,
    )


def _entry_forms(entry: GlossaryEntry) -> list[str]:
    return _dedupe(
        [
            entry.canonical,
            *entry.aliases,
            *entry.spoken_forms,
        ]
    )


def _entry_key(value: str) -> str:
    return normalize_for_wer(value).casefold() or str(value).casefold()


def _term_in_text(entry: GlossaryEntry, text: str, language: str) -> bool:
    return any(
        _contains_phrase(text, form, language=language)
        for form in _entry_forms(entry)
    )


def _terms_present(
    entries: list[GlossaryEntry],
    text: str,
    language: str,
) -> list[str]:
    return [
        entry.canonical
        for entry in entries
        if _term_in_text(entry, text, language)
    ]


def _related(left: str, right: str) -> bool:
    left_key = _entry_key(left).replace("-", " ").replace(".", " ")
    right_key = _entry_key(right).replace("-", " ").replace(".", " ")
    left_key = " ".join(left_key.split())
    right_key = " ".join(right_key.split())
    return (
        left_key == right_key
        or bool(left_key and left_key in right_key)
        or bool(right_key and right_key in left_key)
    )


def _term_metrics(
    required: list[str],
    predicted: list[str],
) -> dict[str, Any]:
    required_unique = _dedupe(required)
    predicted_unique = _dedupe(predicted)
    correct = [
        term
        for term in predicted_unique
        if any(_related(term, required_term) for required_term in required_unique)
    ]
    unexpected = [
        term
        for term in predicted_unique
        if not any(_related(term, required_term) for required_term in required_unique)
    ]
    missing = [
        term
        for term in required_unique
        if not any(_related(term, predicted_term) for predicted_term in predicted_unique)
    ]
    matched_required = len(required_unique) - len(missing)
    precision = (
        len(correct) / len(predicted_unique)
        if predicted_unique
        else (1.0 if not required_unique else 0.0)
    )
    recall = (
        matched_required / len(required_unique)
        if required_unique
        else 1.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "missing": missing,
        "unexpected": unexpected,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "term_error_rate": len(missing) / len(required_unique)
        if required_unique
        else 0.0,
    }


def _load_markdown_glossary(path: Path) -> list[GlossaryEntry]:
    entries: dict[str, GlossaryEntry] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if "->" in stripped:
            source, target = [part.strip(" `") for part in stripped.split("->", 1)]
            if source and target:
                key = target.casefold()
                existing = entries.get(key, GlossaryEntry(canonical=target))
                entries[key] = GlossaryEntry(
                    canonical=existing.canonical,
                    aliases=existing.aliases,
                    spoken_forms=existing.spoken_forms,
                    asr_error_forms=tuple(
                        dict.fromkeys([*existing.asr_error_forms, source])
                    ),
                    language=existing.language,
                    category=existing.category,
                    allowed_contexts=existing.allowed_contexts,
                )
            continue
        if stripped.startswith("|") and stripped.endswith("|"):
            cells = [cell.strip() for cell in stripped.strip("|").split("|")]
            if not cells or cells[0].casefold() in {"term", "---"}:
                continue
            term = cells[0].strip("` ")
            if term and not set(term) <= {"-"}:
                entries.setdefault(term.casefold(), GlossaryEntry(canonical=term))
    return list(entries.values())


def load_glossary(path: str | Path) -> list[GlossaryEntry]:
    """Load JSON controlled glossary or a markdown domain-term list."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.suffix.lower() == ".json":
        return load_reference_glossary(source)
    return _load_markdown_glossary(source)


def _claim_scope(terms_source: str) -> str:
    if terms_source == "oracle_diagnostic":
        return (
            "Oracle/custom-vocabulary diagnostic. Use as an upper-bound or "
            "failure-analysis result, not as a generalization claim."
        )
    return (
        "External or pre-defined glossary result. Valid only for rows whose "
        "reference contains glossary terms."
    )


def _apply_term_rescue(
    hypothesis: str,
    language: str,
    matches: list[Any],
) -> tuple[str, list[str]]:
    corrected = hypothesis
    applied: list[str] = []
    for match in matches:
        before = corrected
        corrected = _replace_phrase(
            corrected,
            match.matched_form,
            match.canonical,
            language=language,
        )
        if corrected != before:
            applied.append(f"{match.matched_form} -> {match.canonical}")
    return corrected, applied


def evaluate_row(
    row: dict[str, str],
    entries: list[GlossaryEntry],
    *,
    terms_source: str,
    glossary_path: str,
    strategy: str,
    verifier: str,
    llm_config: LLMConfig | None = None,
) -> dict[str, str] | None:
    """Evaluate one ASR row; return None when no glossary terms are required."""

    language = row.get("language", "")
    reference = row.get("reference_text", "")
    hypothesis = row.get("hypothesis_text", "")
    required = _terms_present(entries, reference, language)
    if not required:
        return None

    accepted, rejected = retrieve_controlled_matches(
        hypothesis,
        entries,
        strategy=strategy,
        context=f"{row.get('dataset_name', '')} {language}",
    )
    all_matches = [*accepted, *rejected]
    candidate_context = "; ".join(
        [
            f"dataset={row.get('dataset_name', '')}",
            f"language={language}",
            "retrieved_candidate_set="
            + ", ".join(
                f"{match.matched_form}->{match.canonical}"
                for match in all_matches
            ),
        ]
    )
    verification_records: list[dict[str, Any]] = []
    safe_matches = []
    for match in all_matches:
        verification = verify_term_match(
            match,
            raw_text=hypothesis,
            context=candidate_context,
            language=language,
            terms_source=terms_source,
            verifier=verifier,  # type: ignore[arg-type]
            llm_config=llm_config,
        )
        verification_records.append(
            {
                "canonical": match.canonical,
                "matched_form": match.matched_form,
                "retrieval_method": match.retrieval_method,
                "retrieval_score": match.score,
                "decision": verification.decision,
                "reason": verification.reason,
                "backend": verification.backend,
                "api_used": verification.api_used,
                "fallback_used": verification.fallback_used,
            }
        )
        if verification.accepted:
            safe_matches.append(match)
    corrected_text, applied = _apply_term_rescue(
        hypothesis,
        language,
        safe_matches,
    )
    before_terms = _terms_present(entries, hypothesis, language)
    after_terms = _terms_present(entries, corrected_text, language)
    before = _term_metrics(required, before_terms)
    after = _term_metrics(required, after_terms)
    rescued = [
        term
        for term in before["missing"]
        if not any(_related(term, missing) for missing in after["missing"])
    ]
    notes: list[str] = []
    if terms_source == "oracle_diagnostic":
        notes.append("oracle/custom vocabulary diagnostic")
    if rejected:
        notes.append("some candidates require review and were not applied")
    accept_count = sum(
        record["decision"] == "accept" for record in verification_records
    )
    review_count = sum(
        record["decision"] == "needs_review" for record in verification_records
    )
    reject_count = sum(
        record["decision"] == "reject" for record in verification_records
    )
    api_used_count = sum(
        bool(record["api_used"]) for record in verification_records
    )
    fallback_used_count = sum(
        bool(record["fallback_used"]) for record in verification_records
    )
    return {
        "clip_id": row.get("clip_id", ""),
        "dataset_name": row.get("dataset_name", ""),
        "language": language,
        "model_name": row.get("model_name", ""),
        "terms_source": terms_source,
        "glossary_path": glossary_path,
        "metric_name": row.get("metric_name", ""),
        "error_rate": row.get("error_rate", ""),
        "reference_terms": _json(required),
        "baseline_terms": _json(before_terms),
        "rescued_terms": _json(rescued),
        "missing_before": _json(before["missing"]),
        "missing_after": _json(after["missing"]),
        "unexpected_before": _json(before["unexpected"]),
        "unexpected_after": _json(after["unexpected"]),
        "baseline_term_error_rate": f"{before['term_error_rate']:.6f}",
        "rescued_term_error_rate": f"{after['term_error_rate']:.6f}",
        "term_precision_before": f"{before['precision']:.6f}",
        "term_recall_before": f"{before['recall']:.6f}",
        "term_f1_before": f"{before['f1']:.6f}",
        "term_precision_after": f"{after['precision']:.6f}",
        "term_recall_after": f"{after['recall']:.6f}",
        "term_f1_after": f"{after['f1']:.6f}",
        "reference_text": reference,
        "hypothesis_text": hypothesis,
        "corrected_text": corrected_text,
        "verifier": verifier,
        "verifier_decisions": _json(verification_records),
        "verifier_accept_count": str(accept_count),
        "verifier_needs_review_count": str(review_count),
        "verifier_reject_count": str(reject_count),
        "verifier_api_used_count": str(api_used_count),
        "verifier_fallback_used_count": str(fallback_used_count),
        "retrieved_candidates": _json(
            [
                {
                    "canonical": match.canonical,
                    "matched_form": match.matched_form,
                    "score": match.score,
                    "method": match.retrieval_method,
                    "safe_to_apply": match.safe_to_apply,
                }
                for match in accepted
            ]
        ),
        "applied_corrections": _json(applied),
        "needs_review_count": str(review_count),
        "rejected_candidates": _json(
            [
                {
                    "canonical": match.canonical,
                    "matched_form": match.matched_form,
                    "score": match.score,
                    "reason": match.context_reason,
                }
                for match in rejected
            ]
        ),
        "claim_scope": _claim_scope(terms_source),
        "notes": "; ".join(notes),
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: defaultdict[tuple[str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["terms_source"],
                row["dataset_name"],
                row["language"],
                row["model_name"],
                row["verifier"],
            )
        ].append(row)
    summaries: list[dict[str, str]] = []
    for (
        terms_source,
        dataset_name,
        language,
        model_name,
        verifier,
    ), group in sorted(grouped.items()):
        total_terms = sum(len(json.loads(row["reference_terms"])) for row in group)
        missing_before = sum(len(json.loads(row["missing_before"])) for row in group)
        missing_after = sum(len(json.loads(row["missing_after"])) for row in group)
        accept_count = sum(int(row["verifier_accept_count"]) for row in group)
        review_count = sum(
            int(row["verifier_needs_review_count"]) for row in group
        )
        reject_count = sum(int(row["verifier_reject_count"]) for row in group)
        api_used_count = sum(
            int(row["verifier_api_used_count"]) for row in group
        )
        fallback_used_count = sum(
            int(row["verifier_fallback_used_count"]) for row in group
        )
        summaries.append(
            {
                "terms_source": terms_source,
                "dataset_name": dataset_name,
                "language": language,
                "model_name": model_name,
                "num_rows": str(len(group)),
                "total_reference_terms": str(total_terms),
                "missing_terms_before": str(missing_before),
                "missing_terms_after": str(missing_after),
                "rescued_missing_terms": str(missing_before - missing_after),
                "mean_term_error_before": f"{_mean([float(row['baseline_term_error_rate']) for row in group]):.6f}",
                "mean_term_error_after": f"{_mean([float(row['rescued_term_error_rate']) for row in group]):.6f}",
                "mean_recall_before": f"{_mean([float(row['term_recall_before']) for row in group]):.6f}",
                "mean_recall_after": f"{_mean([float(row['term_recall_after']) for row in group]):.6f}",
                "mean_f1_before": f"{_mean([float(row['term_f1_before']) for row in group]):.6f}",
                "mean_f1_after": f"{_mean([float(row['term_f1_after']) for row in group]):.6f}",
                "verifier": verifier,
                "verifier_accept_count": str(accept_count),
                "verifier_needs_review_count": str(review_count),
                "verifier_reject_count": str(reject_count),
                "verifier_api_used_count": str(api_used_count),
                "verifier_fallback_used_count": str(fallback_used_count),
                "claim_scope": _claim_scope(terms_source),
                "notes": (
                    "Rows without reference glossary terms are excluded from "
                    "the aggregate."
                ),
            }
        )
    return summaries


def evaluate_term_rescue_real_audit(
    *,
    input_path: str | Path,
    glossary_path: str | Path,
    output_path: str | Path,
    summary_output_path: str | Path | None = None,
    terms_source: str = "external_or_predefined",
    strategy: str = "fused",
    verifier: str = "rule",
    allow_empty: bool = False,
) -> list[dict[str, str]]:
    """Run term rescue audit on rows that contain reference glossary terms."""

    if terms_source not in {"external_or_predefined", "oracle_diagnostic"}:
        raise ValueError("Unsupported --terms-source.")
    if verifier not in {"rule", "llm", "llm_with_rule_fallback"}:
        raise ValueError("Unsupported --verifier.")
    llm_config = None
    if verifier != "rule":
        llm_config = load_llm_config(
            correction_mode="llm"
            if verifier == "llm"
            else "llm_with_rule_fallback"
        )
    entries = load_glossary(glossary_path)
    if not entries:
        raise ValueError("Glossary contains no terms.")
    with Path(input_path).open(encoding="utf-8", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    if not source_rows:
        raise ValueError("ASR benchmark CSV contains no rows.")
    required_columns = {"clip_id", "reference_text", "hypothesis_text"}
    missing_columns = sorted(required_columns - set(source_rows[0]))
    if missing_columns:
        raise KeyError(
            f"ASR benchmark CSV is missing: {', '.join(missing_columns)}"
        )

    result_rows = [
        result
        for row in source_rows
        if (
            result := evaluate_row(
                row,
                entries,
                terms_source=terms_source,
                glossary_path=str(glossary_path),
                strategy=strategy,
                verifier=verifier,
                llm_config=llm_config,
            )
        )
        is not None
    ]
    if not result_rows and not allow_empty:
        raise ValueError(
            "No ASR rows contained reference glossary terms. Use a targeted "
            "glossary/dataset or pass --allow-empty for a boundary check."
        )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(result_rows)

    if summary_output_path is not None:
        summaries = summarize_rows(result_rows)
        summary_destination = Path(summary_output_path)
        summary_destination.parent.mkdir(parents=True, exist_ok=True)
        with summary_destination.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
            writer.writeheader()
            writer.writerows(summaries)
    return result_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("experiments/results/asr_benchmark_real.csv"),
    )
    parser.add_argument(
        "--glossary",
        type=Path,
        default=Path("docs/knowledge_base/domain_terms.md"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/results/term_rescue_real_audit.csv"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("experiments/results/term_rescue_real_audit_summary.csv"),
    )
    parser.add_argument(
        "--terms-source",
        choices=("external_or_predefined", "oracle_diagnostic"),
        default="external_or_predefined",
        help="Boundary label for how the glossary was obtained.",
    )
    parser.add_argument(
        "--strategy",
        choices=("exact_glossary", "fuzzy", "phonetic_like", "fused"),
        default="fused",
    )
    parser.add_argument(
        "--verifier",
        choices=("rule", "llm", "llm_with_rule_fallback"),
        default="rule",
        help="Candidate verification layer before applying term rescue.",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Write empty outputs instead of failing when no rows contain terms.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        rows = evaluate_term_rescue_real_audit(
            input_path=args.input,
            glossary_path=args.glossary,
            output_path=args.output,
            summary_output_path=args.summary_output,
            terms_source=args.terms_source,
            strategy=args.strategy,
            verifier=args.verifier,
            allow_empty=args.allow_empty,
        )
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        print(f"Real term rescue audit failed: {exc}", file=sys.stderr)
        return 2
    print(
        f"Wrote {len(rows)} real term rescue audit rows to {args.output}. "
        f"terms_source={args.terms_source}; verifier={args.verifier}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
