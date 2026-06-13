#!/usr/bin/env python3
"""Run controlled technical-term retrieval and correction safety variants."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.constrained_correction import apply_constrained_correction  # noqa: E402
from backend.llm_config import load_llm_config  # noqa: E402
from backend.schemas import TemporalAnchor  # noqa: E402
from backend.term_rescue import (  # noqa: E402
    GlossaryEntry,
    TermMatch,
    load_reference_glossary,
    matches_to_candidates,
    retrieve_controlled_matches,
)
from experiments.metrics.text_metrics import evaluate_text  # noqa: E402


BASE_VARIANTS = (
    "no_retrieval",
    "exact_glossary",
    "fuzzy",
    "phonetic_like",
    "fused",
    "fused_plus_rule_correction",
)
LLM_VARIANT = "fused_plus_llm_correction"
OUTPUT_COLUMNS = [
    "case_id",
    "language",
    "difficulty",
    "variant",
    "raw_asr_text",
    "reference_text",
    "retrieved_candidates",
    "applied_corrections",
    "corrected_text",
    "expected_terms",
    "true_positive_terms",
    "false_positive_terms",
    "missed_terms",
    "unsupported_changes",
    "needs_review",
    "api_used",
    "fallback_used",
    "llm_provider",
    "llm_model",
    "prompt_version",
    "term_precision",
    "term_recall",
    "term_f1",
    "text_error_before",
    "text_error_after",
    "metric_name",
    "correction_error",
    "notes",
]


def load_cases(path: str | Path) -> list[dict[str, Any]]:
    """Load and validate text-only controlled JSONL fixtures."""

    required = {
        "case_id",
        "language",
        "raw_asr_text",
        "reference_text",
        "expected_terms",
        "expected_corrections",
        "negative_terms",
        "context",
        "difficulty",
        "notes",
    }
    cases: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"Case line {line_number} must be an object.")
        missing = sorted(required - payload.keys())
        if missing:
            raise ValueError(
                f"Case line {line_number} is missing: {', '.join(missing)}"
            )
        payload["fixture_type"] = "controlled_technical_term"
        cases.append(payload)
    if not cases:
        raise ValueError("Controlled term fixture contains no cases.")
    return cases


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _term_metrics(
    expected_terms: list[str],
    matches: list[TermMatch],
) -> tuple[list[str], list[str], list[str], float, float, float]:
    def related(left: str, right: str) -> bool:
        left_value = left.casefold().replace("-", " ").replace(".", " ")
        right_value = right.casefold().replace("-", " ").replace(".", " ")
        left_value = " ".join(left_value.split())
        right_value = " ".join(right_value.split())
        return (
            left_value == right_value
            or left_value in right_value
            or right_value in left_value
        )

    expected = list(dict.fromkeys(expected_terms))
    predicted = list(
        dict.fromkeys(match.canonical for match in matches)
    )
    true_positive = [
        term
        for term in predicted
        if any(related(term, expected_term) for expected_term in expected)
    ]
    false_positive = [
        term
        for term in predicted
        if not any(related(term, expected_term) for expected_term in expected)
    ]
    missed = [
        term
        for term in expected
        if not any(related(term, predicted_term) for predicted_term in predicted)
    ]
    matched_expected_count = len(expected) - len(missed)
    precision = (
        len(true_positive) / len(predicted)
        if predicted
        else (1.0 if not expected else 0.0)
    )
    recall = (
        matched_expected_count / len(expected)
        if expected
        else 1.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return true_positive, false_positive, missed, precision, recall, f1


def _applied_corrections(
    case: dict[str, Any],
    corrected_text: str,
) -> list[str]:
    raw = str(case["raw_asr_text"])
    if corrected_text == raw:
        return []
    applied: list[str] = []
    for source, target in dict(case["expected_corrections"]).items():
        if (
            str(source).casefold() in raw.casefold()
            and str(target).casefold() in corrected_text.casefold()
        ):
            applied.append(f"{source} -> {target}")
    return applied


def _retrieve(
    case: dict[str, Any],
    entries: list[GlossaryEntry],
    variant: str,
) -> tuple[list[TermMatch], list[TermMatch]]:
    if variant == "no_retrieval":
        return [], []
    strategy = (
        "fused"
        if variant.startswith("fused")
        else variant
    )
    return retrieve_controlled_matches(
        str(case["raw_asr_text"]),
        entries,
        strategy=strategy,
        context=str(case["context"]),
    )


def _correct(
    case: dict[str, Any],
    matches: list[TermMatch],
    *,
    mode: str,
    runtime_config: Any = None,
) -> tuple[str, dict[str, Any]]:
    anchor = TemporalAnchor(
        anchor_id=f"{case['case_id']}_anchor_001",
        clip_id=str(case["case_id"]),
        start=0.0,
        end=max(1.0, len(str(case["raw_asr_text"]).split()) * 0.4),
        speaker="SPEAKER_00",
        speakers=["SPEAKER_00"],
        raw_text=str(case["raw_asr_text"]),
        language=str(case["language"]),
        confidence=0.9,
        asr_confidence=0.8,
        diarization_confidence=1.0,
        retrieved_terms=[match.canonical for match in matches],
    )
    candidates = matches_to_candidates(matches, anchor_id=anchor.anchor_id)
    anchors, audits, _mode = apply_constrained_correction(
        [anchor],
        candidates,
        [],
        llm_config={
            "correction_mode": mode,
            "runtime_config": runtime_config,
        },
    )
    audit = audits[0]
    return anchors[0].corrected_text or anchors[0].raw_text, audit.to_dict()


def run_experiment(
    *,
    cases_path: str | Path,
    terms_path: str | Path,
    output_path: str | Path,
    candidates_output_path: str | Path,
    include_llm_if_configured: bool = False,
) -> list[dict[str, Any]]:
    """Run all offline variants and an optional strict real-LLM variant."""

    cases = load_cases(cases_path)
    entries = load_reference_glossary(terms_path)
    variants = list(BASE_VARIANTS)
    llm_config = None
    if include_llm_if_configured:
        candidate_config = load_llm_config(
            correction_mode="llm_with_rule_fallback"
        )
        if candidate_config.is_configured:
            candidate_config.validate(require_api=True)
            llm_config = candidate_config
            variants.append(LLM_VARIANT)
        else:
            print(
                "LLM configuration is not valid; optional real-LLM "
                "variant was skipped.",
                file=sys.stderr,
            )

    results: list[dict[str, Any]] = []
    candidate_records: list[dict[str, Any]] = []
    for case in cases:
        before = evaluate_text(
            str(case["reference_text"]),
            str(case["raw_asr_text"]),
            str(case["language"]),
        )
        for variant in variants:
            matches, rejected = _retrieve(case, entries, variant)
            corrected_text = str(case["raw_asr_text"])
            audit: dict[str, Any] = {
                "unsupported_changes": [],
                "needs_review": False,
                "api_used": False,
                "fallback_used": False,
                "llm_provider": "",
                "llm_model": "",
                "prompt_version": "",
            }
            correction_error = ""
            if variant == "fused_plus_rule_correction":
                corrected_text, audit = _correct(
                    case,
                    matches,
                    mode="rule_fallback",
                )
            elif variant == LLM_VARIANT:
                try:
                    corrected_text, audit = _correct(
                        case,
                        matches,
                        mode="llm",
                        runtime_config=llm_config,
                    )
                except RuntimeError as exc:
                    correction_error = str(exc)
                    audit.update(
                        {
                            "needs_review": True,
                            "api_used": True,
                            "fallback_used": False,
                            "llm_provider": llm_config.provider,
                            "llm_model": llm_config.model,
                            "prompt_version": llm_config.prompt_version,
                        }
                    )

            after = evaluate_text(
                str(case["reference_text"]),
                corrected_text,
                str(case["language"]),
            )
            (
                true_positive,
                false_positive,
                missed,
                precision,
                recall,
                f1,
            ) = _term_metrics(list(case["expected_terms"]), matches)
            rejected_review = bool(rejected)
            needs_review = bool(audit["needs_review"]) or rejected_review
            notes = [
                "Controlled text fixture; not public audio or measured ASR.",
                f"Retrieval variant={variant}.",
            ]
            if rejected:
                notes.append(
                    "Ambiguous candidates were withheld by context safety "
                    "and require review."
                )
            if variant == LLM_VARIANT:
                notes.append("Strict real LLM mode; no silent rule fallback.")
                if correction_error:
                    notes.append(
                        "API output failed grounding validation; raw text "
                        "was retained."
                    )
            result = {
                "case_id": case["case_id"],
                "language": case["language"],
                "difficulty": case["difficulty"],
                "variant": variant,
                "raw_asr_text": case["raw_asr_text"],
                "reference_text": case["reference_text"],
                "retrieved_candidates": _json(
                    [match.canonical for match in matches]
                ),
                "applied_corrections": _json(
                    _applied_corrections(case, corrected_text)
                ),
                "corrected_text": corrected_text,
                "expected_terms": _json(case["expected_terms"]),
                "true_positive_terms": _json(true_positive),
                "false_positive_terms": _json(false_positive),
                "missed_terms": _json(missed),
                "unsupported_changes": _json(
                    audit["unsupported_changes"]
                ),
                "needs_review": needs_review,
                "api_used": bool(audit["api_used"]),
                "fallback_used": bool(audit["fallback_used"]),
                "llm_provider": audit["llm_provider"],
                "llm_model": audit["llm_model"],
                "prompt_version": audit["prompt_version"],
                "term_precision": round(precision, 6),
                "term_recall": round(recall, 6),
                "term_f1": round(f1, 6),
                "text_error_before": round(float(before["error_rate"]), 6),
                "text_error_after": round(float(after["error_rate"]), 6),
                "metric_name": before["metric_name"],
                "correction_error": correction_error,
                "notes": " ".join(notes),
            }
            results.append(result)
            candidate_records.append(
                {
                    "case_id": case["case_id"],
                    "variant": variant,
                    "fixture_type": "controlled_technical_term",
                    "accepted": [asdict(match) for match in matches],
                    "rejected_for_context": [
                        asdict(match) for match in rejected
                    ],
                }
            )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=OUTPUT_COLUMNS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(results)
    candidates_output = Path(candidates_output_path)
    candidates_output.parent.mkdir(parents=True, exist_ok=True)
    candidates_output.write_text(
        "\n".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True)
            for record in candidate_records
        )
        + "\n",
        encoding="utf-8",
    )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--terms", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidates-output", type=Path, required=True)
    parser.add_argument(
        "--include-llm-if-configured",
        action="store_true",
        help="Add strict real-LLM rows only when .env is valid.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        rows = run_experiment(
            cases_path=args.cases,
            terms_path=args.terms,
            output_path=args.output,
            candidates_output_path=args.candidates_output,
            include_llm_if_configured=args.include_llm_if_configured,
        )
    except (FileNotFoundError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"Controlled term rescue experiment failed: {exc}", file=sys.stderr)
        return 2
    variants = sorted({str(row["variant"]) for row in rows})
    print(
        f"Wrote {len(rows)} controlled rows across {len(variants)} variants: "
        f"{args.output}"
    )
    print(f"Variants: {', '.join(variants)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
