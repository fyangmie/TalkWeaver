#!/usr/bin/env python3
"""Evaluate domain-term recovery before and after correction."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from experiments.evaluate_wer import read_text_input


DOMAIN_TERMS = (
    "pyannote.audio",
    "pyannote",
    "speaker diarization",
    "diarization",
    "overlapping speech",
    "cross-speech",
    "ASR",
    "WER",
    "DER",
    "WDER",
    "RAG",
    "faster-whisper",
    "VAD",
    "LLM correction",
    "temporal anchor",
)


def _contains_term(text: str, term: str) -> bool:
    escaped = re.escape(term.lower()).replace(r"\ ", r"\s+")
    pattern = rf"(?<![a-z0-9]){escaped}(?![a-z0-9])"
    return re.search(pattern, text.lower()) is not None


def find_domain_terms(
    text: str,
    domain_terms: tuple[str, ...] | list[str] = DOMAIN_TERMS,
) -> list[str]:
    """Return domain terms found in text, preserving glossary order."""

    return [term for term in domain_terms if _contains_term(text, term)]


def evaluate_term_error(
    reference: str,
    hypothesis: str,
    *,
    domain_terms: tuple[str, ...] | list[str] = DOMAIN_TERMS,
) -> dict[str, Any]:
    """Score required domain-term presence in a hypothesis."""

    required = find_domain_terms(reference, domain_terms)
    predicted = find_domain_terms(hypothesis, domain_terms)
    correct = [term for term in required if term in predicted]
    missing = [term for term in required if term not in predicted]
    unexpected = [term for term in predicted if term not in required]
    recall = len(correct) / len(required) if required else 1.0
    precision = len(correct) / len(predicted) if predicted else (1.0 if not required else 0.0)
    return {
        "term_error_rate": len(missing) / len(required) if required else 0.0,
        "precision": precision,
        "recall": recall,
        "required_terms": required,
        "predicted_terms": predicted,
        "correct_terms": correct,
        "missing_terms": missing,
        "unexpected_terms": unexpected,
    }


def term_error_rate(reference_terms: list[str], hypothesis: str) -> float:
    """Backward-compatible TER for an explicit required-term list."""

    if not reference_terms:
        return 0.0
    misses = sum(not _contains_term(hypothesis, term) for term in reference_terms)
    return misses / len(reference_terms)


def compare_term_recovery(
    reference: str,
    hypotheses: dict[str, str],
) -> list[dict[str, Any]]:
    """Evaluate named pipeline variants against the same reference."""

    return [
        {"pipeline": name, **evaluate_term_error(reference, hypothesis)}
        for name, hypothesis in hypotheses.items()
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--hypothesis")
    parser.add_argument("--whisper")
    parser.add_argument("--llm")
    parser.add_argument("--rag")
    parser.add_argument("--terms", nargs="+")
    args = parser.parse_args()

    reference = read_text_input(args.reference)
    variants = {
        name: read_text_input(value)
        for name, value in (
            ("Whisper only", args.whisper),
            ("LLM correction", args.llm),
            ("LLM correction + RAG glossary", args.rag),
            ("Hypothesis", args.hypothesis),
        )
        if value is not None
    }
    if not variants:
        parser.error("Provide --hypothesis or one of --whisper/--llm/--rag.")

    terms = tuple(args.terms) if args.terms else DOMAIN_TERMS
    for name, hypothesis in variants.items():
        result = evaluate_term_error(
            reference,
            hypothesis,
            domain_terms=terms,
        )
        print(
            f"{name}: TER={result['term_error_rate']:.4f} "
            f"Precision={result['precision']:.4f} "
            f"Recall={result['recall']:.4f}"
        )
        print(f"  Missing={', '.join(result['missing_terms']) or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
