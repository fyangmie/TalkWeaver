#!/usr/bin/env python3
"""Compute WER from text or TalkWeaver JSON artifacts."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[._'-][A-Za-z0-9]+)*")


def normalize_text(text: str) -> str:
    """Apply a documented lowercase word-token normalization."""

    return " ".join(TOKEN_PATTERN.findall(text.lower()))


def _fallback_wer(reference: str, hypothesis: str) -> float:
    ref = reference.split()
    hyp = hypothesis.split()
    if not ref:
        return 0.0 if not hyp else 1.0

    previous = list(range(len(hyp) + 1))
    for row, ref_word in enumerate(ref, start=1):
        current = [row]
        for column, hyp_word in enumerate(hyp, start=1):
            substitution = previous[column - 1] + (ref_word != hyp_word)
            insertion = current[column - 1] + 1
            deletion = previous[column] + 1
            current.append(min(substitution, insertion, deletion))
        previous = current
    return previous[-1] / len(ref)


def word_error_rate(
    reference: str,
    hypothesis: str,
    *,
    prefer_jiwer: bool = True,
) -> float:
    """Compute normalized WER with an optional jiwer backend."""

    normalized_reference = normalize_text(reference)
    normalized_hypothesis = normalize_text(hypothesis)
    if prefer_jiwer:
        try:
            from jiwer import wer
        except ImportError:
            pass
        else:
            return float(wer(normalized_reference, normalized_hypothesis))
    return _fallback_wer(normalized_reference, normalized_hypothesis)


def extract_text(payload: Any, *, field: str | None = None) -> str:
    """Extract transcript text from common TalkWeaver JSON structures."""

    if isinstance(payload, str):
        return payload
    if isinstance(payload, list):
        parts: list[str] = []
        for item in payload:
            if not isinstance(item, dict):
                parts.append(str(item))
                continue
            candidates = (
                [field]
                if field
                else ["corrected_text", "text", "raw_text"]
            )
            value = next(
                (
                    item.get(candidate)
                    for candidate in candidates
                    if candidate and item.get(candidate)
                ),
                "",
            )
            parts.append(str(value))
        return " ".join(part for part in parts if part).strip()
    if isinstance(payload, dict):
        if field and field in payload and isinstance(payload[field], str):
            return str(payload[field])
        for key in (
            "transcript",
            "segments",
            "asr_segments",
            "temporal_transcript",
        ):
            if key in payload:
                return extract_text(payload[key], field=field)
        for key in ("corrected_text", "text", "raw_text", "reference"):
            if payload.get(key):
                return str(payload[key])
    raise ValueError("Unable to extract transcript text from the input.")


def read_text_input(value: str, *, field: str | None = None) -> str:
    """Read literal text, a plain-text file, or a JSON transcript."""

    path = Path(value)
    if not path.exists():
        return value
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return extract_text(payload, field=field)
    return path.read_text(encoding="utf-8").strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference",
        required=True,
        help="Reference text or path to TXT/JSON.",
    )
    parser.add_argument(
        "--hypothesis",
        required=True,
        help="Hypothesis text or path to TXT/JSON.",
    )
    parser.add_argument("--reference-field")
    parser.add_argument("--hypothesis-field")
    parser.add_argument(
        "--fallback",
        action="store_true",
        help="Use the built-in Levenshtein implementation even if jiwer exists.",
    )
    args = parser.parse_args()

    reference = read_text_input(args.reference, field=args.reference_field)
    hypothesis = read_text_input(args.hypothesis, field=args.hypothesis_field)
    score = word_error_rate(
        reference,
        hypothesis,
        prefer_jiwer=not args.fallback,
    )
    backend = "fallback" if args.fallback else "jiwer_or_fallback"
    print(f"WER={score:.4f}")
    print(f"Backend={backend}")
    print(f"ReferenceWords={len(normalize_text(reference).split())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
