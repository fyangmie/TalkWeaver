#!/usr/bin/env python3
"""Compute word error rate from reference and hypothesis text."""

from __future__ import annotations

import argparse


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Compute token-level Levenshtein distance divided by reference length."""

    ref = reference.lower().split()
    hyp = hypothesis.lower().split()
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference")
    parser.add_argument("--hypothesis")
    args = parser.parse_args()
    if args.reference is None or args.hypothesis is None:
        parser.error("Provide --reference and --hypothesis text.")
    print(f"WER={word_error_rate(args.reference, args.hypothesis):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
