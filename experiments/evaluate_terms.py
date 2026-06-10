#!/usr/bin/env python3
"""Evaluate domain-term recovery against an explicit reference list."""

from __future__ import annotations

import argparse


def term_error_rate(reference_terms: list[str], hypothesis: str) -> float:
    """Return the fraction of reference terms absent from the hypothesis."""

    if not reference_terms:
        return 0.0
    lowered = hypothesis.lower()
    misses = sum(term.lower() not in lowered for term in reference_terms)
    return misses / len(reference_terms)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terms", nargs="+")
    parser.add_argument("--hypothesis")
    args = parser.parse_args()
    if args.terms is None or args.hypothesis is None:
        parser.error("Provide --terms and --hypothesis.")
    print(f"TermErrorRate={term_error_rate(args.terms, args.hypothesis):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
