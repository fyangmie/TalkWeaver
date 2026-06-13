#!/usr/bin/env python3
"""Run a safe, one-anchor LLM correction configuration smoke test."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.constrained_correction import (  # noqa: E402
    apply_constrained_correction,
)
from backend.llm_config import (  # noqa: E402
    CORRECTION_MODES,
    load_llm_config,
)
from backend.schemas import TemporalAnchor  # noqa: E402
from backend.term_rescue import retrieve_term_candidates  # noqa: E402


RAW_TEXT = (
    "we use piano note for speaker diary station and rack correction"
)


def run_smoke(mode: str) -> dict[str, object]:
    """Run one grounded correction and return printable safe metadata."""

    runtime_config = load_llm_config(correction_mode=mode)
    anchor = TemporalAnchor(
        anchor_id="llm_smoke_anchor_001",
        clip_id="llm_smoke",
        start=0.0,
        end=5.0,
        speaker="SPEAKER_00",
        speakers=["SPEAKER_00"],
        raw_text=RAW_TEXT,
        language="en",
        confidence=0.9,
        asr_confidence=0.9,
        diarization_confidence=0.9,
    )
    candidates = retrieve_term_candidates([anchor])
    anchors, audits, correction_mode = apply_constrained_correction(
        [anchor],
        candidates,
        [],
        llm_config={
            "correction_mode": mode,
            "runtime_config": runtime_config,
        },
    )
    audit = audits[0]
    return {
        "raw_text": anchor.raw_text,
        "corrected_text": anchors[0].corrected_text,
        "correction_mode": correction_mode,
        "api_used": audit.api_used,
        "fallback_used": audit.fallback_used,
        "unsupported_changes": audit.unsupported_changes,
        "needs_review": audit.needs_review,
        "provider": audit.llm_provider or runtime_config.provider,
        "model": audit.llm_model or runtime_config.model,
        "prompt_version": audit.prompt_version,
        "temperature": audit.temperature,
        "api_key": runtime_config.masked_api_key,
        "retrieved_terms": anchor.retrieved_terms,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=sorted(CORRECTION_MODES),
        default="rule_fallback",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = run_smoke(args.mode)
    except (RuntimeError, TypeError, ValueError) as exc:
        print(f"LLM correction smoke test failed: {exc}", file=sys.stderr)
        return 2
    for key in (
        "raw_text",
        "corrected_text",
        "correction_mode",
        "api_used",
        "fallback_used",
        "unsupported_changes",
        "needs_review",
        "provider",
        "model",
        "prompt_version",
        "temperature",
        "api_key",
        "retrieved_terms",
    ):
        print(f"{key}={result[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
