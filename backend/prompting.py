"""Diarization-structured prompts for segment-level correction."""

from __future__ import annotations

from typing import Any


SYSTEM_PROMPT = """You correct one ASR transcript segment at a time.
Preserve the supplied timestamp and speaker attribution.
Use only evidence in the raw text and retrieved domain terms.
Correct punctuation and obvious ASR substitutions, but add no new facts.
Do not silently delete uncertain content.
For overlapping speech, make only high-confidence glossary substitutions and
mark the result uncertain.
Return JSON only with corrected_text, uncertain, and note."""


def format_timestamp(seconds: float) -> str:
    """Format seconds as MM:SS.ss."""

    minutes, remainder = divmod(float(seconds), 60)
    return f"{int(minutes):02d}:{remainder:05.2f}"


def format_segment_prompt(segment: dict[str, Any]) -> str:
    """Format one speaker-time segment for constrained correction."""

    terms = ", ".join(segment.get("retrieved_terms", [])) or "none"
    speakers = ", ".join(segment.get("speakers", [])) or "none"
    overlap = str(bool(segment.get("overlap"))).lower()
    uncertain = "true" if segment.get("overlap") else "false"
    uncertainty_rule = (
        "Overlap constraint: keep ambiguous wording, make only supported "
        "domain-term substitutions, and set uncertain=true."
        if segment.get("overlap")
        else (
            "Single-speaker constraint: make only corrections supported by "
            "the raw text or retrieved terms."
        )
    )
    return (
        f"[{format_timestamp(segment['start'])}-"
        f"{format_timestamp(segment['end'])}] "
        f"{segment['speaker']} | speakers={speakers} | overlap={overlap} | "
        f"confidence={float(segment['confidence']):.2f}\n"
        f"Raw: {segment['raw_text']}\n"
        f"Retrieved terms: {terms}\n"
        f"{uncertainty_rule}\n"
        "Output JSON: "
        f'{{"corrected_text":"...", "uncertain":{uncertain}, '
        '"note":"..."}}'
    )


def build_correction_messages(
    segment: dict[str, Any],
) -> list[dict[str, str]]:
    """Build an OpenAI-compatible system/user message list."""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": format_segment_prompt(segment)},
    ]
