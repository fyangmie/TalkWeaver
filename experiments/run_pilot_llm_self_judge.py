#!/usr/bin/env python3
"""Run an optional real-LLM self-judge on pilot correction proposals."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.llm_config import LLMConfig, load_llm_config  # noqa: E402
from backend.llm_correction import request_json_completion  # noqa: E402


MODES = ("no_evidence", "with_evidence")
DECISIONS = ("accept", "reject", "needs_review")
OUTPUT_COLUMNS = [
    "proposal_id",
    "mode",
    "provider",
    "model",
    "decision",
    "rationale",
    "api_used",
    "fallback_used",
    "latency_seconds",
    "prompt_version",
]


def _as_bool(value: Any) -> bool:
    return str(value).strip().casefold() in {"1", "true", "yes", "y"}


def _parse_terms(value: Any) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return [part.strip() for part in text.split(",") if part.strip()]
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    return [str(parsed)]


def build_self_judge_messages(
    proposal: dict[str, Any],
    mode: str,
) -> list[dict[str, str]]:
    """Build a strict selective-correction decision prompt."""

    if mode not in MODES:
        raise ValueError(f"mode must be one of: {', '.join(MODES)}")
    evidence_lines = ""
    if mode == "with_evidence":
        evidence_lines = "\n".join(
            [
                f"Context: {proposal.get('context', '')}",
                "Retrieved terms: "
                + json.dumps(
                    _parse_terms(proposal.get("retrieved_terms")),
                    ensure_ascii=False,
                ),
                f"Overlap: {_as_bool(proposal.get('overlap_flag'))}",
                "Heavy overlap: "
                + str(_as_bool(proposal.get("heavy_overlap_flag"))),
                "Speaker ambiguity: "
                + str(_as_bool(proposal.get("speaker_ambiguity_flag"))),
                "Partial utterance: "
                + str(_as_bool(proposal.get("partial_utterance_flag"))),
            ]
        )
    user_content = "\n".join(
        part
        for part in (
            f"Raw ASR text: {proposal.get('raw_asr_text', '')}",
            "Proposed corrected text: "
            + str(proposal.get("proposed_corrected_text", "")),
            evidence_lines,
            (
                "Choose exactly one decision: accept, reject, or needs_review. "
                "Accept only if the edit is supported. Reject unsupported or "
                "speaker-changing edits. Use needs_review when evidence is "
                "ambiguous. Return JSON only with keys decision and rationale."
            ),
        )
        if part
    )
    return [
        {
            "role": "system",
            "content": (
                "You are an independent safety judge for ASR corrections. "
                "Do not improve or rewrite the transcript. Judge only the "
                "given proposal. Abstain with needs_review when uncertain."
            ),
        },
        {"role": "user", "content": user_content},
    ]


def _normalize_response(payload: dict[str, Any]) -> tuple[str, str]:
    decision = str(payload.get("decision", "")).strip().casefold()
    decision = decision.replace("-", "_").replace(" ", "_")
    if decision not in DECISIONS:
        raise ValueError(
            "LLM judge returned invalid decision; expected accept, reject, "
            f"or needs_review, received {decision!r}."
        )
    rationale = str(payload.get("rationale", "")).strip()
    if not rationale:
        raise ValueError("LLM judge returned an empty rationale.")
    return decision, rationale


def run_self_judge(
    *,
    input_path: str | Path,
    mode: str,
    output_path: str | Path,
    config: LLMConfig,
    append: bool = False,
    max_proposals: int | None = None,
    request_fn: Callable[
        [LLMConfig, list[dict[str, str]]],
        dict[str, Any],
    ] = request_json_completion,
) -> list[dict[str, Any]]:
    """Judge pilot proposals with a configured API and no fake fallback."""

    config.validate(require_api=True)
    with Path(input_path).open(encoding="utf-8", newline="") as handle:
        proposals = list(csv.DictReader(handle))
    if max_proposals is not None:
        proposals = proposals[:max_proposals]
    if not proposals:
        raise ValueError("Pilot input contains no proposals.")

    prompt_version = f"talkweaver.pilot_self_judge.{mode}.v1"
    new_rows: list[dict[str, Any]] = []
    for index, proposal in enumerate(proposals, start=1):
        started = time.perf_counter()
        response = request_fn(
            config,
            build_self_judge_messages(proposal, mode),
        )
        latency = time.perf_counter() - started
        decision, rationale = _normalize_response(response)
        new_rows.append(
            {
                "proposal_id": proposal["proposal_id"],
                "mode": mode,
                "provider": config.provider,
                "model": config.model,
                "decision": decision,
                "rationale": rationale,
                "api_used": True,
                "fallback_used": False,
                "latency_seconds": round(latency, 4),
                "prompt_version": prompt_version,
            }
        )
        if index % 10 == 0 or index == len(proposals):
            print(f"Judged {index}/{len(proposals)} proposals ({mode}).")

    output = Path(output_path)
    existing: list[dict[str, Any]] = []
    if append and output.is_file():
        with output.open(encoding="utf-8", newline="") as handle:
            existing = list(csv.DictReader(handle))
    replacement_keys = {
        (str(row["proposal_id"]), str(row["mode"])) for row in new_rows
    }
    combined = [
        row
        for row in existing
        if (str(row.get("proposal_id")), str(row.get("mode")))
        not in replacement_keys
    ] + new_rows
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=OUTPUT_COLUMNS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(combined)
    return new_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a real configured LLM as an accept/reject/review judge. "
            "No rule fallback is used."
        )
    )
    parser.add_argument(
        "--input",
        default="data/pilot/selective_correction_pilot.csv",
    )
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument(
        "--output",
        default=(
            "experiments/results/pilot/"
            "llm_self_judge_pilot_predictions.csv"
        ),
    )
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--max-proposals", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_llm_config(correction_mode="llm")
        rows = run_self_judge(
            input_path=args.input,
            mode=args.mode,
            output_path=args.output,
            config=config,
            append=args.append,
            max_proposals=args.max_proposals,
        )
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"LLM self-judge unavailable: {exc}", file=sys.stderr)
        print(
            "Setup: copy .env.example to .env, configure LLM_PROVIDER, "
            "LLM_API_KEY, LLM_MODEL, and LLM_BASE_URL, then rerun.",
            file=sys.stderr,
        )
        return 2
    print(
        f"Completed {len(rows)} real API judgments with "
        f"{config.provider}/{config.model}."
    )
    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
