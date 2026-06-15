#!/usr/bin/env python3
"""Run a real optional LLM on the binary safe-to-apply benchmark."""

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
DECISIONS = ("safe_to_apply", "do_not_apply")
OUTPUT_COLUMNS = [
    "proposal_id",
    "mode",
    "provider",
    "model",
    "decision",
    "rationale",
    "api_used",
    "latency_seconds",
    "prompt_version",
]


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
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


def build_binary_judge_messages(
    proposal: dict[str, Any],
    mode: str,
) -> list[dict[str, str]]:
    """Build a binary correction-safety prompt without reference leakage."""

    if mode not in MODES:
        raise ValueError(f"mode must be one of: {', '.join(MODES)}")
    evidence = ""
    if mode == "with_evidence":
        evidence = "\n".join(
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
    content = "\n".join(
        part
        for part in (
            f"Raw ASR text: {proposal.get('raw_asr_text', '')}",
            "Proposed corrected text: "
            + str(proposal.get("proposed_corrected_text", "")),
            evidence,
            (
                "Choose exactly one decision: safe_to_apply or do_not_apply. "
                "Use safe_to_apply only when the proposed edit is clearly "
                "supported and does not invent content or change speaker "
                "ownership. If evidence is insufficient, choose do_not_apply. "
                "Return JSON only with keys decision and rationale."
            ),
        )
        if part
    )
    return [
        {
            "role": "system",
            "content": (
                "You are a conservative binary safety judge for ASR "
                "correction proposals. Do not rewrite the text and do not "
                "assume missing evidence."
            ),
        },
        {"role": "user", "content": content},
    ]


def _normalize_response(payload: dict[str, Any]) -> tuple[str, str]:
    decision = str(payload.get("decision", "")).strip().casefold()
    decision = decision.replace("-", "_").replace(" ", "_")
    if decision not in DECISIONS:
        raise ValueError(
            "LLM returned an invalid binary decision; expected "
            f"{DECISIONS}, received {decision!r}."
        )
    rationale = str(payload.get("rationale", "")).strip()
    if not rationale:
        raise ValueError("LLM returned an empty rationale.")
    return decision, rationale


def run_binary_self_judge(
    *,
    input_path: str | Path,
    mode: str,
    output_path: str | Path,
    config: LLMConfig,
    append: bool = False,
    max_proposals: int | None = None,
    retries: int = 3,
    request_fn: Callable[
        [LLMConfig, list[dict[str, str]]],
        dict[str, Any],
    ] = request_json_completion,
) -> list[dict[str, Any]]:
    """Run real API judgments with retries and resumable checkpoints."""

    config.validate(require_api=True)
    if retries < 0:
        raise ValueError("retries must be non-negative.")
    with Path(input_path).open(encoding="utf-8", newline="") as handle:
        proposals = list(csv.DictReader(handle))
    if max_proposals is not None:
        proposals = proposals[:max_proposals]
    if not proposals:
        raise ValueError("Binary benchmark contains no proposals.")

    output = Path(output_path)
    existing = []
    if append and output.is_file():
        with output.open(encoding="utf-8", newline="") as handle:
            existing = list(csv.DictReader(handle))
    completed_keys = {
        (str(row.get("proposal_id")), str(row.get("mode")))
        for row in existing
    }
    proposals = [
        proposal
        for proposal in proposals
        if (str(proposal["proposal_id"]), mode) not in completed_keys
    ]
    if not proposals:
        return []

    prompt_version = f"talkweaver.binary_self_judge.{mode}.v1"
    new_rows = []
    for index, proposal in enumerate(proposals, start=1):
        started = time.perf_counter()
        response = None
        for attempt in range(retries + 1):
            try:
                response = request_fn(
                    config,
                    build_binary_judge_messages(proposal, mode),
                )
                break
            except RuntimeError:
                if attempt >= retries:
                    _write_predictions(output, existing + new_rows)
                    raise
                time.sleep(min(2 ** attempt, 8))
        if response is None:
            raise RuntimeError("LLM request produced no response.")
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
                "latency_seconds": round(latency, 4),
                "prompt_version": prompt_version,
            }
        )
        if index % 20 == 0 or index == len(proposals):
            print(
                f"Judged {index}/{len(proposals)} proposals ({mode}).",
                flush=True,
            )
        if index % 10 == 0:
            _write_predictions(output, existing + new_rows)
    _write_predictions(output, existing + new_rows)
    return new_rows


def _write_predictions(
    output: Path,
    rows: list[dict[str, Any]],
) -> None:
    deduplicated = {
        (str(row["proposal_id"]), str(row["mode"])): row for row in rows
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=OUTPUT_COLUMNS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(deduplicated.values())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a configured real LLM as a binary safe-to-apply correction "
            "judge. No rule fallback is used."
        )
    )
    parser.add_argument(
        "--input",
        default="data/pilot/binary_safe_apply_benchmark.csv",
    )
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument(
        "--output",
        default=(
            "experiments/results/binary_safe_apply/"
            "llm_self_judge_binary_predictions.csv"
        ),
    )
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--max-proposals", type=int)
    parser.add_argument("--retries", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_llm_config(correction_mode="llm")
        rows = run_binary_self_judge(
            input_path=args.input,
            mode=args.mode,
            output_path=args.output,
            config=config,
            append=args.append,
            max_proposals=args.max_proposals,
            retries=args.retries,
        )
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"Binary LLM self-judge unavailable: {exc}", file=sys.stderr)
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
