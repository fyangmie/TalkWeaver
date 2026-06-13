#!/usr/bin/env python3
"""Run controlled overlap-aware correction safety variants."""

from __future__ import annotations

import argparse
import csv
import json
import sys
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
from experiments.metrics.correction_safety_metrics import (  # noqa: E402
    applied_changes,
    evaluate_correction_safety,
)
from experiments.metrics.text_metrics import evaluate_text  # noqa: E402


RULE_VARIANTS = (
    "no_overlap_awareness_rule",
    "overlap_aware_rule",
)
LLM_VARIANTS = (
    "no_overlap_awareness_llm",
    "overlap_aware_llm",
)
OUTPUT_COLUMNS = [
    "case_id",
    "variant",
    "language",
    "overlap",
    "uncertainty_level",
    "raw_asr_text",
    "reference_text",
    "corrected_text",
    "expected_safe_behavior",
    "forbidden_changes",
    "applied_changes",
    "unsupported_changes",
    "invented_content",
    "forbidden_change_count",
    "speaker_attribution_changed",
    "needs_review",
    "review_flag_accuracy",
    "correction_allowed",
    "correction_rejected",
    "overcorrection",
    "conservative_rejection",
    "api_used",
    "fallback_used",
    "llm_provider",
    "llm_model",
    "prompt_version",
    "text_error_before",
    "text_error_after",
    "metric_name",
    "safety_pass",
    "notes",
]


def load_cases(path: str | Path) -> list[dict[str, Any]]:
    """Load controlled overlap JSONL fixtures with required evidence fields."""

    required = {
        "case_id",
        "language",
        "raw_asr_text",
        "reference_text",
        "speakers",
        "overlap",
        "overlap_span",
        "uncertainty_level",
        "expected_safe_behavior",
        "forbidden_changes",
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
        if not isinstance(payload["expected_safe_behavior"], dict):
            raise ValueError(
                f"Case line {line_number} expected_safe_behavior must be an object."
            )
        payload["fixture_type"] = "controlled_overlap_correction"
        cases.append(payload)
    if not cases:
        raise ValueError("Controlled overlap fixture contains no cases.")
    return cases


def load_safety_policy(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate the controlled correction policy."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "safe_correction_rules",
        "forbidden_hallucination_patterns",
        "overlap_conservative_rules",
        "speaker_attribution_rules",
        "when_to_mark_needs_review",
        "when_to_reject_llm_output",
    }
    if not isinstance(payload, dict):
        raise ValueError("Overlap safety policy must be a JSON object.")
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(
            "Overlap safety policy is missing: " + ", ".join(missing)
        )
    return payload


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _uncertainty_confidence(level: str) -> float:
    return {
        "low": 0.92,
        "medium": 0.62,
        "high": 0.35,
    }.get(level.casefold(), 0.5)


def _is_incomplete(text: str, policy: dict[str, Any]) -> bool:
    rules = policy["overlap_conservative_rules"]
    markers = [str(value) for value in rules.get("incomplete_markers", [])]
    return any(marker in text for marker in markers)


def _requires_conservative_rejection(
    case: dict[str, Any],
    policy: dict[str, Any],
) -> bool:
    rules = policy["overlap_conservative_rules"]
    reject_levels = {
        str(value).casefold()
        for value in rules.get("reject_uncertainty_levels", [])
    }
    uncertainty = str(case["uncertainty_level"]).casefold()
    if uncertainty in reject_levels:
        return True
    return bool(
        case["overlap"]
        and rules.get("reject_incomplete_overlap", True)
        and _is_incomplete(str(case["raw_asr_text"]), policy)
    )


def _needs_review(
    case: dict[str, Any],
    *,
    overlap_aware: bool,
    rejected: bool,
    audit_review: bool,
) -> bool:
    if audit_review or rejected:
        return True
    if not overlap_aware:
        return False
    return bool(
        case["overlap"]
        or str(case["uncertainty_level"]).casefold() != "low"
        or len(case["speakers"]) != 1
        or "..." in str(case["raw_asr_text"])
        or "[inaudible]" in str(case["raw_asr_text"])
        or "[cut off]" in str(case["raw_asr_text"])
    )


def _retrieve_matches(
    case: dict[str, Any],
    entries: list[GlossaryEntry],
) -> tuple[list[TermMatch], list[TermMatch]]:
    return retrieve_controlled_matches(
        str(case["raw_asr_text"]),
        entries,
        strategy="fused",
        context=str(case["context"]),
    )


def _run_correction(
    case: dict[str, Any],
    matches: list[TermMatch],
    *,
    overlap_aware: bool,
    mode: str,
    runtime_config: Any = None,
) -> tuple[str, dict[str, Any]]:
    speakers = [str(value) for value in case["speakers"]]
    overlap = bool(case["overlap"]) if overlap_aware else False
    speaker = (
        "OVERLAP"
        if overlap and len(speakers) > 1
        else (speakers[0] if speakers else "UNKNOWN")
    )
    anchor = TemporalAnchor(
        anchor_id=f"{case['case_id']}_anchor_001",
        clip_id=str(case["case_id"]),
        start=0.0,
        end=max(1.0, len(str(case["raw_asr_text"]).split()) * 0.4),
        speaker=speaker,
        speakers=speakers if overlap_aware else speakers[:1],
        raw_text=str(case["raw_asr_text"]),
        language=str(case["language"]),
        overlap=overlap,
        confidence=_uncertainty_confidence(
            str(case["uncertainty_level"])
        ),
        asr_confidence=_uncertainty_confidence(
            str(case["uncertainty_level"])
        ),
        diarization_confidence=0.6 if overlap else 0.9,
        retrieved_terms=[match.canonical for match in matches],
    )
    candidates = matches_to_candidates(matches, anchor_id=anchor.anchor_id)
    anchors, audits, _correction_mode = apply_constrained_correction(
        [anchor],
        candidates,
        [],
        llm_config={
            "correction_mode": mode,
            "runtime_config": runtime_config,
        },
    )
    return (
        anchors[0].corrected_text or anchors[0].raw_text,
        audits[0].to_dict(),
    )


def _evaluate_variant(
    case: dict[str, Any],
    policy: dict[str, Any],
    entries: list[GlossaryEntry],
    variant: str,
    llm_config: Any = None,
) -> dict[str, Any]:
    overlap_aware = variant.startswith("overlap_aware")
    use_llm = variant.endswith("_llm")
    matches, rejected_matches = _retrieve_matches(case, entries)
    proactive_rejection = bool(
        overlap_aware
        and _requires_conservative_rejection(case, policy)
    )
    correction_allowed = not proactive_rejection
    correction_rejected = proactive_rejection
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
    if correction_allowed:
        try:
            corrected_text, audit = _run_correction(
                case,
                matches,
                overlap_aware=overlap_aware,
                mode="llm" if use_llm else "rule_fallback",
                runtime_config=llm_config,
            )
        except RuntimeError as exc:
            correction_error = str(exc)
            correction_rejected = True
            audit.update(
                {
                    "needs_review": True,
                    "api_used": use_llm,
                    "fallback_used": False,
                    "llm_provider": (
                        llm_config.provider if llm_config else ""
                    ),
                    "llm_model": llm_config.model if llm_config else "",
                    "prompt_version": (
                        llm_config.prompt_version if llm_config else ""
                    ),
                }
            )
    if audit.get("unsupported_changes"):
        correction_rejected = True
    needs_review = _needs_review(
        case,
        overlap_aware=overlap_aware,
        rejected=correction_rejected,
        audit_review=bool(audit.get("needs_review")),
    )
    supported_evidence = [
        str(case["reference_text"]),
        *[match.canonical for match in matches],
    ]
    safety = evaluate_correction_safety(
        raw_text=str(case["raw_asr_text"]),
        corrected_text=corrected_text,
        supported_evidence=supported_evidence,
        forbidden_changes=[
            *policy.get("forbidden_hallucination_patterns", []),
            *case["forbidden_changes"],
        ],
        original_speakers=case["speakers"],
        corrected_speakers=case["speakers"],
        expected_safe_behavior=case["expected_safe_behavior"],
        needs_review=needs_review,
        correction_rejected=correction_rejected,
    )
    before = evaluate_text(
        str(case["reference_text"]),
        str(case["raw_asr_text"]),
        str(case["language"]),
    )
    after = evaluate_text(
        str(case["reference_text"]),
        corrected_text,
        str(case["language"]),
    )
    notes = [
        "Controlled overlap text fixture; not measured ASR output.",
        (
            "Overlap and uncertainty evidence enabled."
            if overlap_aware
            else "Overlap and uncertainty evidence withheld for ablation."
        ),
    ]
    if proactive_rejection:
        notes.append(
            "Policy rejected correction before model execution and retained raw text."
        )
    if rejected_matches:
        notes.append(
            "Context-rejected term candidates were not supplied to correction."
        )
    if correction_error:
        notes.append(
            f"Strict correction output rejected: {correction_error}"
        )
    return {
        "case_id": case["case_id"],
        "variant": variant,
        "language": case["language"],
        "overlap": bool(case["overlap"]),
        "uncertainty_level": case["uncertainty_level"],
        "raw_asr_text": case["raw_asr_text"],
        "reference_text": case["reference_text"],
        "corrected_text": corrected_text,
        "expected_safe_behavior": _json(
            case["expected_safe_behavior"]
        ),
        "forbidden_changes": _json(case["forbidden_changes"]),
        "applied_changes": _json(
            applied_changes(case["raw_asr_text"], corrected_text)
        ),
        "unsupported_changes": _json(safety["unsupported_changes"]),
        "invented_content": safety["invented_content"],
        "forbidden_change_count": safety["forbidden_change_count"],
        "speaker_attribution_changed": safety[
            "speaker_attribution_changed"
        ],
        "needs_review": needs_review,
        "review_flag_accuracy": safety["review_flag_accuracy"],
        "correction_allowed": correction_allowed,
        "correction_rejected": correction_rejected,
        "overcorrection": safety["overcorrection"],
        "conservative_rejection": safety["conservative_rejection"],
        "api_used": bool(audit.get("api_used")),
        "fallback_used": bool(audit.get("fallback_used")),
        "llm_provider": audit.get("llm_provider", ""),
        "llm_model": audit.get("llm_model", ""),
        "prompt_version": audit.get("prompt_version", ""),
        "text_error_before": round(float(before["error_rate"]), 6),
        "text_error_after": round(float(after["error_rate"]), 6),
        "metric_name": before["metric_name"],
        "safety_pass": safety["safety_pass"],
        "notes": " ".join(notes),
    }


def run_experiment(
    *,
    cases_path: str | Path,
    policy_path: str | Path,
    output_path: str | Path,
    include_llm_if_configured: bool = False,
) -> list[dict[str, Any]]:
    """Run rule variants and optional strict real-LLM variants."""

    cases = load_cases(cases_path)
    policy = load_safety_policy(policy_path)
    glossary_path = Path(str(policy.get("term_glossary_path", "")))
    if not glossary_path.is_absolute():
        glossary_path = ROOT / glossary_path
    entries = load_reference_glossary(glossary_path)
    variants = list(RULE_VARIANTS)
    llm_config = None
    if include_llm_if_configured:
        candidate_config = load_llm_config(
            correction_mode="llm_with_rule_fallback"
        )
        if candidate_config.is_configured:
            candidate_config.validate(require_api=True)
            llm_config = candidate_config
            variants.extend(LLM_VARIANTS)
        else:
            print(
                "LLM configuration is not valid; optional overlap LLM "
                "variants were skipped.",
                file=sys.stderr,
            )

    results = [
        _evaluate_variant(
            case,
            policy,
            entries,
            variant,
            llm_config,
        )
        for case in cases
        for variant in variants
    ]
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
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
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
            policy_path=args.policy,
            output_path=args.output,
            include_llm_if_configured=args.include_llm_if_configured,
        )
    except (FileNotFoundError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(
            f"Controlled overlap safety experiment failed: {exc}",
            file=sys.stderr,
        )
        return 2
    variants = sorted({str(row["variant"]) for row in rows})
    print(
        f"Wrote {len(rows)} controlled overlap rows across "
        f"{len(variants)} variants: {args.output}"
    )
    print(f"Variants: {', '.join(variants)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
