#!/usr/bin/env python3
"""Evaluate binary safe-to-apply correction policies."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.eccogate_binary import score_binary_correction  # noqa: E402


LABELS = ("safe_to_apply", "do_not_apply")
RESULT_COLUMNS = [
    "proposal_id",
    "source",
    "category",
    "language",
    "method",
    "gold_label",
    "label_source",
    "predicted_label",
    "correct",
    "unsafe_apply",
    "false_block",
    "applied",
    "error_before",
    "error_after",
    "error_delta",
    "support_score",
    "risk_score",
    "explanation",
    "provider",
    "model",
    "api_used",
]
SUMMARY_COLUMNS = [
    "method",
    "num_examples",
    "label_sources",
    "accuracy",
    "macro_f1",
    "safe_apply_precision",
    "safe_apply_recall",
    "unsafe_apply_rate",
    "false_block_rate",
    "coverage",
    "error_delta_when_applied",
    "num_applied_with_reference",
    "per_category_unsafe_apply_rate",
    "provider",
    "model",
    "api_used_count",
]


def _clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return _clean(value).casefold() in {"1", "true", "yes", "y"}


def _as_float(value: Any) -> float | None:
    text = _clean(value)
    if not text:
        return None
    try:
        result = float(text)
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def _terms(value: Any) -> list[str]:
    text = _clean(value)
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = [part.strip() for part in text.split(",")]
    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if str(item).strip()]
    return [str(parsed).strip()] if str(parsed).strip() else []


def compute_binary_metrics(
    true_labels: Iterable[str],
    predicted_labels: Iterable[str],
) -> dict[str, float]:
    """Compute binary safety and application metrics."""

    true = list(true_labels)
    predicted = list(predicted_labels)
    if len(true) != len(predicted):
        raise ValueError("True and predicted labels have different lengths.")
    if not true:
        return {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "safe_apply_precision": 0.0,
            "safe_apply_recall": 0.0,
            "unsafe_apply_rate": 0.0,
            "false_block_rate": 0.0,
            "coverage": 0.0,
        }
    per_class_f1 = []
    for label in LABELS:
        tp = sum(
            actual == label and guess == label
            for actual, guess in zip(true, predicted)
        )
        fp = sum(
            actual != label and guess == label
            for actual, guess in zip(true, predicted)
        )
        fn = sum(
            actual == label and guess != label
            for actual, guess in zip(true, predicted)
        )
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        per_class_f1.append(
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
    safe_total = sum(label == "safe_to_apply" for label in true)
    blocked_total = sum(label == "do_not_apply" for label in true)
    applied_total = sum(label == "safe_to_apply" for label in predicted)
    safe_applied = sum(
        actual == "safe_to_apply" and guess == "safe_to_apply"
        for actual, guess in zip(true, predicted)
    )
    unsafe_applied = sum(
        actual == "do_not_apply" and guess == "safe_to_apply"
        for actual, guess in zip(true, predicted)
    )
    false_blocks = sum(
        actual == "safe_to_apply" and guess == "do_not_apply"
        for actual, guess in zip(true, predicted)
    )
    return {
        "accuracy": sum(a == b for a, b in zip(true, predicted)) / len(true),
        "macro_f1": sum(per_class_f1) / len(per_class_f1),
        "safe_apply_precision": (
            safe_applied / applied_total if applied_total else 0.0
        ),
        "safe_apply_recall": safe_applied / safe_total if safe_total else 0.0,
        "unsafe_apply_rate": (
            unsafe_applied / blocked_total if blocked_total else 0.0
        ),
        "false_block_rate": (
            false_blocks / safe_total if safe_total else 0.0
        ),
        "coverage": applied_total / len(true),
    }


def _retrieval_only(row: dict[str, Any]) -> dict[str, Any]:
    terms = _terms(row.get("retrieved_terms"))
    corrected = _clean(row.get("proposed_corrected_text")).casefold()
    raw = _clean(row.get("raw_asr_text")).casefold()
    supported = any(
        term.casefold() in corrected and term.casefold() not in raw
        for term in terms
    )
    return {
        "decision": "safe_to_apply" if supported else "do_not_apply",
        "explanation": (
            "Applied because a retrieved term appears only in the proposal."
            if supported
            else "Blocked because no new retrieved term supports the edit."
        ),
    }


def _binary_gate(
    row: dict[str, Any],
    *,
    overlap_aware: bool,
) -> dict[str, Any]:
    proposal = dict(row)
    if not overlap_aware:
        proposal.update(
            {
                "overlap_flag": False,
                "heavy_overlap_flag": False,
                "speaker_ambiguity_flag": False,
                "partial_utterance_flag": False,
            }
        )
    return score_binary_correction(proposal).to_dict()


def _load_llm_predictions(path: str | Path | None) -> list[dict[str, Any]]:
    if not path or not Path(path).is_file():
        return []
    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    deduplicated = {}
    for row in rows:
        decision = _clean(row.get("decision")).casefold()
        if decision not in LABELS:
            continue
        deduplicated[
            (_clean(row.get("proposal_id")), _clean(row.get("mode")))
        ] = row
    return list(deduplicated.values())


def evaluate_binary_benchmark(
    benchmark_rows: list[dict[str, Any]],
    llm_rows: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Evaluate deterministic policies and available real LLM predictions."""

    methods: dict[str, dict[str, dict[str, Any]]] = {
        "always_apply": {},
        "never_apply": {},
        "retrieval_only": {},
        "overlap_unaware_policy": {},
        "binary_eccogate": {},
    }
    for row in benchmark_rows:
        proposal_id = _clean(row["proposal_id"])
        methods["always_apply"][proposal_id] = {
            "decision": "safe_to_apply",
            "explanation": "Baseline always applies.",
        }
        methods["never_apply"][proposal_id] = {
            "decision": "do_not_apply",
            "explanation": "Baseline blocks every proposal.",
        }
        methods["retrieval_only"][proposal_id] = _retrieval_only(row)
        methods["overlap_unaware_policy"][proposal_id] = _binary_gate(
            row,
            overlap_aware=False,
        )
        methods["binary_eccogate"][proposal_id] = _binary_gate(
            row,
            overlap_aware=True,
        )
    for row in llm_rows or []:
        method = f"llm_self_judge_{_clean(row.get('mode'))}"
        methods.setdefault(method, {})[_clean(row["proposal_id"])] = {
            "decision": _clean(row["decision"]),
            "explanation": _clean(row.get("rationale")),
            "provider": _clean(row.get("provider")),
            "model": _clean(row.get("model")),
            "api_used": _as_bool(row.get("api_used")),
        }

    results = []
    for method, predictions in methods.items():
        for row in benchmark_rows:
            proposal_id = _clean(row["proposal_id"])
            if proposal_id not in predictions:
                continue
            prediction = predictions[proposal_id]
            gold = _clean(row["binary_label"])
            predicted = _clean(prediction["decision"])
            results.append(
                {
                    "proposal_id": proposal_id,
                    "source": row["source"],
                    "category": row["category"],
                    "language": row["language"],
                    "method": method,
                    "gold_label": gold,
                    "label_source": row["label_source"],
                    "predicted_label": predicted,
                    "correct": predicted == gold,
                    "unsafe_apply": (
                        predicted == "safe_to_apply"
                        and gold == "do_not_apply"
                    ),
                    "false_block": (
                        predicted == "do_not_apply"
                        and gold == "safe_to_apply"
                    ),
                    "applied": predicted == "safe_to_apply",
                    "error_before": row.get("error_before", ""),
                    "error_after": row.get("error_after", ""),
                    "error_delta": row.get("error_delta", ""),
                    "support_score": prediction.get("support_score", ""),
                    "risk_score": prediction.get("risk_score", ""),
                    "explanation": prediction.get("explanation", ""),
                    "provider": prediction.get("provider", ""),
                    "model": prediction.get("model", ""),
                    "api_used": prediction.get("api_used", False),
                }
            )

    summary = []
    for method in methods:
        scoped = [row for row in results if row["method"] == method]
        if not scoped:
            continue
        metrics = compute_binary_metrics(
            [str(row["gold_label"]) for row in scoped],
            [str(row["predicted_label"]) for row in scoped],
        )
        applied_deltas = [
            value
            for row in scoped
            if _as_bool(row["applied"])
            and (value := _as_float(row.get("error_delta"))) is not None
        ]
        category_rates = {}
        for category in sorted({str(row["category"]) for row in scoped}):
            blocked = [
                row
                for row in scoped
                if row["category"] == category
                and row["gold_label"] == "do_not_apply"
            ]
            category_rates[category] = (
                sum(_as_bool(row["unsafe_apply"]) for row in blocked)
                / len(blocked)
                if blocked
                else 0.0
            )
        label_sources = Counter(str(row["label_source"]) for row in scoped)
        summary.append(
            {
                "method": method,
                "num_examples": len(scoped),
                "label_sources": json.dumps(
                    label_sources,
                    sort_keys=True,
                ),
                **{key: round(value, 6) for key, value in metrics.items()},
                "error_delta_when_applied": (
                    round(sum(applied_deltas) / len(applied_deltas), 6)
                    if applied_deltas
                    else ""
                ),
                "num_applied_with_reference": len(applied_deltas),
                "per_category_unsafe_apply_rate": json.dumps(
                    category_rates,
                    sort_keys=True,
                ),
                "provider": next(
                    (str(row["provider"]) for row in scoped if row["provider"]),
                    "",
                ),
                "model": next(
                    (str(row["model"]) for row in scoped if row["model"]),
                    "",
                ),
                "api_used_count": sum(
                    _as_bool(row["api_used"]) for row in scoped
                ),
            }
        )
    return results, summary


def _write(
    path: str | Path,
    rows: list[dict[str, Any]],
    columns: list[str],
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate binary safe-to-apply correction policies."
    )
    parser.add_argument(
        "--input",
        default="data/pilot/binary_safe_apply_benchmark.csv",
    )
    parser.add_argument(
        "--llm-predictions",
        default=(
            "experiments/results/binary_safe_apply/"
            "llm_self_judge_binary_predictions.csv"
        ),
    )
    parser.add_argument(
        "--output",
        default=(
            "experiments/results/binary_safe_apply/"
            "binary_safe_apply_results.csv"
        ),
    )
    parser.add_argument(
        "--summary-output",
        default=(
            "experiments/results/binary_safe_apply/"
            "binary_safe_apply_summary.csv"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with Path(args.input).open(encoding="utf-8", newline="") as handle:
        benchmark = list(csv.DictReader(handle))
    llm_rows = _load_llm_predictions(args.llm_predictions)
    results, summary = evaluate_binary_benchmark(benchmark, llm_rows)
    _write(args.output, results, RESULT_COLUMNS)
    _write(args.summary_output, summary, SUMMARY_COLUMNS)
    for row in summary:
        print(
            f"{row['method']}: macro_f1={row['macro_f1']:.3f}, "
            f"unsafe_apply={row['unsafe_apply_rate']:.3f}, "
            f"false_block={row['false_block_rate']:.3f}, "
            f"coverage={row['coverage']:.3f}"
        )
    if not llm_rows:
        print(
            "Binary LLM predictions were not found; deterministic policies "
            "were evaluated only."
        )
    print(f"Results: {args.output}")
    print(f"Summary: {args.summary_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
