#!/usr/bin/env python3
"""Compare abstention policies on the selective-correction pilot."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.eccogate import score_correction_proposal  # noqa: E402


LABELS = ("accept", "reject", "needs_review")
RESULT_COLUMNS = [
    "proposal_id",
    "category",
    "language",
    "method",
    "gold_label",
    "label_source",
    "predicted_label",
    "correct",
    "unsafe_accept",
    "covered",
    "support_score",
    "risk_score",
    "explanation",
    "provider",
    "model",
    "api_used",
    "fallback_used",
]
SUMMARY_COLUMNS = [
    "method",
    "num_examples",
    "label_source",
    "macro_f1",
    "unsafe_accept_rate",
    "needs_review_recall",
    "accept_precision",
    "reject_recall",
    "coverage",
    "per_category_unsafe_accept_rate",
    "provider",
    "model",
    "api_used_count",
    "fallback_used_count",
]


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().casefold() in {"1", "true", "yes", "y"}


def _gold_label(row: dict[str, Any]) -> tuple[str, str]:
    checked = str(row.get("human_checked_label", "")).strip().casefold()
    if checked:
        if checked not in LABELS:
            raise ValueError(
                f"Invalid human_checked_label for {row.get('proposal_id')}: "
                f"{checked}"
            )
        return checked, "human_checked"
    suggested = str(row.get("suggested_gold_label", "")).strip().casefold()
    if suggested not in LABELS:
        raise ValueError(
            f"Invalid suggested_gold_label for {row.get('proposal_id')}: "
            f"{suggested}"
        )
    return suggested, "pilot_auto_labeled"


def compute_selective_metrics(
    true_labels: Iterable[str],
    predicted_labels: Iterable[str],
) -> dict[str, float]:
    true = list(true_labels)
    predicted = list(predicted_labels)
    if len(true) != len(predicted):
        raise ValueError("True and predicted label lengths do not match.")
    class_f1: list[float] = []
    for label in LABELS:
        true_positive = sum(
            actual == label and guess == label
            for actual, guess in zip(true, predicted)
        )
        false_positive = sum(
            actual != label and guess == label
            for actual, guess in zip(true, predicted)
        )
        false_negative = sum(
            actual == label and guess != label
            for actual, guess in zip(true, predicted)
        )
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        class_f1.append(
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )

    unsafe_pool = sum(label in {"reject", "needs_review"} for label in true)
    unsafe_accepts = sum(
        actual in {"reject", "needs_review"} and guess == "accept"
        for actual, guess in zip(true, predicted)
    )
    review_total = sum(label == "needs_review" for label in true)
    review_hits = sum(
        actual == "needs_review" and guess == "needs_review"
        for actual, guess in zip(true, predicted)
    )
    accept_predictions = sum(label == "accept" for label in predicted)
    safe_accepts = sum(
        actual == "accept" and guess == "accept"
        for actual, guess in zip(true, predicted)
    )
    reject_total = sum(label == "reject" for label in true)
    reject_hits = sum(
        actual == "reject" and guess == "reject"
        for actual, guess in zip(true, predicted)
    )
    return {
        "macro_f1": sum(class_f1) / len(class_f1),
        "unsafe_accept_rate": (
            unsafe_accepts / unsafe_pool if unsafe_pool else 0.0
        ),
        "needs_review_recall": (
            review_hits / review_total if review_total else 0.0
        ),
        "accept_precision": (
            safe_accepts / accept_predictions if accept_predictions else 0.0
        ),
        "reject_recall": reject_hits / reject_total if reject_total else 0.0,
        "coverage": (
            sum(label != "needs_review" for label in predicted) / len(predicted)
            if predicted
            else 0.0
        ),
    }


def _load_llm_predictions(path: str | Path | None) -> list[dict[str, Any]]:
    if not path or not Path(path).is_file():
        return []
    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    deduplicated: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        decision = str(row.get("decision", "")).strip().casefold()
        if decision not in LABELS:
            continue
        deduplicated[
            (str(row.get("proposal_id")), str(row.get("mode")))
        ] = row
    return list(deduplicated.values())


def evaluate_pilot(
    pilot_rows: list[dict[str, Any]],
    llm_rows: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Evaluate transparent baselines, EccoGate, and available LLM judges."""

    methods: dict[str, dict[str, dict[str, Any]]] = {
        "always_accept": {},
        "always_review": {},
        "EccoGate": {},
    }
    for row in pilot_rows:
        proposal_id = str(row["proposal_id"])
        methods["always_accept"][proposal_id] = {
            "decision": "accept",
            "explanation": "Baseline always accepts.",
        }
        methods["always_review"][proposal_id] = {
            "decision": "needs_review",
            "explanation": "Baseline always abstains for review.",
        }
        methods["EccoGate"][proposal_id] = score_correction_proposal(row).to_dict()

    for row in llm_rows or []:
        mode = str(row.get("mode"))
        method = f"llm_self_judge_{mode}"
        methods.setdefault(method, {})[str(row["proposal_id"])] = {
            "decision": row["decision"],
            "explanation": row.get("rationale", ""),
            "provider": row.get("provider", ""),
            "model": row.get("model", ""),
            "api_used": row.get("api_used", ""),
            "fallback_used": row.get("fallback_used", ""),
        }

    result_rows: list[dict[str, Any]] = []
    for method, predictions in methods.items():
        for proposal in pilot_rows:
            proposal_id = str(proposal["proposal_id"])
            if proposal_id not in predictions:
                continue
            prediction = predictions[proposal_id]
            gold, label_source = _gold_label(proposal)
            predicted = str(prediction["decision"])
            result_rows.append(
                {
                    "proposal_id": proposal_id,
                    "category": proposal["category"],
                    "language": proposal["language"],
                    "method": method,
                    "gold_label": gold,
                    "label_source": label_source,
                    "predicted_label": predicted,
                    "correct": predicted == gold,
                    "unsafe_accept": (
                        predicted == "accept" and gold != "accept"
                    ),
                    "covered": predicted != "needs_review",
                    "support_score": prediction.get("support_score", ""),
                    "risk_score": prediction.get("risk_score", ""),
                    "explanation": prediction.get("explanation", ""),
                    "provider": prediction.get("provider", ""),
                    "model": prediction.get("model", ""),
                    "api_used": prediction.get("api_used", False),
                    "fallback_used": prediction.get("fallback_used", False),
                }
            )

    summary_rows: list[dict[str, Any]] = []
    for method in methods:
        scoped = [row for row in result_rows if row["method"] == method]
        if not scoped:
            continue
        metrics = compute_selective_metrics(
            [str(row["gold_label"]) for row in scoped],
            [str(row["predicted_label"]) for row in scoped],
        )
        category_rates = {}
        for category in sorted({str(row["category"]) for row in scoped}):
            category_rows = [
                row for row in scoped if row["category"] == category
            ]
            unsafe_pool = [
                row for row in category_rows if row["gold_label"] != "accept"
            ]
            category_rates[category] = (
                sum(_as_bool(row["unsafe_accept"]) for row in unsafe_pool)
                / len(unsafe_pool)
                if unsafe_pool
                else 0.0
            )
        source_counts = Counter(str(row["label_source"]) for row in scoped)
        summary_rows.append(
            {
                "method": method,
                "num_examples": len(scoped),
                "label_source": (
                    next(iter(source_counts))
                    if len(source_counts) == 1
                    else json.dumps(source_counts, sort_keys=True)
                ),
                **{key: round(value, 6) for key, value in metrics.items()},
                "per_category_unsafe_accept_rate": json.dumps(
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
                "fallback_used_count": sum(
                    _as_bool(row["fallback_used"]) for row in scoped
                ),
            }
        )
    return result_rows, summary_rows


def _write_csv(
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
        description="Evaluate selective correction and abstention pilot methods."
    )
    parser.add_argument(
        "--input",
        default="data/pilot/selective_correction_pilot.csv",
    )
    parser.add_argument(
        "--llm-predictions",
        default=(
            "experiments/results/pilot/"
            "llm_self_judge_pilot_predictions.csv"
        ),
    )
    parser.add_argument(
        "--output",
        default=(
            "experiments/results/pilot/"
            "selective_correction_pilot_results.csv"
        ),
    )
    parser.add_argument(
        "--summary-output",
        default=(
            "experiments/results/pilot/"
            "selective_correction_pilot_summary.csv"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with Path(args.input).open(encoding="utf-8", newline="") as handle:
        pilot_rows = list(csv.DictReader(handle))
    llm_rows = _load_llm_predictions(args.llm_predictions)
    results, summary = evaluate_pilot(pilot_rows, llm_rows)
    _write_csv(args.output, results, RESULT_COLUMNS)
    _write_csv(args.summary_output, summary, SUMMARY_COLUMNS)
    for row in summary:
        print(
            f"{row['method']}: macro_f1={row['macro_f1']:.3f}, "
            f"unsafe_accept={row['unsafe_accept_rate']:.3f}, "
            f"review_recall={row['needs_review_recall']:.3f}, "
            f"coverage={row['coverage']:.3f}"
        )
    if not llm_rows:
        print(
            "LLM predictions were not found; baselines and EccoGate were "
            "evaluated only."
        )
    print(f"Results: {args.output}")
    print(f"Summary: {args.summary_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
