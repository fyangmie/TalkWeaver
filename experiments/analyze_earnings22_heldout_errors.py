#!/usr/bin/env python3
"""Write a compact error analysis for Earnings-22 held-out RAG corrections."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.term_rescue import load_reference_glossary  # noqa: E402
from experiments.metrics.text_normalization import normalize_for_wer  # noqa: E402


OUTPUT_COLUMNS = [
    "clip_id",
    "model_name",
    "source_text",
    "replacement_text",
    "canonical_term",
    "category",
    "decision",
    "wer_delta",
    "term_recall_delta",
    "reason",
]


def _json_list(value: str) -> list[Any]:
    try:
        payload = json.loads(value or "[]")
    except json.JSONDecodeError:
        return []
    return payload if isinstance(payload, list) else []


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _category_map(glossary_path: Path) -> dict[str, str]:
    entries = load_reference_glossary(glossary_path)
    return {
        normalize_for_wer(entry.canonical): entry.category
        for entry in entries
    }


def build_error_rows(
    rows: list[dict[str, str]],
    *,
    glossary_path: Path,
) -> list[dict[str, str]]:
    categories = _category_map(glossary_path)
    output: list[dict[str, str]] = []
    for row in rows:
        wer_delta = row.get("wer_delta", "")
        recall_delta = (
            f"{float(row['term_recall_after']) - float(row['term_recall_before']):.6f}"
            if row.get("term_recall_after") and row.get("term_recall_before")
            else ""
        )
        applied = _json_list(row.get("applied_corrections", "[]"))
        rejected = _json_list(row.get("rejected_corrections", "[]"))
        review = _json_list(row.get("needs_review_corrections", "[]"))
        no_op = _json_list(row.get("no_op_corrections", "[]"))
        if not applied and not rejected and not review and not no_op:
            output.append(
                {
                    "clip_id": row.get("clip_id", ""),
                    "model_name": row.get("model_name", ""),
                    "source_text": "",
                    "replacement_text": "",
                    "canonical_term": "",
                    "category": "no_action",
                    "decision": "unchanged",
                    "wer_delta": wer_delta,
                    "term_recall_delta": recall_delta,
                    "reason": (
                        "No safe glossary-grounded candidate was applied, "
                        "or baseline already covered the glossary terms."
                    ),
                }
            )
            continue
        for item in applied:
            if not isinstance(item, dict):
                continue
            canonical = str(item.get("canonical_term", ""))
            output.append(
                {
                    "clip_id": row.get("clip_id", ""),
                    "model_name": row.get("model_name", ""),
                    "source_text": str(item.get("source_text", "")),
                    "replacement_text": str(item.get("replacement_text", "")),
                    "canonical_term": canonical,
                    "category": categories.get(normalize_for_wer(canonical), "unknown"),
                    "decision": "applied",
                    "wer_delta": wer_delta,
                    "term_recall_delta": recall_delta,
                    "reason": str(item.get("reason", item.get("validation", ""))),
                }
            )
        for item in rejected:
            if not isinstance(item, dict):
                continue
            canonical = str(item.get("canonical_term", ""))
            output.append(
                {
                    "clip_id": row.get("clip_id", ""),
                    "model_name": row.get("model_name", ""),
                    "source_text": str(item.get("source_text", "")),
                    "replacement_text": str(item.get("replacement_text", "")),
                    "canonical_term": canonical,
                    "category": categories.get(normalize_for_wer(canonical), "unknown"),
                    "decision": "rejected",
                    "wer_delta": wer_delta,
                    "term_recall_delta": recall_delta,
                    "reason": str(item.get("reason", item.get("validation", ""))),
                }
            )
        for item in review:
            if not isinstance(item, dict):
                continue
            canonical = str(item.get("candidate_term", ""))
            output.append(
                {
                    "clip_id": row.get("clip_id", ""),
                    "model_name": row.get("model_name", ""),
                    "source_text": str(item.get("source_text", "")),
                    "replacement_text": "",
                    "canonical_term": canonical,
                    "category": "needs_review",
                    "decision": "needs_review",
                    "wer_delta": wer_delta,
                    "term_recall_delta": recall_delta,
                    "reason": str(item.get("reason", "")),
                }
            )
        for item in no_op:
            if not isinstance(item, dict):
                continue
            canonical = str(
                item.get("canonical_term", item.get("candidate_term", ""))
            )
            output.append(
                {
                    "clip_id": row.get("clip_id", ""),
                    "model_name": row.get("model_name", ""),
                    "source_text": str(item.get("source_text", "")),
                    "replacement_text": str(item.get("replacement_text", "")),
                    "canonical_term": canonical,
                    "category": categories.get(normalize_for_wer(canonical), "no_op"),
                    "decision": "no_op",
                    "wer_delta": wer_delta,
                    "term_recall_delta": recall_delta,
                    "reason": str(item.get("reason", item.get("validation", ""))),
                }
            )
    return output


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _markdown_table(rows: list[dict[str, str]], columns: list[str]) -> str:
    if not rows:
        return "_No rows._\n"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row.get(column, "").replace("|", "\\|") for column in columns) + " |")
    return "\n".join(lines) + "\n"


def write_markdown(path: Path, rows: list[dict[str, str]]) -> None:
    by_category = Counter(row["category"] for row in rows if row["decision"] == "applied")
    by_decision = Counter(row["decision"] for row in rows)
    summary_rows = [
        {"name": key, "count": str(value)}
        for key, value in sorted(by_decision.items())
    ]
    category_rows = [
        {"name": key, "count": str(value)}
        for key, value in sorted(by_category.items())
    ]
    examples = [
        row
        for row in rows
        if row["decision"] in {"applied", "rejected", "needs_review", "no_op"}
    ][:20]
    content = [
        "# Earnings-22 Held-Out Error Analysis",
        "",
        "This report summarizes the correction decisions on the frozen held-out subset. Reference text is used only for scoring and analysis.",
        "",
        "## Decision Counts",
        "",
        _markdown_table(summary_rows, ["name", "count"]),
        "## Applied Correction Categories",
        "",
        _markdown_table(category_rows, ["name", "count"]),
        "## Examples",
        "",
        _markdown_table(
            examples,
            [
                "clip_id",
                "model_name",
                "source_text",
                "replacement_text",
                "canonical_term",
                "category",
                "decision",
                "wer_delta",
                "term_recall_delta",
            ],
        ),
        "## Interpretation Notes",
        "",
        "- `unchanged` rows are not failures by default; they usually mean no safe glossary-grounded candidate was available or the baseline already had the terms.",
        "- `no_op` rows are equivalent wording or already acceptable forms, not term-rescue wins.",
        "- Held-out errors that look fixable but are missing from the frozen glossary should be recorded for a future dev iteration, not patched into this held-out run.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(content), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--glossary",
        type=Path,
        default=Path("data/controlled_terms/earnings22_multi_context_terms.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/results/earnings22_heldout_error_analysis.csv"),
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("docs/earnings22_heldout_error_analysis.md"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows = build_error_rows(_read_csv(args.input), glossary_path=args.glossary)
    _write_csv(args.output, rows)
    write_markdown(args.markdown_output, rows)
    print(f"Wrote {len(rows)} held-out error-analysis rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
