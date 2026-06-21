#!/usr/bin/env python3
"""Summarize Earnings-22 ASR/RAG/LLM ablation rows without extra API calls."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.term_rescue import load_reference_glossary  # noqa: E402
from experiments.evaluate_rag_llm_correction import (  # noqa: E402
    _apply_llm_corrections,
    _candidate_corrections,
    _entry_by_key,
)
from experiments.evaluate_term_rescue_real_audit import (  # noqa: E402
    _term_metrics,
    _terms_present,
)
from experiments.metrics.text_metrics import evaluate_text  # noqa: E402


OUTPUT_COLUMNS = [
    "variant",
    "clip_id",
    "dataset_name",
    "language",
    "model_name",
    "wer",
    "term_recall",
    "term_f1",
    "candidate_count",
    "applied_count",
    "rejected_count",
    "needs_review_count",
    "no_op_count",
    "api_used",
    "reference_text",
    "hypothesis_text",
    "corrected_text",
    "notes",
]

SUMMARY_COLUMNS = [
    "variant",
    "dataset_name",
    "language",
    "model_name",
    "num_rows",
    "mean_wer",
    "mean_term_recall",
    "mean_term_f1",
    "candidate_count",
    "applied_count",
    "rejected_count",
    "needs_review_count",
    "no_op_count",
    "api_used_count",
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


def _score_row(
    *,
    variant: str,
    row: dict[str, str],
    corrected_text: str,
    candidate_count: int,
    applied_count: int,
    rejected_count: int,
    needs_review_count: int,
    no_op_count: int,
    api_used: bool,
    glossary_entries: list[Any],
    notes: str,
) -> dict[str, str]:
    score = evaluate_text(
        row["reference_text"],
        corrected_text,
        row.get("language"),
    )
    reference_terms = _terms_present(
        glossary_entries,
        row["reference_text"],
        row.get("language", ""),
    )
    corrected_terms = _terms_present(
        glossary_entries,
        corrected_text,
        row.get("language", ""),
    )
    term_score = _term_metrics(reference_terms, corrected_terms)
    return {
        "variant": variant,
        "clip_id": row.get("clip_id", ""),
        "dataset_name": row.get("dataset_name", ""),
        "language": row.get("language", ""),
        "model_name": row.get("model_name", ""),
        "wer": f"{float(score['error_rate']):.6f}",
        "term_recall": f"{term_score['recall']:.6f}",
        "term_f1": f"{term_score['f1']:.6f}",
        "candidate_count": str(candidate_count),
        "applied_count": str(applied_count),
        "rejected_count": str(rejected_count),
        "needs_review_count": str(needs_review_count),
        "no_op_count": str(no_op_count),
        "api_used": str(api_used).lower(),
        "reference_text": row["reference_text"],
        "hypothesis_text": row["hypothesis_text"],
        "corrected_text": corrected_text,
        "notes": notes,
    }


def build_ablation_rows(
    *,
    asr_rows: list[dict[str, str]],
    llm_rows: list[dict[str, str]],
    glossary_entries: list[Any],
    gate_version: str = "v2",
) -> list[dict[str, str]]:
    entries_by_key = _entry_by_key(glossary_entries)
    llm_by_key = {
        (row["clip_id"], row["model_name"]): row
        for row in llm_rows
    }
    results: list[dict[str, str]] = []
    for row in asr_rows:
        key = (row["clip_id"], row["model_name"])
        candidates = _candidate_corrections(row["hypothesis_text"], glossary_entries)
        results.append(
            _score_row(
                variant="asr_only",
                row=row,
                corrected_text=row["hypothesis_text"],
                candidate_count=len(candidates),
                applied_count=0,
                rejected_count=0,
                needs_review_count=0,
                no_op_count=0,
                api_used=False,
                glossary_entries=glossary_entries,
                notes="Raw ASR baseline with no correction.",
            )
        )

        results.append(
            _score_row(
                variant="glossary_candidates_only",
                row=row,
                corrected_text=row["hypothesis_text"],
                candidate_count=len(candidates),
                applied_count=0,
                rejected_count=0,
                needs_review_count=0,
                no_op_count=0,
                api_used=False,
                glossary_entries=glossary_entries,
                notes=(
                    "Retrieved glossary candidates are counted but no text is "
                    "changed; this isolates retrieval coverage from correction."
                ),
            )
        )

        results.append(
            _score_row(
                variant="llm_without_rag_conservative",
                row=row,
                corrected_text=row["hypothesis_text"],
                candidate_count=0,
                applied_count=0,
                rejected_count=0,
                needs_review_count=0,
                no_op_count=0,
                api_used=False,
                glossary_entries=glossary_entries,
                notes=(
                    "Offline no-RAG conservative baseline: no retrieved "
                    "candidate is available, so no term substitution is made."
                ),
            )
        )

        deterministic_text, applied, rejected, needs_review, no_op = _apply_llm_corrections(
            raw_text=row["hypothesis_text"],
            payload={"corrections": candidates, "needs_review": []},
            entries_by_key=entries_by_key,
            gate_version=gate_version,
        )
        results.append(
            _score_row(
                variant=f"rag_evidence_gate_{gate_version}",
                row=row,
                corrected_text=deterministic_text,
                candidate_count=len(candidates),
                applied_count=len(applied),
                rejected_count=len(rejected),
                needs_review_count=len(needs_review),
                no_op_count=len(no_op),
                api_used=False,
                glossary_entries=glossary_entries,
                notes=(
                    "RAG candidates passed through the deterministic v2 "
                    f"evidence gate {gate_version}, without LLM verification."
                ),
            )
        )

        llm_row = llm_by_key.get(key)
        if llm_row is None:
            continue
        results.append(
            _score_row(
                variant=f"rag_llm_verifier_{gate_version}",
                row=row,
                corrected_text=llm_row["corrected_text"],
                candidate_count=len(_json_list(llm_row["candidate_corrections"])),
                applied_count=len(_json_list(llm_row["applied_corrections"])),
                rejected_count=len(_json_list(llm_row["rejected_corrections"])),
                needs_review_count=len(_json_list(llm_row["needs_review_corrections"])),
                no_op_count=len(_json_list(llm_row.get("no_op_corrections", "[]"))),
                api_used=llm_row.get("api_used", "").lower() == "true",
                glossary_entries=glossary_entries,
                notes=(
                    "RAG candidates sent to the conservative LLM verifier, "
                    f"then checked by the same {gate_version} evidence gate; rows without "
                    "candidates skip the API."
                ),
            )
        )
    return results


def summarize(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: defaultdict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["variant"],
                row["dataset_name"],
                row["language"],
                row["model_name"],
            )
        ].append(row)
    summaries: list[dict[str, str]] = []
    for (variant, dataset_name, language, model_name), group in sorted(grouped.items()):
        summaries.append(
            {
                "variant": variant,
                "dataset_name": dataset_name,
                "language": language,
                "model_name": model_name,
                "num_rows": str(len(group)),
                "mean_wer": f"{sum(float(row['wer']) for row in group) / len(group):.6f}",
                "mean_term_recall": f"{sum(float(row['term_recall']) for row in group) / len(group):.6f}",
                "mean_term_f1": f"{sum(float(row['term_f1']) for row in group) / len(group):.6f}",
                "candidate_count": str(sum(int(row["candidate_count"]) for row in group)),
                "applied_count": str(sum(int(row["applied_count"]) for row in group)),
                "rejected_count": str(sum(int(row["rejected_count"]) for row in group)),
                "needs_review_count": str(sum(int(row["needs_review_count"]) for row in group)),
                "no_op_count": str(sum(int(row["no_op_count"]) for row in group)),
                "api_used_count": str(sum(row["api_used"] == "true" for row in group)),
            }
        )
    return summaries


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
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
        lines.append("| " + " | ".join(row.get(column, "") for column in columns) + " |")
    return "\n".join(lines) + "\n"


def write_markdown(
    path: Path,
    summary_rows: list[dict[str, str]],
    *,
    gate_version: str,
) -> None:
    content = [
        "# Earnings-22 Held-Out Ablation",
        "",
        "This report compares raw ASR, deterministic RAG candidate validation, and RAG + LLM verification on the frozen held-out subset.",
        f"The deterministic correction variants use evidence gate {gate_version}: common-token entity rewrites are rejected and equivalent wording is counted as no_op.",
        "",
        _markdown_table(
            summary_rows,
            [
                "variant",
                "model_name",
                "num_rows",
                "mean_wer",
                "mean_term_recall",
                "mean_term_f1",
                "candidate_count",
                "applied_count",
                "no_op_count",
                "api_used_count",
            ],
        ),
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(content), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asr-input", type=Path, required=True)
    parser.add_argument("--llm-input", type=Path, required=True)
    parser.add_argument(
        "--glossary",
        type=Path,
        default=Path("data/controlled_terms/earnings22_multi_context_terms.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/results/earnings22_heldout_ablation.csv"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("experiments/results/earnings22_heldout_ablation_summary.csv"),
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("docs/earnings22_heldout_ablation.md"),
    )
    parser.add_argument(
        "--gate-version",
        choices=("v2", "v3"),
        default="v2",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    glossary_entries = load_reference_glossary(args.glossary)
    rows = build_ablation_rows(
        asr_rows=_read_csv(args.asr_input),
        llm_rows=_read_csv(args.llm_input),
        glossary_entries=glossary_entries,
        gate_version=args.gate_version,
    )
    summary_rows = summarize(rows)
    _write_csv(args.output, rows, OUTPUT_COLUMNS)
    _write_csv(args.summary_output, summary_rows, SUMMARY_COLUMNS)
    write_markdown(
        args.markdown_output,
        summary_rows,
        gate_version=args.gate_version,
    )
    print(f"Wrote {len(rows)} ablation rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
