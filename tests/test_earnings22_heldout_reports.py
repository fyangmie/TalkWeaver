"""Offline tests for Earnings-22 held-out reports."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from backend.term_rescue import load_reference_glossary
from experiments.analyze_earnings22_heldout_errors import build_error_rows
from experiments.evaluate_earnings22_ablation import build_ablation_rows, summarize


ROOT = Path(__file__).resolve().parents[1]
GLOSSARY = ROOT / "data" / "controlled_terms" / "earnings22_multi_context_terms.json"


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class Earnings22HeldoutReportTests(unittest.TestCase):
    def test_ablation_builds_expected_variants(self) -> None:
        asr_rows = [
            {
                "clip_id": "clip",
                "dataset_name": "Earnings-22",
                "language": "en",
                "model_name": "base",
                "reference_text": "These measures include non GAAP measures.",
                "hypothesis_text": "These measures include non-gap measures.",
            }
        ]
        llm_rows = [
            {
                **asr_rows[0],
                "candidate_corrections": json.dumps(
                    [
                        {
                            "source_text": "non-gap",
                            "replacement_text": "non-GAAP",
                            "canonical_term": "non-GAAP",
                        }
                    ]
                ),
                "applied_corrections": json.dumps(
                    [
                        {
                            "source_text": "non-gap",
                            "replacement_text": "non-GAAP",
                            "canonical_term": "non-GAAP",
                            "validation": "accepted",
                        }
                    ]
                ),
                "rejected_corrections": "[]",
                "needs_review_corrections": "[]",
                "no_op_corrections": "[]",
                "corrected_text": "These measures include non-GAAP measures.",
                "api_used": "true",
            }
        ]

        rows = build_ablation_rows(
            asr_rows=asr_rows,
            llm_rows=llm_rows,
            glossary_entries=load_reference_glossary(GLOSSARY),
        )
        variants = {row["variant"] for row in rows}
        self.assertEqual(
            variants,
            {
                "asr_only",
                "glossary_candidates_only",
                "llm_without_rag_conservative",
                "rag_evidence_gate_v2",
                "rag_llm_verifier_v2",
            },
        )
        summaries = summarize(rows)
        self.assertEqual(len(summaries), 5)

    def test_error_analysis_classifies_applied_correction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "rag.csv"
            _write_csv(
                source,
                [
                    {
                        "clip_id": "clip",
                        "model_name": "base",
                        "wer_delta": "-0.1",
                        "term_recall_before": "0.0",
                        "term_recall_after": "1.0",
                        "applied_corrections": json.dumps(
                            [
                                {
                                    "source_text": "non-gap",
                                    "replacement_text": "non-GAAP",
                                    "canonical_term": "non-GAAP",
                                    "reason": "test",
                                }
                            ]
                        ),
                        "rejected_corrections": "[]",
                        "needs_review_corrections": "[]",
                        "no_op_corrections": "[]",
                    }
                ],
            )
            rows = build_error_rows(
                list(csv.DictReader(source.open(encoding="utf-8"))),
                glossary_path=GLOSSARY,
            )

        self.assertEqual(rows[0]["decision"], "applied")
        self.assertEqual(rows[0]["category"], "financial_metric")


if __name__ == "__main__":
    unittest.main()
