"""Tests for Earnings-22 finance RAG + LLM correction safeguards."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.llm_config import LLMConfig
from backend.term_rescue import load_reference_glossary
from experiments.evaluate_rag_llm_correction import (
    _candidate_corrections,
    _entry_by_key,
    _gate_correction,
    _validate_correction,
    evaluate_rows,
)


ROOT = Path(__file__).resolve().parents[1]
GLOSSARY = ROOT / "data" / "controlled_terms" / "earnings22_finance_terms.json"


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class Earnings22FinanceCorrectionTests(unittest.TestCase):
    def test_numeric_unit_candidate_preserves_number(self) -> None:
        entries = load_reference_glossary(GLOSSARY)
        candidates = _candidate_corrections(
            "A reinstatement of our dividend at 262 seems to share.",
            entries,
        )

        self.assertIn(
            {
                "source_text": "262 seems to share",
                "replacement_text": "262 cents a share",
                "canonical_term": "cents a share",
                "reason": "numeric-unit pattern near dividend/share context",
            },
            candidates,
        )

    def test_numeric_unit_correction_rejects_context_free_share(self) -> None:
        entries = load_reference_glossary(GLOSSARY)
        ok, reason = _validate_correction(
            {
                "source_text": "the share",
                "replacement_text": "cents a share",
                "canonical_term": "cents a share",
            },
            raw_text="Please put it on the share folder.",
            entries_by_key=_entry_by_key(entries),
        )

        self.assertFalse(ok)
        self.assertIn("explicit source number", reason)

    def test_acceptable_forms_are_not_canonicalized(self) -> None:
        entries = load_reference_glossary(GLOSSARY)
        ok, reason = _validate_correction(
            {
                "source_text": "Aspen's",
                "replacement_text": "Aspen",
                "canonical_term": "Aspen",
            },
            raw_text="Welcome to Aspen's results presentation.",
            entries_by_key=_entry_by_key(entries),
        )

        self.assertFalse(ok)
        self.assertIn("accepted glossary form", reason)

    def test_cents_per_share_is_no_op_not_rescue(self) -> None:
        entries = load_reference_glossary(GLOSSARY)
        decision, reason = _gate_correction(
            {
                "source_text": "42 cents per share",
                "replacement_text": "42 cents a share",
                "canonical_term": "cents a share",
            },
            raw_text="The dividend was 42 cents per share.",
            entries_by_key=_entry_by_key(entries),
        )

        self.assertEqual(decision, "no_op")
        self.assertIn("acceptable numeric-unit", reason)

    def test_evaluate_rows_writes_summary_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "asr.csv"
            output = root / "result.csv"
            summary = root / "summary.csv"
            report = root / "report.md"
            _write_csv(
                source,
                [
                    {
                        "clip_id": "earnings22_demo",
                        "dataset_name": "Earnings-22",
                        "language": "en",
                        "model_name": "base",
                        "reference_text": "A reinstatement of our dividend at 262 cents a share.",
                        "hypothesis_text": "A reinstatement of our dividend at 262 seems to share.",
                    }
                ],
            )
            config = LLMConfig(
                provider="deepseek",
                api_key="test-key",
                model="deepseek-chat",
                base_url="https://api.deepseek.com",
            )

            with patch(
                "experiments.evaluate_rag_llm_correction.request_json_completion",
                return_value={
                    "corrections": [
                        {
                            "source_text": "262 seems to share",
                            "replacement_text": "262 cents a share",
                            "canonical_term": "cents a share",
                            "reason": "candidate preserves number and dividend context",
                        }
                    ],
                    "needs_review": [],
                },
            ):
                rows = evaluate_rows(
                    input_path=source,
                    glossary_path=GLOSSARY,
                    output_path=output,
                    summary_output_path=summary,
                    markdown_output_path=report,
                    llm_config=config,
                )

            self.assertEqual(rows[0]["wer_after"], "0.000000")
            self.assertIn("262 cents a share", rows[0]["corrected_text"])
            self.assertEqual(json.loads(rows[0]["no_op_corrections"]), [])
            self.assertTrue(output.is_file())
            self.assertTrue(summary.is_file())
            self.assertTrue(report.is_file())
            self.assertIn(
                "262 seems to share -> 262 cents a share",
                report.read_text(encoding="utf-8"),
            )

    def test_evaluate_rows_skips_llm_when_no_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "asr.csv"
            output = root / "result.csv"
            _write_csv(
                source,
                [
                    {
                        "clip_id": "earnings22_demo",
                        "dataset_name": "Earnings-22",
                        "language": "en",
                        "model_name": "base",
                        "reference_text": "Revenue improved this quarter.",
                        "hypothesis_text": "Revenue improved this quarter.",
                    }
                ],
            )
            config = LLMConfig(
                provider="deepseek",
                api_key="test-key",
                model="deepseek-chat",
                base_url="https://api.deepseek.com",
            )

            with patch(
                "experiments.evaluate_rag_llm_correction.request_json_completion"
            ) as completion:
                rows = evaluate_rows(
                    input_path=source,
                    glossary_path=GLOSSARY,
                    output_path=output,
                    llm_config=config,
                )

            completion.assert_not_called()
            self.assertEqual(rows[0]["api_used"], "false")
            self.assertEqual(rows[0]["corrected_text"], "Revenue improved this quarter.")


class Earnings22MultiContextGateTests(unittest.TestCase):
    def test_us_to_ueps_is_rejected_as_common_token_entity_rewrite(self) -> None:
        entries = load_reference_glossary(
            ROOT / "data" / "controlled_terms" / "earnings22_multi_context_terms.json"
        )
        decision, reason = _gate_correction(
            {
                "source_text": "U.S.",
                "replacement_text": "UEPS",
                "canonical_term": "UEPS",
            },
            raw_text=(
                "Forward-looking statements are made under the safe harbor "
                "provisions of the U.S. Private Securities Litigation Reform Act."
            ),
            entries_by_key=_entry_by_key(entries),
        )

        self.assertEqual(decision, "reject")
        self.assertIn("common source", reason)

    def test_non_gap_to_non_gaap_is_accepted_in_financial_context(self) -> None:
        entries = load_reference_glossary(
            ROOT / "data" / "controlled_terms" / "earnings22_multi_context_terms.json"
        )
        decision, reason = _gate_correction(
            {
                "source_text": "non-gap",
                "replacement_text": "non-GAAP",
                "canonical_term": "non-GAAP",
            },
            raw_text=(
                "The earnings press release includes discussions of "
                "unaudited non-gap financial measures."
            ),
            entries_by_key=_entry_by_key(entries),
        )

        self.assertEqual(decision, "accept")
        self.assertIn("v2 evidence gate", reason)

    def test_v3_requires_predefined_error_form(self) -> None:
        entries = load_reference_glossary(GLOSSARY)
        decision, reason = _gate_correction(
            {
                "source_text": "market",
                "replacement_text": "oncology",
                "canonical_term": "oncology",
            },
            raw_text="The products performed well in the market.",
            entries_by_key=_entry_by_key(entries),
            gate_version="v3",
        )

        self.assertEqual(decision, "needs_review")
        self.assertIn("predefined ASR error form", reason)

    def test_v3_accepts_known_error_form_with_context(self) -> None:
        entries = load_reference_glossary(GLOSSARY)
        decision, reason = _gate_correction(
            {
                "source_text": "on college",
                "replacement_text": "oncology",
                "canonical_term": "oncology",
            },
            raw_text="The products and prices in the on college market improved.",
            entries_by_key=_entry_by_key(entries),
            gate_version="v3",
        )

        self.assertEqual(decision, "accept")
        self.assertIn("v3 evidence gate", reason)


if __name__ == "__main__":
    unittest.main()
