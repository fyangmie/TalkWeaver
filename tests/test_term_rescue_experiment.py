"""Offline tests for controlled term recovery and correction safety."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from backend.term_rescue import (
    load_reference_glossary,
    retrieve_controlled_matches,
)
from experiments.plot_term_rescue import plot_results
from experiments.run_term_rescue_experiment import (
    load_cases,
    run_experiment,
)
from experiments.summarize_term_rescue import summarize_results


ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "data" / "controlled_terms" / "term_rescue_cases.jsonl"
TERMS = ROOT / "data" / "controlled_terms" / "reference_terms.json"


class ControlledFixtureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.entries = load_reference_glossary(TERMS)

    def test_fixture_loader_marks_controlled_cases(self) -> None:
        cases = load_cases(CASES)

        self.assertGreaterEqual(len(cases), 20)
        self.assertTrue(
            all(case["fixture_type"] == "controlled_technical_term" for case in cases)
        )

    def test_exact_glossary_retrieves_known_error_form(self) -> None:
        accepted, rejected = retrieve_controlled_matches(
            "we use piano note for the speakers",
            self.entries,
            strategy="exact_glossary",
            context="speaker diarization pipeline",
        )

        self.assertIn("pyannote", [match.canonical for match in accepted])
        self.assertFalse(rejected)

    def test_fuzzy_retrieval_handles_unlisted_typo(self) -> None:
        accepted, _rejected = retrieve_controlled_matches(
            "pianote labels speakers",
            self.entries,
            strategy="fuzzy",
            context="speaker diarization pipeline",
        )

        self.assertIn("pyannote", [match.canonical for match in accepted])

    def test_phonetic_like_retrieval_handles_spoken_variant(self) -> None:
        accepted, _rejected = retrieve_controlled_matches(
            "pie note labels speakers",
            self.entries,
            strategy="phonetic_like",
            context="speaker diarization pipeline",
        )

        self.assertIn("pyannote", [match.canonical for match in accepted])

    def test_fused_retrieval_withholds_common_word_negative_control(self) -> None:
        accepted, rejected = retrieve_controlled_matches(
            "put the router on the rack near the wall",
            self.entries,
            strategy="fused",
            context="physical equipment rack in a server room",
        )

        self.assertNotIn("RAG", [match.canonical for match in accepted])
        self.assertIn("RAG", [match.canonical for match in rejected])
        self.assertTrue(rejected[0].needs_review)


class ControlledRunnerTests(unittest.TestCase):
    def test_safe_correction_and_ambiguous_review_are_auditable(self) -> None:
        cases = load_cases(CASES)
        selected = [cases[0], next(case for case in cases if case["case_id"] == "term_019")]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = root / "cases.jsonl"
            fixture.write_text(
                "\n".join(
                    json.dumps(
                        {key: value for key, value in case.items() if key != "fixture_type"},
                        ensure_ascii=False,
                    )
                    for case in selected
                )
                + "\n",
                encoding="utf-8",
            )
            output = root / "results.csv"
            candidates = root / "candidates.jsonl"

            rows = run_experiment(
                cases_path=fixture,
                terms_path=TERMS,
                output_path=output,
                candidates_output_path=candidates,
            )

            positive = next(
                row
                for row in rows
                if row["case_id"] == "term_001"
                and row["variant"] == "fused_plus_rule_correction"
            )
            negative = next(
                row
                for row in rows
                if row["case_id"] == "term_019"
                and row["variant"] == "fused_plus_rule_correction"
            )
            self.assertIn("pyannote", positive["corrected_text"])
            self.assertEqual(json.loads(positive["unsupported_changes"]), [])
            self.assertEqual(
                negative["corrected_text"],
                negative["raw_asr_text"],
            )
            self.assertTrue(negative["needs_review"])
            self.assertEqual(json.loads(negative["unsupported_changes"]), [])
            self.assertFalse(any(row["api_used"] for row in rows))

    def test_summary_and_plot_scripts_accept_tiny_csv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "results.csv"
            candidates = root / "candidates.jsonl"
            run_experiment(
                cases_path=CASES,
                terms_path=TERMS,
                output_path=results,
                candidates_output_path=candidates,
            )

            summary = root / "summary.csv"
            summary_rows = summarize_results(results, summary)
            charts = plot_results(results, root / "charts")

            self.assertTrue(summary_rows)
            self.assertTrue(summary.exists())
            self.assertEqual(len(charts), 3)
            self.assertTrue(all(chart.exists() for chart in charts))
            with summary.open(encoding="utf-8", newline="") as handle:
                self.assertTrue(list(csv.DictReader(handle)))

    def test_runner_help_requires_no_api(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "experiments" / "run_term_rescue_experiment.py"),
                "--help",
            ],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--include-llm-if-configured", result.stdout)


if __name__ == "__main__":
    unittest.main()
