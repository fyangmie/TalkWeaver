"""Offline tests for overlap-aware correction safety experiments."""

from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from experiments.metrics.correction_safety_metrics import (
    conservative_rejection_rate,
    invented_content_flag,
    overcorrection_rate,
    speaker_attribution_change_flag,
    unsupported_change_count,
    unsupported_changes,
)
from experiments.plot_overlap_safety import plot_results
from experiments.run_overlap_safety_experiment import (
    load_cases,
    load_safety_policy,
    run_experiment,
)
from experiments.summarize_overlap_safety import summarize_results


ROOT = Path(__file__).resolve().parents[1]
CASES = (
    ROOT
    / "data"
    / "controlled_overlap"
    / "overlap_correction_cases.jsonl"
)
POLICY = (
    ROOT
    / "data"
    / "controlled_overlap"
    / "overlap_safety_policy.json"
)


class SafetyMetricTests(unittest.TestCase):
    def test_unsupported_and_invented_content_detection(self) -> None:
        unsupported = unsupported_changes(
            "the result is unclear",
            "the result is approved friday",
            supported_evidence=["the result is unclear"],
        )

        self.assertEqual(unsupported, ["approved", "friday"])
        self.assertEqual(
            unsupported_change_count(
                "the result is unclear",
                "the result is approved friday",
                supported_evidence=["the result is unclear"],
            ),
            2,
        )
        self.assertTrue(
            invented_content_flag(
                "the result is unclear",
                "the result is approved friday",
                supported_evidence=["the result is unclear"],
            )
        )

    def test_speaker_attribution_change_detection(self) -> None:
        self.assertTrue(
            speaker_attribution_change_flag(
                ["SPEAKER_A", "SPEAKER_B"],
                ["SPEAKER_B", "SPEAKER_A"],
            )
        )
        self.assertFalse(
            speaker_attribution_change_flag(
                ["SPEAKER_A", "SPEAKER_B"],
                ["SPEAKER_A", "SPEAKER_B"],
            )
        )
        self.assertEqual(overcorrection_rate([True, False]), 0.5)
        self.assertEqual(
            conservative_rejection_rate([False, True, True]),
            2 / 3,
        )


class ControlledOverlapFixtureTests(unittest.TestCase):
    def test_fixture_and_policy_loaders(self) -> None:
        cases = load_cases(CASES)
        policy = load_safety_policy(POLICY)

        self.assertGreaterEqual(len(cases), 18)
        self.assertTrue(
            all(
                case["fixture_type"]
                == "controlled_overlap_correction"
                for case in cases
            )
        )
        self.assertTrue(
            policy["overlap_conservative_rules"][
                "mark_all_overlap_for_review"
            ]
        )


class ControlledOverlapRunnerTests(unittest.TestCase):
    def test_heavy_overlap_is_retained_and_marked_for_review(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.csv"
            rows = run_experiment(
                cases_path=CASES,
                policy_path=POLICY,
                output_path=output,
            )

            aware = next(
                row
                for row in rows
                if row["case_id"] == "overlap_009"
                and row["variant"] == "overlap_aware_rule"
            )
            unaware = next(
                row
                for row in rows
                if row["case_id"] == "overlap_009"
                and row["variant"] == "no_overlap_awareness_rule"
            )
            self.assertEqual(
                aware["corrected_text"],
                aware["raw_asr_text"],
            )
            self.assertTrue(aware["needs_review"])
            self.assertTrue(aware["correction_rejected"])
            self.assertTrue(aware["safety_pass"])
            self.assertFalse(unaware["needs_review"])
            self.assertFalse(unaware["safety_pass"])
            self.assertFalse(any(row["api_used"] for row in rows))

    def test_negative_control_rack_is_not_converted_to_rag(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            rows = run_experiment(
                cases_path=CASES,
                policy_path=POLICY,
                output_path=Path(directory) / "results.csv",
            )

            rows = [
                row for row in rows if row["case_id"] == "overlap_004"
            ]
            self.assertEqual(len(rows), 2)
            self.assertTrue(
                all(row["corrected_text"] == row["raw_asr_text"] for row in rows)
            )
            self.assertTrue(
                all("RAG" not in row["corrected_text"] for row in rows)
            )

    def test_summary_and_plots_accept_offline_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "results.csv"
            run_experiment(
                cases_path=CASES,
                policy_path=POLICY,
                output_path=results,
            )
            summary = root / "summary.csv"

            summary_rows = summarize_results(results, summary)
            charts = plot_results(results, root / "charts")

            self.assertTrue(summary_rows)
            self.assertTrue(summary.exists())
            self.assertEqual(len(charts), 4)
            self.assertTrue(all(chart.exists() for chart in charts))
            with summary.open(encoding="utf-8", newline="") as handle:
                self.assertTrue(list(csv.DictReader(handle)))

    def test_runner_help_requires_no_api(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(
                    ROOT
                    / "experiments"
                    / "run_overlap_safety_experiment.py"
                ),
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
