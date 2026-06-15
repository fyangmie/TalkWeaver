"""Network-free tests for the Phase R0 selective-correction pilot."""

from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from backend.eccogate import score_correction_proposal
from backend.llm_config import LLMConfig
from experiments.build_selective_correction_pilot import (
    COLUMNS,
    build_pilot_rows,
    validate_pilot_rows,
)
from experiments.plot_pilot_selective_correction import plot_pilot_results
from experiments.run_pilot_llm_self_judge import (
    build_self_judge_messages,
    run_self_judge,
)
from experiments.run_pilot_selective_correction_eval import (
    compute_selective_metrics,
    evaluate_pilot,
)


ROOT = Path(__file__).resolve().parents[1]


class PilotDatasetTests(unittest.TestCase):
    def test_schema_size_and_balance(self) -> None:
        rows = build_pilot_rows()
        validate_pilot_rows(rows)

        self.assertEqual(len(rows), 72)
        self.assertEqual(set(rows[0]), set(COLUMNS))
        labels = pd.Series(
            [row["suggested_gold_label"] for row in rows]
        ).value_counts()
        self.assertEqual(labels.to_dict(), {
            "accept": 24,
            "reject": 24,
            "needs_review": 24,
        })
        self.assertTrue(all(not row["human_checked_label"] for row in rows))
        self.assertTrue(
            all("pilot_auto_labeled" in row["notes"] for row in rows)
        )


class EccoGateTests(unittest.TestCase):
    def test_accepts_supported_local_term_edit(self) -> None:
        prediction = score_correction_proposal(
            {
                "raw_asr_text": "we use piano note",
                "proposed_corrected_text": "we use pyannote",
                "context": "Speaker diarization toolkit discussion.",
                "retrieved_terms": '["pyannote"]',
            }
        )

        self.assertEqual(prediction.decision, "accept")
        self.assertGreater(prediction.support_score, prediction.risk_score)

    def test_rejects_physical_rack_replacement(self) -> None:
        prediction = score_correction_proposal(
            {
                "raw_asr_text": "put it on the rack",
                "proposed_corrected_text": "put it on the RAG",
                "context": "A physical equipment rack is being discussed.",
                "retrieved_terms": '["RAG"]',
            }
        )

        self.assertEqual(prediction.decision, "reject")

    def test_abstains_on_weak_metric_context(self) -> None:
        prediction = score_correction_proposal(
            {
                "raw_asr_text": "where changed after the test",
                "proposed_corrected_text": "WER changed after the test",
                "context": "A test is mentioned without specifying ASR.",
                "retrieved_terms": '["WER"]',
            }
        )

        self.assertEqual(prediction.decision, "needs_review")


class LlmJudgeTests(unittest.TestCase):
    def test_prompt_evidence_modes_are_distinct(self) -> None:
        proposal = build_pilot_rows()[0]
        no_evidence = build_self_judge_messages(
            proposal,
            "no_evidence",
        )[1]["content"]
        with_evidence = build_self_judge_messages(
            proposal,
            "with_evidence",
        )[1]["content"]

        self.assertNotIn("Heavy overlap:", no_evidence)
        self.assertIn("Heavy overlap:", with_evidence)
        self.assertIn("Retrieved terms:", with_evidence)

    def test_runner_works_with_injected_request_without_network(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "pilot.csv"
            output = root / "predictions.csv"
            row = build_pilot_rows()[0]
            with source.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=COLUMNS)
                writer.writeheader()
                writer.writerow(row)
            config = LLMConfig(
                provider="deepseek",
                api_key="test-only-key",
                model="test-model",
                base_url="https://example.invalid",
            )
            rows = run_self_judge(
                input_path=source,
                mode="with_evidence",
                output_path=output,
                config=config,
                request_fn=lambda _config, _messages: {
                    "decision": "accept",
                    "rationale": "Supported by the supplied term evidence.",
                },
            )

            self.assertEqual(rows[0]["decision"], "accept")
            self.assertTrue(rows[0]["api_used"])
            self.assertFalse(rows[0]["fallback_used"])
            self.assertTrue(output.is_file())

    def test_help_does_not_require_api(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "experiments/run_pilot_llm_self_judge.py",
                "--help",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0)
        self.assertIn("no_evidence", completed.stdout)


class PilotEvaluationTests(unittest.TestCase):
    def test_metrics_capture_abstention_and_unsafe_accepts(self) -> None:
        metrics = compute_selective_metrics(
            ["accept", "reject", "needs_review"],
            ["accept", "accept", "needs_review"],
        )

        self.assertEqual(metrics["unsafe_accept_rate"], 0.5)
        self.assertEqual(metrics["needs_review_recall"], 1.0)
        self.assertEqual(metrics["coverage"], 2 / 3)

    def test_evaluator_accepts_tiny_llm_predictions(self) -> None:
        pilot = build_pilot_rows()[:3]
        llm = [
            {
                "proposal_id": row["proposal_id"],
                "mode": "with_evidence",
                "decision": row["suggested_gold_label"],
                "rationale": "Test fixture.",
                "provider": "test",
                "model": "test",
                "api_used": True,
                "fallback_used": False,
            }
            for row in pilot
        ]
        results, summary = evaluate_pilot(pilot, llm)

        methods = {row["method"] for row in summary}
        self.assertIn("EccoGate", methods)
        self.assertIn("llm_self_judge_with_evidence", methods)
        self.assertEqual(
            len(
                [
                    row
                    for row in results
                    if row["method"] == "llm_self_judge_with_evidence"
                ]
            ),
            3,
        )

    def test_plotter_writes_all_four_charts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            summary = pd.DataFrame(
                [
                    {
                        "method": "EccoGate",
                        "macro_f1": 0.6,
                        "unsafe_accept_rate": 0.1,
                        "needs_review_recall": 0.7,
                    },
                    {
                        "method": "always_accept",
                        "macro_f1": 0.2,
                        "unsafe_accept_rate": 1.0,
                        "needs_review_recall": 0.0,
                    },
                ]
            )
            results = pd.DataFrame(
                [
                    {
                        "method": "EccoGate",
                        "category": "heavy_overlap",
                        "gold_label": "reject",
                        "unsafe_accept": False,
                    },
                    {
                        "method": "always_accept",
                        "category": "heavy_overlap",
                        "gold_label": "reject",
                        "unsafe_accept": True,
                    },
                ]
            )
            paths = plot_pilot_results(summary, results, directory)

            self.assertEqual(len(paths), 4)
            self.assertTrue(all(path.is_file() for path in paths))


if __name__ == "__main__":
    unittest.main()
