"""Network-free tests for the Phase R1 binary correction benchmark."""

from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from backend.eccogate_binary import score_binary_correction
from backend.llm_config import LLMConfig
from experiments.build_binary_safe_apply_benchmark import (
    COLUMNS,
    build_binary_benchmark,
    derive_binary_label,
    write_binary_benchmark,
)
from experiments.plot_binary_safe_apply_results import plot_binary_results
from experiments.run_binary_llm_self_judge import (
    build_binary_judge_messages,
    run_binary_self_judge,
)
from experiments.run_binary_safe_apply_experiment import (
    compute_binary_metrics,
    evaluate_binary_benchmark,
)


ROOT = Path(__file__).resolve().parents[1]


class BinaryLabelPolicyTests(unittest.TestCase):
    def test_improvement_above_margin_is_safe(self) -> None:
        label, rule = derive_binary_label(
            0.40,
            0.20,
            margin=0.01,
        )

        self.assertEqual(label, "safe_to_apply")
        self.assertIn("<", rule)

    def test_worse_or_borderline_correction_is_blocked(self) -> None:
        worse, _ = derive_binary_label(0.20, 0.30, margin=0.01)
        borderline, _ = derive_binary_label(0.20, 0.195, margin=0.01)

        self.assertEqual(worse, "do_not_apply")
        self.assertEqual(borderline, "do_not_apply")

    def test_unsafe_override_blocks_even_if_error_improves(self) -> None:
        label, rule = derive_binary_label(
            0.80,
            0.10,
            margin=0.01,
            unsafe_override=True,
        )

        self.assertEqual(label, "do_not_apply")
        self.assertIn("Safety override", rule)


class BinaryEccoGateTests(unittest.TestCase):
    def test_supported_term_correction_is_safe(self) -> None:
        prediction = score_binary_correction(
            {
                "raw_asr_text": "we use piano note",
                "proposed_corrected_text": "we use pyannote",
                "context": "Speaker diarization toolkit discussion.",
                "retrieved_terms": '["pyannote"]',
            }
        )

        self.assertEqual(prediction.decision, "safe_to_apply")

    def test_no_change_and_heavy_overlap_are_blocked(self) -> None:
        unchanged = score_binary_correction(
            {
                "raw_asr_text": "keep the rack",
                "proposed_corrected_text": "keep the rack",
            }
        )
        overlap = score_binary_correction(
            {
                "raw_asr_text": "speaker says piano note",
                "proposed_corrected_text": "speaker says pyannote",
                "context": "Speaker diarization discussion.",
                "retrieved_terms": '["pyannote"]',
                "overlap_flag": True,
                "heavy_overlap_flag": True,
            }
        )

        self.assertEqual(unchanged.decision, "do_not_apply")
        self.assertEqual(overlap.decision, "do_not_apply")


class BinaryBenchmarkTests(unittest.TestCase):
    def test_builder_combines_controlled_and_pilot_rows(self) -> None:
        rows = build_binary_benchmark(
            term_input=ROOT
            / "experiments/results/term_rescue_controlled.csv",
            overlap_input=ROOT
            / "experiments/results/overlap_safety_controlled.csv",
            pilot_input=ROOT / "data/pilot/selective_correction_pilot.csv",
            margin=0.01,
            predictions_dir=None,
        )

        self.assertEqual(len(rows), 327)
        self.assertEqual(set(rows[0]), set(COLUMNS))
        self.assertIn(
            "controlled_reference",
            {row["label_source"] for row in rows},
        )
        self.assertIn(
            "pilot_suggested_if_no_reference",
            {row["label_source"] for row in rows},
        )

    def test_evaluator_computes_unsafe_apply_rate(self) -> None:
        metrics = compute_binary_metrics(
            ["safe_to_apply", "do_not_apply", "do_not_apply"],
            ["safe_to_apply", "safe_to_apply", "do_not_apply"],
        )

        self.assertEqual(metrics["unsafe_apply_rate"], 0.5)
        self.assertEqual(metrics["false_block_rate"], 0.0)
        self.assertEqual(metrics["coverage"], 2 / 3)

    def test_evaluator_accepts_tiny_llm_rows(self) -> None:
        benchmark = [
            {
                "proposal_id": "one",
                "source": "test",
                "category": "test",
                "language": "en",
                "raw_asr_text": "piano note",
                "proposed_corrected_text": "pyannote",
                "reference_text": "pyannote",
                "context": "speaker diarization",
                "retrieved_terms": '["pyannote"]',
                "overlap_flag": False,
                "heavy_overlap_flag": False,
                "speaker_ambiguity_flag": False,
                "partial_utterance_flag": False,
                "error_before": 1.0,
                "error_after": 0.0,
                "error_delta": 1.0,
                "binary_label": "safe_to_apply",
                "label_source": "controlled_reference",
            }
        ]
        llm = [
            {
                "proposal_id": "one",
                "mode": "with_evidence",
                "decision": "safe_to_apply",
                "rationale": "Supported.",
                "provider": "test",
                "model": "test",
                "api_used": True,
            }
        ]
        results, summary = evaluate_binary_benchmark(benchmark, llm)

        self.assertIn(
            "llm_self_judge_with_evidence",
            {row["method"] for row in summary},
        )
        self.assertTrue(
            any(
                row["method"] == "llm_self_judge_with_evidence"
                for row in results
            )
        )


class BinaryLlmAndPlotTests(unittest.TestCase):
    def test_binary_prompt_does_not_include_reference(self) -> None:
        proposal = {
            "raw_asr_text": "where improved",
            "proposed_corrected_text": "WER improved",
            "reference_text": "WER improved",
            "context": "ASR metric",
            "retrieved_terms": '["WER"]',
        }
        prompt = build_binary_judge_messages(
            proposal,
            "with_evidence",
        )[1]["content"]

        self.assertNotIn("Reference", prompt)
        self.assertIn("safe_to_apply", prompt)

    def test_binary_llm_runner_can_use_injected_request(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "benchmark.csv"
            output = root / "predictions.csv"
            rows = [
                {
                    column: ""
                    for column in COLUMNS
                }
            ]
            rows[0].update(
                {
                    "proposal_id": "one",
                    "raw_asr_text": "piano note",
                    "proposed_corrected_text": "pyannote",
                }
            )
            write_binary_benchmark(rows, source)
            config = LLMConfig(
                provider="deepseek",
                api_key="test-only-key",
                model="test-model",
                base_url="https://example.invalid",
            )
            predictions = run_binary_self_judge(
                input_path=source,
                mode="no_evidence",
                output_path=output,
                config=config,
                request_fn=lambda _config, _messages: {
                    "decision": "safe_to_apply",
                    "rationale": "The edit is locally supported.",
                },
            )

            self.assertEqual(predictions[0]["decision"], "safe_to_apply")
            self.assertTrue(predictions[0]["api_used"])

    def test_help_requires_no_api(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "experiments/run_binary_llm_self_judge.py",
                "--help",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0)
        self.assertIn("safe-to-apply", completed.stdout)

    def test_plotter_creates_five_charts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            summary = pd.DataFrame(
                [
                    {
                        "method": "binary_eccogate",
                        "macro_f1": 0.8,
                        "unsafe_apply_rate": 0.1,
                        "false_block_rate": 0.2,
                        "error_delta_when_applied": 0.3,
                    },
                    {
                        "method": "always_apply",
                        "macro_f1": 0.2,
                        "unsafe_apply_rate": 1.0,
                        "false_block_rate": 0.0,
                        "error_delta_when_applied": 0.1,
                    },
                ]
            )
            results = pd.DataFrame(
                [
                    {
                        "method": "binary_eccogate",
                        "category": "overlap",
                        "gold_label": "do_not_apply",
                        "unsafe_apply": False,
                    },
                    {
                        "method": "always_apply",
                        "category": "overlap",
                        "gold_label": "do_not_apply",
                        "unsafe_apply": True,
                    },
                ]
            )
            paths = plot_binary_results(summary, results, directory)

            self.assertEqual(len(paths), 5)
            self.assertTrue(all(path.is_file() for path in paths))


if __name__ == "__main__":
    unittest.main()
