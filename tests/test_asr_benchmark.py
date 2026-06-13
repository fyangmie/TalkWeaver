"""Offline tests for the real ASR baseline experiment infrastructure."""

from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.metrics.text_metrics import (
    character_error_rate,
    evaluate_cleaned_wer,
    word_error_rate,
)
from experiments.metrics.text_normalization import (
    chinese_script_normalization_available,
    normalize_chinese_script,
    normalize_for_cer,
    normalize_for_cleaned_wer,
    normalize_for_wer,
)
from experiments.plot_asr_results import plot_results
from experiments.run_asr_benchmark import run_benchmark
from experiments.summarize_asr_results import summarize_results


ROOT = Path(__file__).resolve().parents[1]


class TextMetricTests(unittest.TestCase):
    def test_word_error_rate(self) -> None:
        self.assertAlmostEqual(
            word_error_rate(
                "speaker diarization works",
                "speaker diary station works",
                prefer_jiwer=False,
            ),
            2 / 3,
        )

    def test_character_error_rate(self) -> None:
        self.assertEqual(character_error_rate("你好世界", "你好世"), 0.25)

    def test_english_and_french_normalization(self) -> None:
        self.assertEqual(
            normalize_for_wer(
                "L’utilisation de faster-whisper, pyannote.audio!"
            ),
            "l'utilisation de faster-whisper pyannote.audio",
        )
        self.assertEqual(
            normalize_for_wer("Hello, WORLD!"),
            "hello world",
        )

    def test_mandarin_normalization(self) -> None:
        self.assertEqual(
            normalize_for_cer("报告警告称，没有人。"),
            "报告警告称没有人",
        )

    def test_chinese_script_normalization_is_optional(self) -> None:
        traditional = "報告警告稱"
        simplified = "报告警告称"
        if chinese_script_normalization_available():
            self.assertEqual(
                normalize_chinese_script(traditional),
                simplified,
            )
            self.assertEqual(
                normalize_for_cer(traditional),
                normalize_for_cer(simplified),
            )
        else:
            with self.assertWarnsRegex(RuntimeWarning, "OpenCC unavailable"):
                result = normalize_chinese_script(traditional)
            self.assertEqual(result, traditional)

    def test_chinese_script_normalization_falls_back_gracefully(self) -> None:
        traditional = "報告警告稱"
        with patch(
            "experiments.metrics.text_normalization._chinese_converter",
            return_value=None,
        ):
            with self.assertWarnsRegex(RuntimeWarning, "OpenCC unavailable"):
                result = normalize_chinese_script(traditional)

        self.assertEqual(result, traditional)

    def test_cleaned_wer_removes_fillers_and_repetitions(self) -> None:
        reference = "Um well this is our our project . Mm-hmm ."
        hypothesis = "well this is our project"
        result = evaluate_cleaned_wer(reference, hypothesis)

        self.assertEqual(
            normalize_for_cleaned_wer(reference),
            "well this is our project",
        )
        self.assertEqual(result["cleaned_error_rate"], 0.0)


class SummaryTests(unittest.TestCase):
    def test_summarizer_aggregates_tiny_csv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.csv"
            output = root / "summary.csv"
            rows = [
                {
                    "model_name": "tiny",
                    "language": "en",
                    "dataset_name": "Demo",
                    "metric_name": "WER",
                    "duration_seconds": "5",
                    "runtime_seconds": "1",
                    "rtf": "0.2",
                    "error_rate": "0.1",
                    "vad_filter": "true",
                    "cold_model_load_seconds": "1.5",
                    "cleaned_metric_name": "WER_DISFLUENCY_CLEANED",
                    "cleaned_error_rate": "0.05",
                    "script_normalized": "false",
                    "normalization_notes": "test normalization",
                },
                {
                    "model_name": "tiny",
                    "language": "en",
                    "dataset_name": "Demo",
                    "metric_name": "WER",
                    "duration_seconds": "7",
                    "runtime_seconds": "2",
                    "rtf": "0.3",
                    "error_rate": "0.3",
                    "vad_filter": "true",
                    "cold_model_load_seconds": "1.5",
                    "cleaned_metric_name": "WER_DISFLUENCY_CLEANED",
                    "cleaned_error_rate": "0.15",
                    "script_normalized": "false",
                    "normalization_notes": "test normalization",
                },
            ]
            with source.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

            summaries = summarize_results(source, output)

            self.assertTrue(output.is_file())
            self.assertEqual(len(summaries), 1)
            self.assertEqual(summaries[0]["num_clips"], 2)
            self.assertEqual(summaries[0]["total_duration_seconds"], 12.0)
            self.assertEqual(summaries[0]["mean_error_rate"], 0.2)
            self.assertEqual(summaries[0]["median_rtf"], 0.25)
            self.assertEqual(
                summaries[0]["mean_cleaned_error_rate"],
                0.1,
            )
            self.assertEqual(
                summaries[0]["cold_model_load_seconds"],
                1.5,
            )

    def test_plotter_writes_by_dataset_chart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.csv"
            output = root / "charts"
            rows = [
                {
                    "model_name": "tiny",
                    "language": "en",
                    "dataset_name": "FLEURS",
                    "metric_name": "WER",
                    "error_rate": "0.2",
                    "rtf": "0.1",
                },
                {
                    "model_name": "base",
                    "language": "en",
                    "dataset_name": "AMI",
                    "metric_name": "WER",
                    "error_rate": "0.3",
                    "rtf": "0.2",
                },
            ]
            with source.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

            charts = plot_results(source, output)

            self.assertEqual(len(charts), 3)
            self.assertTrue(
                (output / "asr_error_by_dataset.png").is_file()
            )


class BenchmarkCliTests(unittest.TestCase):
    def test_benchmark_help_requires_no_model(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "experiments" / "run_asr_benchmark.py"),
                "--help",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--models", result.stdout)
        self.assertIn("--vad-filter", result.stdout)
        self.assertIn("--only-dataset", result.stdout)

    def test_plot_help_requires_no_results(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "experiments" / "plot_asr_results.py"),
                "--help",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--output-dir", result.stdout)

    def test_model_load_failure_does_not_write_mock_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "clip.wav"
            transcript = root / "reference.txt"
            manifest = root / "manifest.csv"
            output = root / "results.csv"
            predictions = root / "predictions"
            audio.write_bytes(b"real-file-placeholder")
            transcript.write_text("real reference", encoding="utf-8")
            row = {
                "clip_id": "clip",
                "audio_path": str(audio),
                "dataset_name": "Test",
                "language": "en",
                "duration_seconds": "1",
                "transcript_path": str(transcript),
            }
            with manifest.open(
                "w",
                encoding="utf-8",
                newline="",
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)

            with patch(
                "experiments.run_asr_benchmark.load_faster_whisper_model",
                side_effect=RuntimeError("real model unavailable"),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "real model unavailable",
                ):
                    run_benchmark(
                        manifest=manifest,
                        models=["tiny"],
                        device="cpu",
                        compute_type="int8",
                        output=output,
                        predictions_dir=predictions,
                    )

            self.assertFalse(output.exists())
            self.assertFalse(predictions.exists())


if __name__ == "__main__":
    unittest.main()
