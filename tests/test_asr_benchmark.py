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
    word_error_rate,
)
from experiments.metrics.text_normalization import (
    normalize_for_cer,
    normalize_for_wer,
)
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
