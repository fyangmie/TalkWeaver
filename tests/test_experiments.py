"""Tests for Phase 7 metrics, ablation output, and charts."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from experiments.evaluate_terms import evaluate_term_error
from experiments.evaluate_wder import overlap_flag_error, simplified_wder
from experiments.evaluate_wer import read_text_input, word_error_rate
from experiments.plot_results import generate_charts
from experiments.run_ablation import generate_mock_ablation


class WerTests(unittest.TestCase):
    def test_fallback_wer_counts_substitution(self) -> None:
        score = word_error_rate(
            "speaker diarization works",
            "speaker diary station works",
            prefer_jiwer=False,
        )
        self.assertAlmostEqual(score, 2 / 3)

    def test_json_transcript_input_supports_field_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "transcript.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "raw_text": "piano note",
                            "corrected_text": "pyannote",
                        },
                        {
                            "raw_text": "the ear",
                            "corrected_text": "DER",
                        },
                    ]
                ),
                encoding="utf-8",
            )
            text = read_text_input(str(path), field="corrected_text")

        self.assertEqual(text, "pyannote DER")


class SpeakerMetricTests(unittest.TestCase):
    def test_temporal_metric_handles_overlap_ties(self) -> None:
        reference = [
            {
                "start": 0.0,
                "end": 3.2,
                "speaker": "SPEAKER_00",
                "speakers": ["SPEAKER_00"],
                "overlap": False,
            },
            {
                "start": 3.0,
                "end": 3.2,
                "speaker": "OVERLAP",
                "speakers": ["SPEAKER_00", "SPEAKER_01"],
                "overlap": True,
            },
        ]
        hypothesis = [dict(segment) for segment in reference]

        self.assertEqual(simplified_wder(reference, hypothesis), 0.0)
        self.assertEqual(overlap_flag_error(reference, hypothesis), 0.0)


class TermMetricTests(unittest.TestCase):
    def test_term_error_before_and_after_correction(self) -> None:
        reference = "Use pyannote for diarization and report WER and DER."
        before = evaluate_term_error(
            reference,
            "Use piano note for diary station and report where and the ear.",
        )
        after = evaluate_term_error(reference, reference)

        self.assertEqual(before["term_error_rate"], 1.0)
        self.assertEqual(after["term_error_rate"], 0.0)
        self.assertEqual(after["recall"], 1.0)


class AblationTests(unittest.TestCase):
    def test_mock_ablation_writes_required_rows_and_charts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_path = root / "ablation_results.csv"
            term_path = root / "term_error_results.csv"
            latency_path = root / "latency_results.csv"
            chart_dir = root / "charts"

            rows = generate_mock_ablation(
                result_path=result_path,
                term_result_path=term_path,
                latency_result_path=latency_path,
            )
            charts = generate_charts(
                result_path=result_path,
                chart_dir=chart_dir,
            )
            with result_path.open(encoding="utf-8") as handle:
                exported = list(csv.DictReader(handle))
            charts_exist = all(path.exists() for path in charts)

        self.assertEqual(len(rows), 6)
        self.assertEqual(len(exported), 6)
        self.assertTrue(all(row["is_mock"] == "true" for row in exported))
        self.assertGreater(float(exported[0]["wer"]), float(exported[-1]["wer"]))
        self.assertEqual(len(charts), 5)
        self.assertTrue(charts_exist)


if __name__ == "__main__":
    unittest.main()
