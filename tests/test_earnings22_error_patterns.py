"""Tests for Earnings-22 slice preparation and ASR error categorization."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from experiments.classify_asr_error_patterns import (
    classify_asr_error_patterns,
    classify_token_error,
)
from scripts.prepare_earnings22_eval_slices import (
    ReferenceToken,
    render_reference_text,
    tokens_for_window,
)


class Earnings22ReferenceSliceTests(unittest.TestCase):
    def test_windowed_tokens_render_reference_text(self) -> None:
        tokens = [
            ReferenceToken("Good", "0", 0.5, 0.6),
            ReferenceToken("morning", "0", 1.0, 1.2, punctuation=","),
            ReferenceToken("Aspen", "0", 181.0, 181.2),
        ]

        selected = tokens_for_window(tokens, start=0.0, end=180.0)
        rendered = render_reference_text(selected)

        self.assertEqual(rendered, "Good morning,")


class AsrErrorPatternTests(unittest.TestCase):
    def test_number_unit_error_is_actionable(self) -> None:
        category, impact, _reason, recommendation = classify_token_error(
            "replace",
            ["262", "cents", "a"],
            ["262", "seems", "to"],
        )

        self.assertEqual(category, "number_unit_error")
        self.assertEqual(impact, "meaning_likely_changed")
        self.assertIn("numeric-unit", recommendation)

    def test_disfluency_error_is_not_treated_as_domain_rescue(self) -> None:
        category, impact, _reason, recommendation = classify_token_error(
            "replace",
            ["gonna"],
            ["going", "to"],
        )

        self.assertEqual(category, "disfluency_or_style_error")
        self.assertEqual(impact, "mostly_style_or_reference_policy")
        self.assertIn("cleaned WER", recommendation)

    def test_classifier_writes_detail_and_summary_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "asr.csv"
            detail = root / "detail.csv"
            summary = root / "summary.csv"
            model_summary = root / "model_summary.csv"
            markdown = root / "report.md"
            row = {
                "clip_id": "clip",
                "dataset_name": "Earnings-22",
                "language": "en",
                "model_name": "base",
                "metric_name": "WER",
                "error_rate": "0.2",
                "cleaned_metric_name": "WER_DISFLUENCY_CLEANED",
                "cleaned_error_rate": "0.18",
                "reference_text": "A dividend at 262 cents a share.",
                "hypothesis_text": "A dividend at 262 seems to share.",
                "normalized_reference": "a dividend at 262 cents a share",
                "normalized_hypothesis": "a dividend at 262 seems to share",
            }
            with source.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)

            errors = classify_asr_error_patterns(
                input_path=source,
                output_path=detail,
                summary_output_path=summary,
                model_summary_output_path=model_summary,
                markdown_output_path=markdown,
            )

            self.assertTrue(detail.is_file())
            self.assertTrue(summary.is_file())
            self.assertTrue(model_summary.is_file())
            self.assertTrue(markdown.is_file())
            self.assertTrue(
                any(error.error_category == "number_unit_error" for error in errors)
            )


if __name__ == "__main__":
    unittest.main()
