"""Tests for the mobile-style ASR trade-off artifact builder."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from experiments.benchmark_mobile_asr import build_mobile_proxy_report


class MobileAsrTradeoffTests(unittest.TestCase):
    def test_proxy_report_marks_rows_as_not_true_mobile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "asr.csv"
            output = root / "mobile.csv"
            metadata = root / "metadata.json"
            rows = [
                {
                    "clip_id": "clip_001",
                    "dataset_name": "Demo",
                    "language": "en",
                    "model_name": "tiny",
                    "device": "cpu",
                    "compute_type": "int8",
                    "vad_filter": "true",
                    "duration_seconds": "10.0",
                    "runtime_seconds": "0.5",
                    "rtf": "0.05",
                    "cold_model_load_seconds": "1.2",
                    "metric_name": "WER",
                    "error_rate": "0.25",
                    "cleaned_metric_name": "WER_DISFLUENCY_CLEANED",
                    "cleaned_error_rate": "0.20",
                }
            ]
            with source.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

            built_rows = build_mobile_proxy_report(
                source,
                output,
                metadata,
                mode="proxy",
                hardware_label="test-cpu",
            )

            self.assertEqual(len(built_rows), 1)
            self.assertTrue(output.is_file())
            self.assertTrue(metadata.is_file())

            with output.open(encoding="utf-8", newline="") as handle:
                output_rows = list(csv.DictReader(handle))
            self.assertEqual(output_rows[0]["claim_level"], "mobile_style_proxy")
            self.assertEqual(output_rows[0]["true_mobile_device"], "false")
            self.assertEqual(output_rows[0]["backend"], "faster-whisper")
            self.assertEqual(output_rows[0]["quantization"], "int8")
            self.assertIn("not a true phone", output_rows[0]["notes"])

            payload = json.loads(metadata.read_text(encoding="utf-8"))
            self.assertEqual(payload["claim_level"], "mobile_style_proxy")
            self.assertFalse(payload["true_mobile_device"])
            self.assertEqual(payload["mode_effective"], "proxy")
            self.assertEqual(payload["row_count"], 1)


if __name__ == "__main__":
    unittest.main()
