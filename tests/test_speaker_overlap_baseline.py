"""Offline tests for Phase 2D speaker-time and overlap baselines."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from experiments.metrics.speaker_time_metrics import (
    best_speaker_label_mapping,
    boundary_mean_absolute_error,
    interruption_event_precision_recall_f1,
    overlap_event_precision_recall_f1,
    speaker_label_error_rate,
    turn_time_coverage,
)
from experiments.run_speaker_overlap_baseline import run_baseline


ROOT = Path(__file__).resolve().parents[1]


class SpeakerTimeMetricTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reference = [
            {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_A"},
            {"start": 2.0, "end": 4.0, "speaker": "SPEAKER_B"},
        ]
        self.permuted = [
            {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_01"},
            {"start": 2.0, "end": 4.0, "speaker": "SPEAKER_00"},
        ]

    def test_best_label_mapping_handles_permutation(self) -> None:
        mapping = best_speaker_label_mapping(
            self.reference,
            self.permuted,
        )

        self.assertEqual(mapping["SPEAKER_01"], "SPEAKER_A")
        self.assertEqual(mapping["SPEAKER_00"], "SPEAKER_B")

    def test_speaker_error_and_boundaries_for_exact_permutation(self) -> None:
        self.assertEqual(
            speaker_label_error_rate(self.reference, self.permuted),
            0.0,
        )
        self.assertEqual(
            turn_time_coverage(self.reference, self.permuted),
            1.0,
        )
        self.assertEqual(
            boundary_mean_absolute_error(
                self.reference,
                self.permuted,
            ),
            0.0,
        )

    def test_no_diarization_baseline_is_covered_but_unknown(self) -> None:
        predicted = [
            {"start": 0.0, "end": 4.0, "speaker": "UNKNOWN"},
        ]

        self.assertEqual(turn_time_coverage(self.reference, predicted), 1.0)
        self.assertEqual(
            speaker_label_error_rate(self.reference, predicted),
            1.0,
        )
        self.assertIsNone(
            boundary_mean_absolute_error(self.reference, predicted)
        )


class EventMetricTests(unittest.TestCase):
    def test_overlap_precision_recall_f1(self) -> None:
        reference = [
            {
                "type": "overlap",
                "start": 1.0,
                "end": 2.0,
                "speakers": ["A", "B"],
            }
        ]
        predicted = [
            {
                "type": "overlap",
                "start": 1.1,
                "end": 2.1,
                "speakers": ["X", "Y"],
            },
            {
                "type": "overlap",
                "start": 3.0,
                "end": 3.5,
                "speakers": ["X", "Y"],
            },
        ]

        result = overlap_event_precision_recall_f1(reference, predicted)

        self.assertEqual(result["precision"], 0.5)
        self.assertEqual(result["recall"], 1.0)
        self.assertAlmostEqual(result["f1"], 2 / 3)

    def test_interruption_matching_uses_speaker_pair(self) -> None:
        reference = [
            {
                "type": "interruption",
                "start": 1.0,
                "end": 1.5,
                "speakers": ["A", "B"],
            }
        ]
        predicted = [
            {
                "type": "interruption",
                "start": 1.1,
                "end": 1.4,
                "speakers": ["B", "A"],
            },
            {
                "type": "interruption",
                "start": 1.1,
                "end": 1.4,
                "speakers": ["A", "C"],
            },
        ]

        result = interruption_event_precision_recall_f1(
            reference,
            predicted,
        )

        self.assertEqual(result["precision"], 0.5)
        self.assertEqual(result["recall"], 1.0)
        self.assertAlmostEqual(result["f1"], 2 / 3)


class BaselineRunnerTests(unittest.TestCase):
    def test_no_diarization_and_reference_modes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            anchors = root / "anchors.json"
            events = root / "events.json"
            manifest = root / "manifest.csv"
            output = root / "results.csv"
            anchors.write_text(
                json.dumps(
                    [
                        {
                            "start": 0.0,
                            "end": 2.0,
                            "speaker": "A",
                            "text": "hello",
                        },
                        {
                            "start": 1.5,
                            "end": 3.0,
                            "speaker": "B",
                            "text": "there",
                        },
                    ]
                ),
                encoding="utf-8",
            )
            events.write_text(
                json.dumps(
                    [
                        {
                            "type": "overlap",
                            "start": 1.5,
                            "end": 2.0,
                            "speakers": ["A", "B"],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            row = {
                "clip_id": "clip",
                "dataset_name": "Test Meeting",
                "language": "en",
                "duration_seconds": "3",
                "audio_path": str(root / "audio.wav"),
                "anchors_path": str(anchors),
                "events_path": str(events),
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
                "experiments.run_speaker_overlap_baseline.get_settings",
                return_value=SimpleNamespace(hf_token=""),
            ):
                rows = run_baseline(
                    manifest=manifest,
                    output=output,
                )

            by_mode = {item["mode"]: item for item in rows}
            self.assertEqual(
                by_mode["no_diarization"]["speaker_label_error_rate"],
                1.0,
            )
            self.assertEqual(
                by_mode["reference_assisted"]["speaker_label_error_rate"],
                0.0,
            )
            self.assertEqual(
                by_mode["reference_assisted"]["overlap_f1"],
                1.0,
            )
            self.assertEqual(
                by_mode["reference_assisted"]["interruption_f1"],
                "",
            )
            self.assertTrue(
                by_mode["pyannote_optional"]["notes"].startswith(
                    "Skipped"
                )
            )


class CliTests(unittest.TestCase):
    def _help(self, relative_path: str) -> str:
        result = subprocess.run(
            [sys.executable, str(ROOT / relative_path), "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout

    def test_baseline_help(self) -> None:
        output = self._help("experiments/run_speaker_overlap_baseline.py")
        self.assertIn("--manifest", output)

    def test_reference_map_help(self) -> None:
        output = self._help("experiments/run_reference_workflow_maps.py")
        self.assertIn("--dataset", output)

    def test_workflow_preserves_existing_and_new_modes(self) -> None:
        output = self._help("scripts/run_talkweaver_workflow.py")
        self.assertIn("real", output)
        self.assertIn("reference", output)
        self.assertIn("prediction-json", output)
        self.assertIn("none", output)
        self.assertIn("pyannote", output)


if __name__ == "__main__":
    unittest.main()
