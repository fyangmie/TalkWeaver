"""Tests for added diarization, interruption, and whisper.cpp research tracks."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from experiments.benchmark_whisper_cpp import run_benchmark as run_whisper_cpp
from experiments.metrics.standard_diarization_metrics import (
    compute_der_jer,
    standard_diarization_metrics_available,
)
from experiments.run_pyannote_diarization_benchmark import (
    run_benchmark as run_pyannote_benchmark,
)
from scripts.dataset_utils import MANIFEST_COLUMNS, write_manifest
from scripts.generate_interruption_label_candidates import build_candidates
from scripts.validate_interruption_labels import validate_labels


def _manifest_row(root: Path) -> dict[str, str]:
    audio = root / "audio.wav"
    transcript = root / "reference.txt"
    anchors = root / "anchors.json"
    terms = root / "terms.json"
    events = root / "events.json"
    audio.write_bytes(b"RIFFtiny-test")
    transcript.write_text("hello there\n", encoding="utf-8")
    anchors.write_text(
        json.dumps(
            [
                {"start": 0.0, "end": 2.0, "speaker": "A", "text": "hello"},
                {"start": 1.5, "end": 3.0, "speaker": "B", "text": "there"},
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
    terms.write_text("[]\n", encoding="utf-8")
    row = {column: "" for column in MANIFEST_COLUMNS}
    row.update(
        {
            "clip_id": "meeting_clip",
            "audio_path": str(audio),
            "source_type": "public_dataset",
            "dataset_name": "Test Meeting",
            "dataset_version": "test",
            "split": "test",
            "language": "en",
            "duration_seconds": "3.0",
            "speaker_count": "2",
            "has_overlap": "true",
            "has_interruptions": "false",
            "has_domain_terms": "false",
            "recording_device": "test",
            "noise_condition": "clean",
            "consent_status": "test",
            "redistribution_status": "test",
            "license_or_access": "test",
            "transcript_path": str(transcript),
            "anchors_path": str(anchors),
            "terms_path": str(terms),
            "events_path": str(events),
            "download_status": "prepared",
            "notes": "unit test",
        }
    )
    return row


class StandardDiarizationMetricTests(unittest.TestCase):
    def test_exact_match_der_jer_zero_when_available(self) -> None:
        reference = [
            {"start": 0.0, "end": 1.0, "speaker": "A"},
            {"start": 1.0, "end": 2.0, "speaker": "B"},
        ]
        result = compute_der_jer(reference, reference, uri="exact")
        if not standard_diarization_metrics_available():
            self.assertEqual(result["status"], "skipped")
            return
        self.assertEqual(result["status"], "ok")
        self.assertAlmostEqual(result["der"], 0.0)
        self.assertAlmostEqual(result["jer"], 0.0)


class PyannoteBenchmarkTests(unittest.TestCase):
    def test_missing_token_writes_skipped_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.csv"
            output = root / "pyannote.csv"
            write_manifest(manifest, [_manifest_row(root)])

            with patch(
                "experiments.run_pyannote_diarization_benchmark.get_settings",
                return_value=SimpleNamespace(hf_token=""),
            ):
                rows = run_pyannote_benchmark(
                    manifest=manifest,
                    output=output,
                    max_clips=1,
                )

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["metric_status"], "skipped")
            self.assertIn("HF_TOKEN", rows[0]["notes"])


class InterruptionLabelTests(unittest.TestCase):
    def test_generate_and_validate_uncertain_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.csv"
            candidates = root / "candidates.csv"
            write_manifest(manifest, [_manifest_row(root)])

            rows = build_candidates(
                manifest=manifest,
                output=candidates,
                min_overlap_seconds=0.2,
            )

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["label"], "uncertain")
            self.assertEqual(validate_labels(labels_path=candidates, manifest=manifest), [])

    def test_validator_rejects_bad_label(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.csv"
            labels = root / "labels.csv"
            write_manifest(manifest, [_manifest_row(root)])
            with labels.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "clip_id",
                        "start",
                        "end",
                        "interrupter",
                        "interrupted",
                        "label",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "clip_id": "meeting_clip",
                        "start": "1.5",
                        "end": "2.0",
                        "interrupter": "B",
                        "interrupted": "A",
                        "label": "maybe",
                    }
                )

            errors = validate_labels(labels_path=labels, manifest=manifest)
            self.assertTrue(any("invalid label" in error for error in errors))


class WhisperCppBenchmarkTests(unittest.TestCase):
    def test_missing_executable_writes_skipped_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.csv"
            output = root / "whisper_cpp.csv"
            write_manifest(manifest, [_manifest_row(root)])

            rows = run_whisper_cpp(
                manifest=manifest,
                output=output,
                executable="definitely_missing_whisper_cpp",
                model_specs=["tiny=missing.bin"],
                max_clips=1,
            )

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["status"], "skipped")
            self.assertIn("executable", rows[0]["notes"])


if __name__ == "__main__":
    unittest.main()
