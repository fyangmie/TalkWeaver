"""Offline tests for the Phase 2E TalkWeaver workflow ablation."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from backend.workflow_variants import (
    VARIANT_NAMES,
    build_constrained_correction_variant,
    build_full_talkweaver_variant,
    build_overlap_aware_variant,
    build_term_rescue_variant,
    build_workflow_variant,
)
from experiments.metrics.text_metrics import evaluate_text
from experiments.plot_workflow_ablation import plot_results
from experiments.prediction_loader import load_prediction_json
from experiments.run_workflow_ablation import run_ablation
from experiments.summarize_workflow_ablation import summarize_results


ROOT = Path(__file__).resolve().parents[1]
GLOSSARY = ROOT / "docs" / "knowledge_base"


def _segments(text: str = "We use piano note.") -> list[dict]:
    words = []
    tokens = text.split()
    step = 1.0 / max(1, len(tokens))
    for index, word in enumerate(tokens):
        words.append(
            {
                "word": word,
                "start": index * step,
                "end": (index + 1) * step,
                "probability": 0.9,
            }
        )
    return [
        {
            "start": 0.0,
            "end": 1.0,
            "text": text,
            "words": words,
        }
    ]


class PredictionLoaderTests(unittest.TestCase):
    def test_loads_real_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "base__clip.json"
            source.write_text(
                json.dumps(
                    {
                        "is_mock": False,
                        "clip_id": "clip",
                        "dataset_name": "Demo",
                        "language": "en",
                        "model_name": "base",
                        "reference_text": "pyannote",
                        "hypothesis_text": "piano note",
                        "metric_name": "WER",
                        "error_rate": 1.0,
                        "asr": {
                            "asr_mode": "real",
                            "segments": _segments("piano note"),
                        },
                    }
                ),
                encoding="utf-8",
            )

            prediction = load_prediction_json(source)

            self.assertEqual(prediction.clip_id, "clip")
            self.assertEqual(
                prediction.as_asr_output()["mode"],
                "real_prediction_json",
            )

    def test_rejects_mock_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "mock.json"
            source.write_text(
                json.dumps(
                    {
                        "is_mock": True,
                        "asr": {"asr_mode": "mock", "segments": []},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "Mock prediction"):
                load_prediction_json(source)


class VariantBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.metadata = {"clip_id": "clip", "language": "en"}
        self.turns = [
            {
                "start": 0.0,
                "end": 1.0,
                "speaker": "SPEAKER_A",
                "confidence": 1.0,
            }
        ]

    def test_all_variants_build_without_external_models(self) -> None:
        maps = {
            variant: build_workflow_variant(
                variant,
                self.metadata,
                _segments(),
                self.turns,
                [],
                GLOSSARY,
                {"use_api": False},
            )
            for variant in VARIANT_NAMES
        }

        self.assertEqual(len(maps["asr_only"].anchors), 1)
        self.assertEqual(
            maps["temporal_anchor_only"].anchors[0].speaker,
            "UNKNOWN",
        )
        self.assertEqual(
            maps["reference_speaker_time"].anchors[0].speaker,
            "SPEAKER_A",
        )
        self.assertFalse(maps["reference_speaker_time"].events)
        self.assertFalse(maps["term_rescue"].correction_audits)
        self.assertTrue(maps["term_rescue"].term_rescues)
        self.assertTrue(
            maps["constrained_correction"].correction_audits
        )
        self.assertTrue(maps["full_talkweaver"].speaker_cards)
        self.assertEqual(
            maps["full_talkweaver"].summary["mode"],
            "deterministic_extractive",
        )

    def test_overlap_variant_marks_simultaneous_speech(self) -> None:
        turns = [
            {"start": 0.0, "end": 0.8, "speaker": "A"},
            {"start": 0.4, "end": 1.0, "speaker": "B"},
        ]

        result = build_overlap_aware_variant(
            self.metadata,
            _segments("one two three four"),
            turns,
            [],
        )

        self.assertTrue(result.events)
        self.assertTrue(any(anchor.overlap for anchor in result.anchors))

    def test_term_rescue_does_not_modify_raw_text(self) -> None:
        result = build_term_rescue_variant(
            self.metadata,
            _segments(),
            self.turns,
            [],
            GLOSSARY,
        )

        self.assertTrue(
            any(
                candidate.canonical.lower() == "pyannote"
                for candidate in result.term_rescues
            )
        )
        self.assertTrue(
            all(not anchor.corrected_text for anchor in result.anchors)
        )

    def test_constrained_and_full_variants_add_audit_layers(self) -> None:
        corrected = build_constrained_correction_variant(
            self.metadata,
            _segments(),
            self.turns,
            [],
            GLOSSARY,
            {"use_api": False},
        )
        full = build_full_talkweaver_variant(
            self.metadata,
            _segments(),
            self.turns,
            [],
            GLOSSARY,
            {"use_api": False},
        )

        self.assertIn("pyannote", corrected.anchors[0].corrected_text)
        self.assertEqual(
            len(corrected.correction_audits),
            len(corrected.anchors),
        )
        self.assertTrue(full.speaker_cards)
        self.assertTrue(full.summary["summary"])


class AblationRunnerTests(unittest.TestCase):
    def test_corrected_scoring_reuses_phase2c_metric_policy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions = root / "predictions"
            predictions.mkdir()
            anchors = root / "anchors.json"
            events = root / "events.json"
            terms = root / "terms.json"
            manifest = root / "manifest.csv"
            output = root / "results.csv"
            maps = root / "maps"
            anchors.write_text(
                json.dumps(
                    [
                        {
                            "start": 0.0,
                            "end": 1.0,
                            "speaker": "A",
                            "text": "pyannote",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            events.write_text("[]", encoding="utf-8")
            terms.write_text("[]", encoding="utf-8")
            row = {
                "clip_id": "clip",
                "dataset_name": "Demo",
                "language": "en",
                "anchors_path": str(anchors),
                "events_path": str(events),
                "terms_path": str(terms),
            }
            with manifest.open(
                "w",
                encoding="utf-8",
                newline="",
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
            (predictions / "base__clip.json").write_text(
                json.dumps(
                    {
                        "is_mock": False,
                        "clip_id": "clip",
                        "dataset_name": "Demo",
                        "language": "en",
                        "model_name": "base",
                        "reference_text": "pyannote",
                        "hypothesis_text": "piano note",
                        "metric_name": "WER",
                        "error_rate": 2.0,
                        "asr": {
                            "asr_mode": "real",
                            "segments": _segments("piano note"),
                        },
                    }
                ),
                encoding="utf-8",
            )

            rows = run_ablation(
                manifest=manifest,
                predictions_dir=predictions,
                asr_model="base",
                output=output,
                maps_dir=maps,
                variants=["constrained_correction"],
            )

            expected = evaluate_text("pyannote", "pyannote", "en")
            self.assertEqual(len(rows), 1)
            self.assertEqual(
                rows[0]["corrected_error_rate"],
                expected["error_rate"],
            )
            self.assertEqual(rows[0]["uses_real_asr_prediction"], "true")
            self.assertEqual(rows[0]["num_term_rescues_applied"], 1)

    def test_unchanged_correction_preserves_original_asr_score(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions = root / "predictions"
            predictions.mkdir()
            anchors = root / "anchors.json"
            events = root / "events.json"
            terms = root / "terms.json"
            manifest = root / "manifest.csv"
            anchors.write_text(
                json.dumps(
                    [
                        {
                            "start": 0.0,
                            "end": 1.0,
                            "speaker": "A",
                            "text": "bonjour monde",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            events.write_text("[]", encoding="utf-8")
            terms.write_text("[]", encoding="utf-8")
            row = {
                "clip_id": "clip",
                "dataset_name": "Demo",
                "language": "fr",
                "anchors_path": str(anchors),
                "events_path": str(events),
                "terms_path": str(terms),
            }
            with manifest.open(
                "w",
                encoding="utf-8",
                newline="",
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
            (predictions / "base__clip.json").write_text(
                json.dumps(
                    {
                        "is_mock": False,
                        "clip_id": "clip",
                        "dataset_name": "Demo",
                        "language": "fr",
                        "model_name": "base",
                        "reference_text": "bonjour monde",
                        "hypothesis_text": "bonjour, monde",
                        "metric_name": "WER",
                        "error_rate": 0.125,
                        "asr": {
                            "asr_mode": "real",
                            "segments": _segments("bonjour, monde"),
                        },
                    }
                ),
                encoding="utf-8",
            )

            rows = run_ablation(
                manifest=manifest,
                predictions_dir=predictions,
                asr_model="base",
                output=root / "results.csv",
                maps_dir=root / "maps",
                variants=["constrained_correction"],
            )

            self.assertEqual(rows[0]["corrected_error_rate"], 0.125)


class SummaryAndPlotTests(unittest.TestCase):
    def _write_rows(self, path: Path) -> None:
        rows = [
            {
                "variant": "asr_only",
                "dataset_name": "Demo",
                "language": "en",
                "num_anchors": "1",
                "num_speaker_labeled_anchors": "0",
                "num_overlap_anchors": "0",
                "num_events": "0",
                "num_term_candidates": "0",
                "num_term_rescues_applied": "0",
                "num_correction_audits": "0",
                "num_unsupported_changes": "0",
                "num_needs_review": "0",
                "asr_error_rate": "0.5",
                "corrected_error_rate": "",
                "anchor_coverage": "1.0",
            },
            {
                "variant": "full_talkweaver",
                "dataset_name": "Demo",
                "language": "en",
                "num_anchors": "3",
                "num_speaker_labeled_anchors": "2",
                "num_overlap_anchors": "1",
                "num_events": "1",
                "num_term_candidates": "1",
                "num_term_rescues_applied": "1",
                "num_correction_audits": "3",
                "num_unsupported_changes": "0",
                "num_needs_review": "1",
                "asr_error_rate": "0.5",
                "corrected_error_rate": "0.25",
                "anchor_coverage": "1.0",
            },
        ]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_summary_and_charts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.csv"
            summary = root / "summary.csv"
            charts = root / "charts"
            self._write_rows(source)

            summaries = summarize_results(source, summary)
            outputs = plot_results(source, charts)

            self.assertEqual(len(summaries), 2)
            self.assertTrue(summary.is_file())
            self.assertEqual(len(outputs), 2)
            self.assertTrue(
                (charts / "workflow_ablation_completeness.png").is_file()
            )
            self.assertTrue(
                (charts / "workflow_ablation_review_flags.png").is_file()
            )


class CliTests(unittest.TestCase):
    def test_ablation_help(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "experiments" / "run_workflow_ablation.py"),
                "--help",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--variants", result.stdout)


if __name__ == "__main__":
    unittest.main()
