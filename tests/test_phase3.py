"""Focused tests for Phase 3 diarization, overlap, and alignment."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.alignment import align_segments, align_words_to_speakers
from backend.asr import MOCK_ASR_SEGMENTS
from backend.diarization import diarize_with_metadata
from backend.export import (
    export_overlap_regions,
    export_temporal_anchor_transcript,
)
from backend.overlap import detect_overlap_regions
from backend.pipeline import run_pipeline


class DiarizationTests(unittest.TestCase):
    def test_mock_diarization_has_two_speakers_and_overlap(self) -> None:
        result = diarize_with_metadata(mock=True)
        speakers = {turn["speaker"] for turn in result["turns"]}
        regions = detect_overlap_regions(result["turns"])

        self.assertEqual(result["mode"], "mock_demo")
        self.assertEqual(speakers, {"SPEAKER_00", "SPEAKER_01"})
        self.assertEqual(
            regions,
            [
                {
                    "start": 3.0,
                    "end": 3.4,
                    "speakers": ["SPEAKER_00", "SPEAKER_01"],
                    "duration": 0.4,
                    "type": "overlap",
                }
            ],
        )

    def test_missing_token_falls_back_without_audio(self) -> None:
        result = diarize_with_metadata(hf_token="", fallback_to_mock=True)

        self.assertEqual(result["mode"], "mock_fallback")
        self.assertTrue(result["is_mock"])
        self.assertIn("HF_TOKEN", result["fallback_reason"])

    def test_missing_pyannote_falls_back_with_token(self) -> None:
        with patch.dict(sys.modules, {"pyannote.audio": None}):
            result = diarize_with_metadata(
                hf_token="test-token",
                fallback_to_mock=True,
            )

        self.assertEqual(result["mode"], "mock_fallback")
        self.assertIn("not installed", result["fallback_reason"])


class AlignmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.diarization = diarize_with_metadata(mock=True)

    def test_words_use_midpoint_and_mark_overlap(self) -> None:
        words = align_words_to_speakers(
            MOCK_ASR_SEGMENTS,
            self.diarization["turns"],
        )
        overlap_words = [word for word in words if word["overlap"]]

        self.assertEqual(len(overlap_words), 1)
        self.assertEqual(overlap_words[0]["word"], "The")
        self.assertEqual(overlap_words[0]["speaker"], "OVERLAP")
        self.assertEqual(
            overlap_words[0]["speakers"],
            ["SPEAKER_00", "SPEAKER_01"],
        )
        self.assertLessEqual(overlap_words[0]["confidence"], 0.6)

    def test_unknown_midpoint_uses_unknown_speaker(self) -> None:
        asr = [
            {
                "start": 10.0,
                "end": 10.5,
                "text": "Outside.",
                "words": [
                    {"word": "Outside.", "start": 10.0, "end": 10.5}
                ],
            }
        ]
        transcript = align_segments(asr, self.diarization["turns"])

        self.assertEqual(transcript[0]["speaker"], "UNKNOWN")
        self.assertEqual(transcript[0]["speakers"], [])
        self.assertEqual(transcript[0]["confidence"], 0.5)

    def test_temporal_anchor_structure_and_exports(self) -> None:
        transcript = align_segments(
            MOCK_ASR_SEGMENTS,
            self.diarization["turns"],
        )
        required = {
            "start",
            "end",
            "speaker",
            "speakers",
            "raw_text",
            "corrected_text",
            "overlap",
            "confidence",
            "retrieved_terms",
        }
        self.assertTrue(required.issubset(transcript[0]))
        self.assertTrue(
            all(segment["corrected_text"] == "" for segment in transcript)
        )
        self.assertTrue(
            all(segment["retrieved_terms"] == [] for segment in transcript)
        )

        regions = detect_overlap_regions(self.diarization["turns"])
        with tempfile.TemporaryDirectory() as directory:
            temporal_paths = export_temporal_anchor_transcript(
                directory,
                "mock",
                transcript,
                mode="mock_demo",
            )
            overlap_paths = export_overlap_regions(
                directory,
                regions,
                mode="mock_demo",
            )
            exported = json.loads(
                temporal_paths["json"].read_text(encoding="utf-8")
            )
            transcript_markdown = temporal_paths["markdown"].read_text(
                encoding="utf-8"
            )
            overlap_markdown = overlap_paths["markdown"].read_text(
                encoding="utf-8"
            )

        self.assertEqual(len(exported), 4)
        self.assertIn("OVERLAP - REVIEW REQUIRED", transcript_markdown)
        self.assertIn("SPEAKER_00, SPEAKER_01", overlap_markdown)


class PipelineTests(unittest.TestCase):
    def test_mock_pipeline_preserves_phase3_temporal_anchors(self) -> None:
        result = run_pipeline(mock=True)

        self.assertEqual(result["mode"], "mock_demo")
        self.assertEqual(len(result["overlap_regions"]), 1)
        self.assertTrue(
            all(
                segment["corrected_text"] == ""
                and segment["retrieved_terms"] == []
                for segment in result["temporal_transcript"]
            )
        )
        self.assertTrue(
            Path(result["artifacts"]["temporal_anchor_json"]).exists()
        )


if __name__ == "__main__":
    unittest.main()
