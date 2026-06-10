"""Tests for reusable Phase 6 Streamlit helper logic."""

from __future__ import annotations

import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np
import pandas as pd

from webapp.components.audio_player import (
    get_audio_metadata,
    safe_upload_name,
    save_uploaded_audio,
)
from webapp.components.metrics_dashboard import (
    contains_mock_metrics,
    discover_result_csvs,
    load_result_frames,
    metric_snapshot,
)
from webapp.components.speaker_timeline import build_timeline_rows
from webapp.components.transcript_viewer import temporal_anchor_rows
from webapp.components.waveform_viewer import load_waveform_preview


def _wav_bytes(path: Path) -> bytes:
    sample_rate = 8_000
    times = np.arange(sample_rate // 10) / sample_rate
    audio = 0.2 * np.sin(2 * np.pi * 220 * times)
    pcm = np.round(audio * 32_767).astype("<i2")
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())
    return path.read_bytes()


class FakeUpload:
    def __init__(self, name: str, payload: bytes) -> None:
        self.name = name
        self._payload = payload

    def getbuffer(self) -> memoryview:
        return memoryview(self._payload)


class AudioHelperTests(unittest.TestCase):
    def test_upload_is_sanitized_persisted_and_decoded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.wav"
            payload = _wav_bytes(source)
            upload = FakeUpload("../../Team Meeting.wav", payload)

            saved = save_uploaded_audio(upload, root / "uploads")
            metadata = get_audio_metadata(saved)
            preview = load_waveform_preview(saved, max_points=100)

        self.assertEqual(safe_upload_name(upload.name), "Team_Meeting.wav")
        self.assertEqual(saved.name, "Team_Meeting.wav")
        self.assertEqual(metadata["sample_rate"], 8_000)
        self.assertEqual(metadata["channels"], 1)
        self.assertAlmostEqual(metadata["duration_seconds"], 0.1, places=2)
        self.assertLessEqual(len(preview["amplitude"]), 100)
        self.assertEqual(len(preview["time"]), len(preview["amplitude"]))


class ReviewHelperTests(unittest.TestCase):
    def test_timeline_adds_dedicated_overlap_lane(self) -> None:
        rows = build_timeline_rows(
            [
                {"start": 0.0, "end": 3.4, "speaker": "SPEAKER_00"},
                {"start": 3.0, "end": 6.5, "speaker": "SPEAKER_01"},
            ],
            [
                {
                    "start": 3.0,
                    "end": 3.4,
                    "speakers": ["SPEAKER_00", "SPEAKER_01"],
                }
            ],
        )

        overlap = [row for row in rows if row["type"] == "overlap"]
        self.assertEqual(len(overlap), 1)
        self.assertEqual(overlap[0]["lane"], "OVERLAP")
        self.assertEqual(overlap[0]["speakers"], "SPEAKER_00, SPEAKER_01")

    def test_temporal_table_keeps_required_fields(self) -> None:
        rows = temporal_anchor_rows(
            [
                {
                    "start": 3.0,
                    "end": 3.2,
                    "speaker": "OVERLAP",
                    "speakers": ["SPEAKER_00", "SPEAKER_01"],
                    "raw_text": "The",
                    "corrected_text": "The",
                    "overlap": True,
                    "confidence": 0.55,
                    "retrieved_terms": [],
                }
            ]
        )

        self.assertEqual(
            set(rows[0]),
            {
                "start",
                "end",
                "speaker",
                "speakers",
                "raw_text",
                "corrected_text",
                "overlap",
                "confidence",
                "retrieved_terms",
            },
        )
        self.assertEqual(rows[0]["speakers"], "SPEAKER_00, SPEAKER_01")


class MetricsHelperTests(unittest.TestCase):
    def test_mock_csv_is_loaded_without_fabricating_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_path = root / "results.csv"
            pd.DataFrame(
                [
                    {
                        "result_type": "mock_demo_not_measured",
                        "wer": "",
                        "speaker_error": "",
                        "term_error_rate": "",
                        "latency_seconds": "",
                    }
                ]
            ).to_csv(result_path, index=False)

            paths = discover_result_csvs([root])
            frame = load_result_frames(paths)
            snapshot = metric_snapshot(frame)

        self.assertEqual(paths, [result_path.resolve()])
        self.assertTrue(contains_mock_metrics(frame))
        self.assertTrue(all(value is None for value in snapshot.values()))


if __name__ == "__main__":
    unittest.main()
