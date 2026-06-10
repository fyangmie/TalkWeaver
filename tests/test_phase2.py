"""Focused tests for Phase 2 preprocessing and ASR export."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from backend.asr import transcribe_with_metadata
from backend.export import export_raw_transcript
from backend.preprocessing import preprocess_audio


def _write_stereo_test_wav(path: Path) -> None:
    sample_rate = 8_000
    duration = 0.25
    times = np.arange(int(sample_rate * duration)) / sample_rate
    left = 0.2 * np.sin(2 * np.pi * 440 * times)
    right = 0.1 * np.sin(2 * np.pi * 660 * times)
    stereo = np.column_stack([left, right])
    pcm = np.round(stereo * 32_767).astype("<i2")
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(2)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())


class PreprocessingTests(unittest.TestCase):
    def test_mock_preprocessing_does_not_require_input_file(self) -> None:
        result = preprocess_audio("missing.wav", mock=True, denoise=True)
        self.assertEqual(result["mode"], "mock_demo")
        self.assertEqual(result["sample_rate"], 16_000)
        self.assertEqual(result["channels"], 1)
        self.assertFalse(result["denoise_applied"])

    def test_real_wav_becomes_normalized_mono_16k(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "stereo_8k.wav"
            output = root / "processed.wav"
            _write_stereo_test_wav(source)

            result = preprocess_audio(source, output_path=output)

            self.assertEqual(result["mode"], "real")
            self.assertEqual(result["input_sample_rate"], 8_000)
            self.assertEqual(result["input_channels"], 2)
            self.assertEqual(result["sample_rate"], 16_000)
            self.assertEqual(result["channels"], 1)
            self.assertAlmostEqual(
                result["peak_after_normalization"],
                0.95,
                places=3,
            )
            with wave.open(str(output), "rb") as processed:
                self.assertEqual(processed.getframerate(), 16_000)
                self.assertEqual(processed.getnchannels(), 1)
                self.assertEqual(processed.getsampwidth(), 2)

    def test_optional_denoising_is_applied_when_available(self) -> None:
        fake_module = SimpleNamespace(
            reduce_noise=lambda *, y, sr: y * 0.5,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.wav"
            output = root / "output.wav"
            _write_stereo_test_wav(source)
            with patch.dict(sys.modules, {"noisereduce": fake_module}):
                result = preprocess_audio(
                    source,
                    output_path=output,
                    denoise=True,
                )

        self.assertTrue(result["denoise_requested"])
        self.assertTrue(result["denoise_applied"])
        self.assertEqual(result["warnings"], [])

    def test_soundfile_loader_accepts_flac(self) -> None:
        import soundfile as sf

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.flac"
            output = root / "output.wav"
            sample_rate = 22_050
            times = np.arange(sample_rate // 10) / sample_rate
            stereo = np.column_stack(
                [
                    0.1 * np.sin(2 * np.pi * 220 * times),
                    0.1 * np.sin(2 * np.pi * 330 * times),
                ]
            )
            sf.write(source, stereo, sample_rate)

            result = preprocess_audio(source, output_path=output)

        self.assertEqual(result["loader"], "soundfile")
        self.assertEqual(result["input_sample_rate"], sample_rate)
        self.assertEqual(result["input_channels"], 2)


class ASRTests(unittest.TestCase):
    def test_mock_asr_exports_json_markdown_and_metadata(self) -> None:
        result = transcribe_with_metadata(mock=True, model_size="tiny")
        with tempfile.TemporaryDirectory() as directory:
            paths = export_raw_transcript(directory, "mock test", result)
            segments = json.loads(paths["json"].read_text(encoding="utf-8"))
            markdown = paths["markdown"].read_text(encoding="utf-8")
            metadata = json.loads(
                paths["metadata"].read_text(encoding="utf-8")
            )

        self.assertEqual(len(segments), 3)
        self.assertTrue(segments[0]["words"])
        self.assertIn("deterministic mock/demo output", markdown)
        self.assertEqual(metadata["mode"], "mock_demo")

    def test_missing_faster_whisper_falls_back_to_mock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            audio_path = Path(directory) / "audio.wav"
            _write_stereo_test_wav(audio_path)
            with patch.dict(sys.modules, {"faster_whisper": None}):
                result = transcribe_with_metadata(audio_path)

        self.assertEqual(result["mode"], "mock_fallback")
        self.assertIn("not installed", result["fallback_reason"])
        self.assertEqual(len(result["segments"]), 3)

    def test_faster_whisper_segments_and_words_are_serialized(self) -> None:
        word = SimpleNamespace(
            word=" TalkWeaver",
            start=0.1,
            end=0.5,
            probability=0.97,
        )
        segment = SimpleNamespace(
            start=0.1,
            end=0.8,
            text=" TalkWeaver works.",
            words=[word],
            avg_logprob=-0.2,
            no_speech_prob=0.01,
        )
        info = SimpleNamespace(
            language="en",
            language_probability=0.99,
            duration=1.0,
            duration_after_vad=0.8,
        )

        class FakeWhisperModel:
            def __init__(self, model_size: str, **kwargs: object) -> None:
                self.model_size = model_size
                self.kwargs = kwargs

            def transcribe(
                self,
                audio_path: str,
                **kwargs: object,
            ) -> tuple[object, object]:
                return (item for item in [segment]), info

        fake_module = SimpleNamespace(WhisperModel=FakeWhisperModel)
        with tempfile.TemporaryDirectory() as directory:
            audio_path = Path(directory) / "audio.wav"
            _write_stereo_test_wav(audio_path)
            with patch.dict(sys.modules, {"faster_whisper": fake_module}):
                result = transcribe_with_metadata(
                    audio_path,
                    model_size="tiny",
                )

        self.assertEqual(result["mode"], "faster_whisper")
        self.assertEqual(result["segments"][0]["text"], "TalkWeaver works.")
        self.assertEqual(
            result["segments"][0]["words"][0]["probability"],
            0.97,
        )


if __name__ == "__main__":
    unittest.main()
