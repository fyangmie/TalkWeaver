"""Offline tests for optional dependency setup and real-ASR failure policy."""

from __future__ import annotations

import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from backend.asr import (
    REAL_ASR_DEPENDENCY_ERROR,
    transcribe_with_metadata,
)
from scripts import run_talkweaver_workflow as workflow


ROOT = Path(__file__).resolve().parents[1]
CHECK_SCRIPT = ROOT / "scripts" / "check_optional_dependencies.py"


def _write_tiny_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(b"\x00\x00" * 160)


class DependencyCheckTests(unittest.TestCase):
    def test_help_requires_no_optional_packages(self) -> None:
        result = subprocess.run(
            [sys.executable, str(CHECK_SCRIPT), "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--strict", result.stdout)

    def test_non_strict_mode_succeeds_when_optional_deps_missing(self) -> None:
        result = subprocess.run(
            [sys.executable, str(CHECK_SCRIPT)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("optional dependency check", result.stdout.lower())

    def test_strict_mode_fails_for_deliberately_missing_package(self) -> None:
        package = "talkweaver-package-that-does-not-exist"
        result = subprocess.run(
            [sys.executable, str(CHECK_SCRIPT), "--strict", package],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(package, result.stderr)


class RealAsrFailureTests(unittest.TestCase):
    def test_missing_real_asr_dependency_never_creates_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "audio.wav"
            output = root / "fake_real_output.json"
            _write_tiny_wav(audio)
            with patch.dict(sys.modules, {"faster_whisper": None}):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "Real ASR requested but faster-whisper is not installed",
                ) as caught:
                    transcribe_with_metadata(
                        audio,
                        mock=False,
                        fallback_to_mock=False,
                    )

            self.assertEqual(str(caught.exception), REAL_ASR_DEPENDENCY_ERROR)
            self.assertFalse(output.exists())

    def test_workflow_reports_missing_dependency_without_fake_output(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "audio.wav"
            anchors = root / "anchors.json"
            events = root / "events.json"
            output_dir = root / "conversation_maps"
            _write_tiny_wav(audio)
            anchors.write_text(
                json.dumps(
                    [
                        {
                            "start": 0.0,
                            "end": 0.01,
                            "speaker": "SPEAKER_00",
                            "text": "Reference text.",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            events.write_text("[]", encoding="utf-8")
            row = {
                "clip_id": "dependency_test",
                "audio_path": str(audio),
                "anchors_path": str(anchors),
                "events_path": str(events),
                "language": "en",
                "duration_seconds": "0.01",
            }
            stderr = io.StringIO()
            argv = [
                "run_talkweaver_workflow.py",
                "--manifest",
                str(root / "manifest.csv"),
                "--clip-id",
                row["clip_id"],
                "--diarization-source",
                "reference",
                "--output",
                str(output_dir),
            ]
            with (
                patch.object(
                    workflow,
                    "load_manifest_row",
                    return_value=row,
                ),
                patch.object(
                    workflow,
                    "transcribe_with_metadata",
                    side_effect=RuntimeError(REAL_ASR_DEPENDENCY_ERROR),
                ),
                patch.object(workflow, "save_conversation_map") as save_mock,
                patch.object(sys, "argv", argv),
                contextlib.redirect_stderr(stderr),
            ):
                return_code = workflow.main()

            self.assertEqual(return_code, 2)
            self.assertIn(REAL_ASR_DEPENDENCY_ERROR, stderr.getvalue())
            save_mock.assert_not_called()
            self.assertFalse(output_dir.exists())


if __name__ == "__main__":
    unittest.main()
