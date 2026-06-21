"""Offline tests for Phase 2A-REAL manifest tooling."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path

from experiments.validate_manifest import validate_manifest
from scripts.build_formal_eval_manifest import build_combined_manifest
from scripts.dataset_utils import MANIFEST_COLUMNS, write_manifest
from scripts.download_earnings22_subset import _unique_file_ids
from scripts.download_mandarin_meeting_subset import (
    prepare_local_mandarin_meeting_subset,
)


ROOT = Path(__file__).resolve().parents[1]


def _complete_row(root: Path, clip_id: str = "tiny_001") -> dict[str, str]:
    audio = root / "audio.wav"
    transcript = root / "reference.txt"
    anchors = root / "anchors.json"
    terms = root / "terms.json"
    events = root / "events.json"
    audio.write_bytes(b"RIFFtiny-test")
    transcript.write_text("a tiny reference\n", encoding="utf-8")
    for path in (anchors, terms, events):
        path.write_text(json.dumps([]), encoding="utf-8")
    row = {column: "" for column in MANIFEST_COLUMNS}
    row.update(
        {
            "clip_id": clip_id,
            "audio_path": str(audio),
            "source_type": "public_dataset",
            "dataset_name": "tiny",
            "dataset_version": "test",
            "split": "test",
            "language": "en",
            "duration_seconds": "1.0",
            "speaker_count": "1",
            "has_overlap": "false",
            "has_interruptions": "false",
            "has_domain_terms": "false",
            "recording_device": "test",
            "noise_condition": "clean",
            "consent_status": "dataset_terms",
            "redistribution_status": "test_only",
            "license_or_access": "test fixture",
            "transcript_path": str(transcript),
            "anchors_path": str(anchors),
            "terms_path": str(terms),
            "events_path": str(events),
            "download_status": "prepared",
            "notes": "offline unit test",
        }
    )
    return row


class ManifestValidationTests(unittest.TestCase):
    def test_valid_local_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.csv"
            write_manifest(manifest, [_complete_row(root)])
            self.assertEqual(
                validate_manifest(
                    manifest,
                    require_real_files=True,
                    repo_root=root,
                ),
                [],
            )

    def test_missing_audio_fails_real_file_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = _complete_row(root)
            Path(row["audio_path"]).unlink()
            manifest = root / "manifest.csv"
            write_manifest(manifest, [row])
            errors = validate_manifest(
                manifest,
                require_real_files=True,
                repo_root=root,
            )
            self.assertTrue(any("audio_path does not exist" in error for error in errors))

    def test_metadata_reference_json_objects_are_valid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = _complete_row(root)
            Path(row["anchors_path"]).write_text(
                json.dumps({"tokens": []}),
                encoding="utf-8",
            )
            Path(row["events_path"]).write_text(
                json.dumps({"overlap_events": [], "interruption_events": []}),
                encoding="utf-8",
            )
            Path(row["terms_path"]).write_text(
                json.dumps({"terms": []}),
                encoding="utf-8",
            )
            manifest = root / "manifest.csv"
            write_manifest(manifest, [row])
            self.assertEqual(
                validate_manifest(
                    manifest,
                    require_real_files=True,
                    repo_root=root,
                ),
                [],
            )

    def test_combined_manifest_skips_incomplete_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            complete = _complete_row(root, "complete")
            missing = dict(complete)
            missing["clip_id"] = "missing"
            missing["audio_path"] = str(root / "missing.wav")
            source = root / "source.csv"
            output = root / "combined.csv"
            write_manifest(source, [complete, missing])

            rows, skipped = build_combined_manifest(
                [source],
                output,
                repo_root=root,
            )

            self.assertEqual([row["clip_id"] for row in rows], ["complete"])
            self.assertTrue(any("missing audio_path" in item for item in skipped))

    def test_unique_file_ids_deduplicates_and_drops_blank_values(self) -> None:
        self.assertEqual(
            _unique_file_ids(["4453225", "", "4467434", "4453225", " 4481221 "]),
            ["4453225", "4467434", "4481221"],
        )

    def test_local_mandarin_meeting_import_creates_valid_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio_root = root / "audio"
            transcript_root = root / "transcripts"
            reference_root = root / "references"
            manifest = root / "manifest.csv"
            audio_root.mkdir()
            transcript_root.mkdir()
            audio = audio_root / "clip001.wav"
            with wave.open(str(audio), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16000)
                handle.writeframes(b"\x00\x00" * 16000)
            (transcript_root / "clip001.txt").write_text(
                "大家好 我们开始会议\n",
                encoding="utf-8",
            )

            rows = prepare_local_mandarin_meeting_subset(
                local_audio_root=audio_root,
                local_transcript_root=transcript_root,
                reference_root=reference_root,
                manifest_out=manifest,
                max_clips=3,
            )

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["language"], "zh-CN")
            self.assertEqual(
                validate_manifest(
                    manifest,
                    require_real_files=True,
                    repo_root=ROOT,
                ),
                [],
            )


class DownloaderHelpTests(unittest.TestCase):
    def test_download_scripts_expose_help_without_network(self) -> None:
        for script in (
            "download_earnings22_subset.py",
            "download_common_voice_subset.py",
            "download_meeting_subset.py",
            "download_mandarin_meeting_subset.py",
        ):
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / script), "--help"],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("usage:", result.stdout.lower())


if __name__ == "__main__":
    unittest.main()
