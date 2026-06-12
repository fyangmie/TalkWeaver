"""Tests for the temporal-anchor evidence-grounded TalkWeaver workflow."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from backend.asr import MOCK_ASR_SEGMENTS
from backend.constrained_correction import (
    apply_constrained_correction,
    audit_correction,
)
from backend.conversation_map import (
    build_conversation_map,
    save_conversation_map,
)
from backend.diarization import build_mock_speaker_turns
from backend.events import (
    detect_interruption_events,
    detect_overlap_events,
)
from backend.schemas import (
    ASRWord,
    ConversationMap,
    SpeakerTurn,
    TemporalAnchor,
)
from backend.temporal_anchor import build_temporal_anchors
from backend.term_rescue import retrieve_term_candidates


ROOT = Path(__file__).resolve().parents[1]


class SchemaTests(unittest.TestCase):
    def test_schemas_are_json_serializable(self) -> None:
        word = ASRWord("hello", 0.0, 0.5, 0.9, "en")
        turn = SpeakerTurn("SPEAKER_00", 0.0, 1.0, 0.95, "test")
        payload = {"word": asdict(word), "turn": asdict(turn)}

        encoded = json.dumps(payload)

        self.assertIn("SPEAKER_00", encoded)
        self.assertIn("hello", encoded)


class TemporalAnchorTests(unittest.TestCase):
    def test_grouping_preserves_speaker_time_and_overlap(self) -> None:
        turns = build_mock_speaker_turns()
        overlap = [
            {
                "type": "overlap",
                "start": 3.0,
                "end": 3.4,
                "speakers": ["SPEAKER_00", "SPEAKER_01"],
            }
        ]

        anchors = build_temporal_anchors(
            MOCK_ASR_SEGMENTS,
            turns,
            overlap,
            "mock_clip",
            "en",
        )

        self.assertEqual(len(anchors), 4)
        self.assertEqual(anchors[0].start, 0.0)
        self.assertEqual(anchors[-1].end, 9.4)
        self.assertTrue(any(anchor.overlap for anchor in anchors))
        self.assertTrue(
            any(anchor.speaker == "OVERLAP" for anchor in anchors)
        )
        self.assertEqual(
            anchors[0].raw_text,
            "We use piano note for diary station.",
        )


class EventTests(unittest.TestCase):
    def test_overlap_detection(self) -> None:
        turns = [
            {"speaker": "A", "start": 0.0, "end": 2.0},
            {"speaker": "B", "start": 1.5, "end": 1.8},
        ]
        events = detect_overlap_events(turns, clip_id="clip")

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].type, "overlap")
        self.assertEqual(events[0].speakers, ["A", "B"])

    def test_interruption_requires_floor_takeover(self) -> None:
        turns = [
            {"speaker": "A", "start": 0.0, "end": 2.0},
            {"speaker": "B", "start": 1.5, "end": 3.0},
            {"speaker": "C", "start": 0.5, "end": 0.9},
        ]
        events = detect_interruption_events(
            turns,
            clip_id="clip",
            min_overlap_seconds=0.2,
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].speakers, ["A", "B"])
        self.assertIn("timing-based", " ".join(events[0].notes))


class RetrievalAndCorrectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.anchor = TemporalAnchor(
            anchor_id="clip_anchor_001",
            clip_id="clip",
            start=0.0,
            end=3.0,
            speaker="SPEAKER_00",
            speakers=["SPEAKER_00"],
            raw_text="We use piano note for diary station.",
            language="en",
            confidence=0.9,
            asr_confidence=0.9,
            diarization_confidence=0.9,
        )

    def test_term_rescue_connects_candidates_to_anchor(self) -> None:
        candidates = retrieve_term_candidates([self.anchor])

        canonicals = {candidate.canonical for candidate in candidates}
        self.assertIn("pyannote", canonicals)
        self.assertIn("diarization", canonicals)
        self.assertTrue(
            all(
                self.anchor.anchor_id in candidate.evidence_anchor_ids
                for candidate in candidates
            )
        )

    def test_rule_fallback_preserves_anchor_and_builds_audit(self) -> None:
        candidates = retrieve_term_candidates([self.anchor])
        anchors, audits, mode = apply_constrained_correction(
            [self.anchor],
            candidates,
            [],
            llm_config={"use_api": False},
        )

        self.assertEqual(
            anchors[0].corrected_text,
            "We use pyannote for diarization.",
        )
        self.assertEqual(anchors[0].speaker, "SPEAKER_00")
        self.assertIn("rule_based", mode)
        self.assertEqual(audits[0].unsupported_changes, [])

    def test_audit_flags_unsupported_large_change(self) -> None:
        audit = audit_correction(
            self.anchor,
            (
                "We use pyannote for diarization and approved a new revenue "
                "forecast for next quarter."
            ),
            candidates=retrieve_term_candidates([self.anchor]),
        )

        self.assertTrue(audit.needs_review)
        self.assertEqual(audit.hallucination_risk, "high")
        self.assertIn("revenue", audit.unsupported_changes)


class ConversationMapTests(unittest.TestCase):
    def test_conversation_map_serializes_to_json(self) -> None:
        asr = {"mode": "mock_demo", "segments": MOCK_ASR_SEGMENTS}
        diarization = {
            "mode": "mock_demo",
            "turns": build_mock_speaker_turns(),
        }
        conversation_map = build_conversation_map(
            {"clip_id": "mock_clip", "language": "en"},
            asr,
            diarization,
            [],
            ROOT / "docs" / "knowledge_base",
            {"use_api": False},
        )

        self.assertIsInstance(conversation_map, ConversationMap)
        self.assertEqual(conversation_map.metadata["asr_mode"], "mock")
        self.assertEqual(
            conversation_map.metadata["diarization_mode"], "mock"
        )
        self.assertEqual(
            conversation_map.metadata["llm_mode"], "rule_fallback"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = save_conversation_map(conversation_map, directory)
            exported = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(exported["clip_id"], "mock_clip")
        self.assertTrue(exported["anchors"])
        self.assertIn("correction_audits", exported)

    def test_cli_help_requires_no_models(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "run_talkweaver_workflow.py"),
                "--help",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--mock-models", result.stdout)


if __name__ == "__main__":
    unittest.main()
