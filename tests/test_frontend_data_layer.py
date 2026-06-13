"""Network-free tests for the AI Meeting Detective frontend foundation."""

from __future__ import annotations

import importlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from webapp.data_loader import (
    discover_charts,
    frame_warning,
    get_best_term_rescue_examples,
    get_best_available_demo_clip,
    load_overlap_safety_cases,
    list_available_conversation_maps,
    load_asr_summary,
    load_conversation_map,
    load_term_rescue_cases,
)
from webapp.report_export import build_detective_report, export_detective_report
from webapp.ui_components import render_text_diff


def _sample_map(clip_id: str = "sample_case") -> dict:
    return {
        "clip_id": clip_id,
        "metadata": {
            "dataset_name": "Controlled Test",
            "language": "en",
            "duration_seconds": 3.2,
            "asr_mode": "real_prediction_json",
            "diarization_mode": "reference",
            "llm_mode": "rule_fallback",
            "reference_assisted": True,
            "is_mock": False,
        },
        "anchors": [
            {
                "anchor_id": f"{clip_id}_anchor_001",
                "start": 0.0,
                "end": 3.2,
                "speaker": "OVERLAP",
                "speakers": ["SPEAKER_00", "SPEAKER_01"],
                "raw_text": "we use piano note",
                "corrected_text": "we use pyannote",
                "overlap": True,
                "interruption": False,
                "confidence": 0.55,
                "retrieved_terms": ["pyannote"],
                "unsupported_changes": [],
                "needs_review": True,
                "notes": ["Overlap evidence requires review."],
            }
        ],
        "events": [
            {
                "event_id": "event_001",
                "type": "overlap",
                "start": 1.0,
                "end": 1.4,
                "speakers": ["SPEAKER_00", "SPEAKER_01"],
                "description": "Simultaneous speech.",
            }
        ],
        "term_rescues": [
            {
                "canonical": "pyannote",
                "retrieval_method": "fused",
                "retrieved_score": 1.0,
                "evidence_anchor_ids": [f"{clip_id}_anchor_001"],
            }
        ],
        "correction_audits": [
            {
                "anchor_id": f"{clip_id}_anchor_001",
                "raw_text": "we use piano note",
                "corrected_text": "we use pyannote",
                "unsupported_changes": [],
                "hallucination_risk": "medium",
                "needs_review": True,
                "api_used": False,
                "fallback_used": False,
            }
        ],
        "speaker_cards": [],
        "summary": {},
    }


class FrontendDataLoaderTests(unittest.TestCase):
    def test_missing_files_return_actionable_empty_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame = load_asr_summary(root / "missing.csv")
            payload = load_conversation_map(root / "missing.json")

        self.assertTrue(frame.empty)
        self.assertIn("not available", frame_warning(frame))
        self.assertIn("_warning", payload)

    def test_loads_and_selects_sample_conversation_map(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "plain" / "plain_conversation_map.json"
            first.parent.mkdir(parents=True)
            first.write_text(json.dumps(_sample_map("plain")), encoding="utf-8")

            best = root / "ablation_real" / "full_talkweaver" / "ami_case_conversation_map.json"
            best.parent.mkdir(parents=True)
            payload = _sample_map("ami_case")
            payload["metadata"]["dataset_name"] = "AMI Meeting Corpus"
            payload["metadata"]["variant"] = "full_talkweaver"
            payload["metadata"]["uses_real_asr_prediction"] = True
            best.write_text(json.dumps(payload), encoding="utf-8")

            paths = list_available_conversation_maps(root)
            selected = get_best_available_demo_clip(root)
            loaded = load_conversation_map(selected)

        self.assertEqual(len(paths), 2)
        self.assertEqual(selected, best.resolve())
        self.assertEqual(loaded["clip_id"], "ami_case")
        self.assertTrue(loaded["_source_path"].endswith(best.name))

    def test_chart_discovery_only_returns_existing_images(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "present.png").write_bytes(b"png")
            charts = discover_charts(
                ["present.png", "missing.png"],
                chart_root=root,
            )

        self.assertEqual(set(charts), {"present.png"})

    def test_controlled_case_loaders_read_committed_results(self) -> None:
        term_cases = load_term_rescue_cases()
        overlap_cases = load_overlap_safety_cases()

        self.assertFalse(term_cases.empty)
        self.assertFalse(overlap_cases.empty)
        self.assertIn("raw_asr_text", term_cases)
        self.assertIn("correction_rejected", overlap_cases)

    def test_curated_term_examples_include_visible_corrections(self) -> None:
        examples = get_best_term_rescue_examples()

        self.assertFalse(examples.empty)
        self.assertTrue(
            examples["raw_asr_text"].ne(examples["corrected_text"]).all()
        )
        text = " ".join(examples["raw_asr_text"].astype(str)).lower()
        self.assertIn("piano note", text)
        self.assertIn("temporal anger", text)


class DetectiveReportTests(unittest.TestCase):
    def test_report_export_creates_markdown(self) -> None:
        conversation_map = _sample_map("case/unsafe name")
        with tempfile.TemporaryDirectory() as directory:
            report = build_detective_report(conversation_map, ["chart.png"])
            path = export_detective_report(conversation_map, directory, ["chart.png"])
            saved = path.read_text(encoding="utf-8")

        self.assertEqual(path.name, "case_unsafe_name_detective_report.md")
        self.assertIn("TalkWeaver Detective Report", report)
        self.assertIn("Overlap And Interruption Warnings", saved)
        self.assertIn("pyannote", saved)

    def test_app_import_does_not_start_streamlit_main(self) -> None:
        module = importlib.import_module("webapp.app")
        self.assertTrue(callable(module.main))


class TextDiffTests(unittest.TestCase):
    @staticmethod
    def _streamlit_mock() -> MagicMock:
        streamlit = MagicMock()
        streamlit.columns.return_value = (MagicMock(), MagicMock())
        return streamlit

    def test_render_text_diff_handles_identical_text(self) -> None:
        streamlit = self._streamlit_mock()
        with patch("webapp.ui_components.st", streamlit):
            result = render_text_diff("same evidence", "same evidence")

        self.assertTrue(result["identical"])
        self.assertEqual(result["changes"], [])
        streamlit.info.assert_called_once()

    def test_render_text_diff_highlights_changed_text(self) -> None:
        streamlit = self._streamlit_mock()
        with patch("webapp.ui_components.st", streamlit):
            result = render_text_diff(
                "we use piano note",
                "we use pyannote",
            )

        self.assertFalse(result["identical"])
        self.assertEqual(result["changes"][0]["removed"], "piano note")
        self.assertEqual(result["changes"][0]["added"], "pyannote")
        self.assertIn("tw-diff-removed", result["raw_html"])
        self.assertIn("tw-diff-added", result["corrected_html"])
        streamlit.info.assert_not_called()


if __name__ == "__main__":
    unittest.main()
