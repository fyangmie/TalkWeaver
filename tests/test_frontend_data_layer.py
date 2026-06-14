"""Network-free tests for the AI Meeting Detective frontend foundation."""

from __future__ import annotations

import importlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from webapp.data_loader import (
    audio_available,
    build_speaker_evidence_cards,
    discover_charts,
    frame_warning,
    get_best_evidence_gate_model,
    get_best_term_rescue_examples,
    get_best_available_demo_clip,
    get_evidence_gate_examples,
    get_event_audio_window,
    load_evidence_gate_metrics,
    load_overlap_safety_cases,
    list_available_conversation_maps,
    load_asr_summary,
    load_conversation_map,
    load_term_rescue_cases,
    resolve_local_audio_path,
)
from webapp.report_export import build_detective_report, export_detective_report
from webapp.ui_components import render_text_diff


def _sample_map(clip_id: str = "sample_case") -> dict:
    return {
        "clip_id": clip_id,
        "metadata": {
            "audio_path": "data/raw/missing.wav",
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

    def test_evidence_gate_artifacts_are_loadable(self) -> None:
        metrics = load_evidence_gate_metrics()
        best = get_best_evidence_gate_model()
        examples = get_evidence_gate_examples()

        self.assertFalse(metrics.empty)
        self.assertIsNotNone(best)
        self.assertIn(best["model_name"], metrics["model_name"].tolist())
        self.assertFalse(examples["accept"].empty)
        self.assertFalse(examples["reject"].empty)
        self.assertFalse(examples["needs_review"].empty)

    def test_evidence_gate_missing_results_are_graceful(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics = load_evidence_gate_metrics(root / "missing_metrics.csv")
            examples = get_evidence_gate_examples(
                root / "missing_predictions.csv",
                root / "missing_metrics.csv",
            )

        self.assertTrue(metrics.empty)
        self.assertTrue(all(frame.empty for frame in examples.values()))

    def test_curated_term_examples_include_visible_corrections(self) -> None:
        examples = get_best_term_rescue_examples()

        self.assertFalse(examples.empty)
        self.assertTrue(
            examples["raw_asr_text"].ne(examples["corrected_text"]).all()
        )
        text = " ".join(examples["raw_asr_text"].astype(str)).lower()
        self.assertIn("piano note", text)
        self.assertIn("temporal anger", text)

    def test_audio_path_resolver_handles_missing_audio(self) -> None:
        conversation_map = _sample_map()
        with tempfile.TemporaryDirectory() as directory:
            resolved = resolve_local_audio_path(conversation_map, directory)
            available = audio_available(conversation_map, directory)

        self.assertIsNone(resolved)
        self.assertFalse(available)

    def test_event_audio_window_is_padded_and_nonnegative(self) -> None:
        window = get_event_audio_window(
            {"start": 0.2, "end": 1.0},
            padding_seconds=0.5,
            duration_seconds=1.2,
        )

        self.assertEqual(window["start"], 0.0)
        self.assertEqual(window["end"], 1.2)
        self.assertEqual(window["duration"], 1.2)

    def test_speaker_cards_are_derived_from_anchor_evidence(self) -> None:
        cards = build_speaker_evidence_cards(_sample_map())

        self.assertEqual(
            {card["speaker_id"] for card in cards},
            {"SPEAKER_00", "SPEAKER_01"},
        )
        for card in cards:
            self.assertEqual(card["num_anchors"], 1)
            self.assertEqual(card["num_overlap_anchors"], 1)
            self.assertEqual(card["needs_review_anchors"], 1)
            self.assertEqual(
                card["evidence_anchor_ids"],
                ["sample_case_anchor_001"],
            )


class DetectiveReportTests(unittest.TestCase):
    def test_report_export_creates_markdown(self) -> None:
        conversation_map = _sample_map("case/unsafe name")
        with tempfile.TemporaryDirectory() as directory:
            report = build_detective_report(conversation_map, ["chart.png"])
            path = export_detective_report(conversation_map, directory, ["chart.png"])
            saved = path.read_text(encoding="utf-8")

        self.assertEqual(path.name, "case_unsafe_name_detective_report.md")
        self.assertIn("TalkWeaver Detective Report", report)
        self.assertIn("Event Investigation", saved)
        self.assertIn("Speaker Evidence Cards", saved)
        self.assertIn("event_001", saved)
        self.assertIn("SPEAKER_00", saved)
        self.assertIn("Audio window", saved)
        self.assertIn("pyannote", saved)
        self.assertIn("EvidenceGate Safety Model", saved)

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
