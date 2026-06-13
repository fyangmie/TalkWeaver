"""Tests for RAG retrieval, structured correction, and summarization."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.alignment import align_segments
from backend.asr import MOCK_ASR_SEGMENTS
from backend.diarization import diarize_with_metadata
from backend.llm_correction import (
    correct_segments,
    validate_corrected_text,
)
from backend.pipeline import run_pipeline
from backend.prompting import format_segment_prompt
from backend.rag import (
    TfidfKnowledgeBase,
    enrich_segments_with_terms,
    load_knowledge_base,
)
from backend.summarizer import answer_question, summarize_segments


class RetrievalTests(unittest.TestCase):
    def test_loads_all_local_markdown_documents(self) -> None:
        documents = load_knowledge_base()
        sources = {document.source for document in documents}

        self.assertEqual(len(documents), 5)
        self.assertIn("domain_terms.md", sources)
        self.assertIn("diarization_background.md", sources)

    def test_tfidf_retrieval_recovers_confusion_terms(self) -> None:
        index = TfidfKnowledgeBase()

        first = index.retrieve(
            "We use piano note for diary station."
        )["terms"]
        overlap = index.retrieve("The")["terms"]
        metrics = index.retrieve(
            "We should compare where and the ear."
        )["terms"]

        self.assertEqual(
            first,
            [
                "pyannote",
                "diarization",
                "pyannote.audio",
                "speaker diarization",
            ],
        )
        self.assertEqual(overlap, [])
        self.assertEqual(metrics[:2], ["WER", "DER"])


class PromptAndCorrectionTests(unittest.TestCase):
    def setUp(self) -> None:
        turns = diarize_with_metadata(mock=True)["turns"]
        temporal = align_segments(MOCK_ASR_SEGMENTS, turns)
        self.enriched, _metadata = enrich_segments_with_terms(temporal)

    def test_prompt_contains_diarization_structure(self) -> None:
        prompt = format_segment_prompt(self.enriched[0])

        self.assertIn("[00:00.00-00:03.20]", prompt)
        self.assertIn("SPEAKER_00", prompt)
        self.assertIn("overlap=false", prompt)
        self.assertIn("confidence=0.90", prompt)
        self.assertIn("pyannote", prompt)
        self.assertIn("Raw: We use piano note", prompt)

    def test_rule_correction_preserves_temporal_anchor_fields(self) -> None:
        corrected = correct_segments(self.enriched, mock=True)

        self.assertEqual(
            corrected[0]["corrected_text"],
            "We use pyannote for diarization.",
        )
        self.assertEqual(
            corrected[2]["corrected_text"],
            "RAG glossary can reduce term errors.",
        )
        self.assertEqual(
            corrected[3]["corrected_text"],
            "We should compare WER and DER.",
        )
        for before, after in zip(self.enriched, corrected):
            self.assertEqual(after["start"], before["start"])
            self.assertEqual(after["end"], before["end"])
            self.assertEqual(after["speaker"], before["speaker"])
            self.assertEqual(after["speakers"], before["speakers"])

        self.assertTrue(corrected[1]["correction_uncertain"])
        self.assertEqual(corrected[1]["corrected_text"], "The")
        self.assertEqual(corrected[1]["uncertainty"], "uncertain")

    def test_hallucination_validator_rejects_new_facts(self) -> None:
        valid, reason = validate_corrected_text(
            "The rack system improves where.",
            "The RAG system improves WER and revenue.",
            ["RAG", "WER"],
        )

        self.assertFalse(valid)
        self.assertIn("revenue", reason)

        reordered, reorder_reason = validate_corrected_text(
            "The rack system improves where.",
            "WER improves the RAG system.",
            ["RAG", "WER"],
        )
        self.assertFalse(reordered)
        self.assertIn("word order", reorder_reason)

    def test_api_result_is_validated_and_anchor_is_preserved(self) -> None:
        response = {
            "corrected_text": "We use pyannote for diarization.",
            "uncertain": False,
            "note": "Glossary-supported substitutions only.",
        }
        with patch(
            "backend.llm_correction._chat_completion",
            return_value=response,
        ):
            corrected = correct_segments(
                [self.enriched[0]],
                mock=False,
                openai_api_key="test-key",
            )

        self.assertEqual(
            corrected[0]["correction_mode"],
            "llm_with_rule_fallback",
        )
        self.assertEqual(
            corrected[0]["correction_backend_mode"],
            "api_openai",
        )
        self.assertTrue(corrected[0]["api_used"])
        self.assertFalse(corrected[0]["fallback_used"])
        self.assertEqual(corrected[0]["start"], self.enriched[0]["start"])
        self.assertEqual(corrected[0]["speaker"], "SPEAKER_00")

    def test_no_api_key_uses_rule_based_fallback(self) -> None:
        corrected = correct_segments([self.enriched[2]], mock=False)

        self.assertEqual(
            corrected[0]["correction_mode"],
            "rule_fallback",
        )
        self.assertEqual(
            corrected[0]["corrected_text"],
            "RAG glossary can reduce term errors.",
        )


class SummaryAndPipelineTests(unittest.TestCase):
    def test_summary_is_extractive_and_action_is_sourced(self) -> None:
        result = run_pipeline(mock=True)
        summary = summarize_segments(result["transcript"])

        self.assertIn(
            "We should compare WER and DER.",
            summary["summary"],
        )
        self.assertEqual(summary["action_items"][0]["speaker"], "SPEAKER_00")
        self.assertEqual(
            summary["action_items"][0]["text"],
            "Compare WER and DER.",
        )
        answer = answer_question(
            "What should we compare?",
            result["transcript"],
        )
        self.assertIn("WER and DER", answer["answer"])
        self.assertEqual(answer["source"]["speaker"], "SPEAKER_00")

    def test_pipeline_exports_corrected_transcript_and_summary(self) -> None:
        result = run_pipeline(mock=True)

        self.assertEqual(result["summary"]["mode"], "deterministic_extractive")
        self.assertTrue(
            Path(result["artifacts"]["corrected_transcript_json"]).exists()
        )
        self.assertTrue(Path(result["artifacts"]["summary_json"]).exists())
        self.assertEqual(
            result["transcript"][0]["corrected_text"],
            "We use pyannote for diarization.",
        )

    def test_knowledge_base_requires_markdown_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                load_knowledge_base(directory)


if __name__ == "__main__":
    unittest.main()
