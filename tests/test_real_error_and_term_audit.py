"""Tests for real-ASR error audit and term-rescue audit helpers."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.llm_config import LLMConfig
from backend.term_rescue import TermMatch
from backend.term_verifier import build_verifier_messages, verify_term_match
from experiments.audit_asr_errors import audit_asr_errors, audit_row
from experiments.evaluate_term_rescue_real_audit import (
    evaluate_term_rescue_real_audit,
    load_glossary,
)


ROOT = Path(__file__).resolve().parents[1]


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = list(
        dict.fromkeys(key for row in rows for key in row.keys())
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class AsrErrorAuditTests(unittest.TestCase):
    def test_named_entity_failure_is_tagged_as_term_candidate(self) -> None:
        row = {
            "clip_id": "fleurs_en_1578",
            "dataset_name": "Google FLEURS",
            "language": "en",
            "model_name": "tiny",
            "metric_name": "WER",
            "error_rate": "0.9",
            "reference_text": "Then, Lakkha Singh took the lead in singing the bhajans.",
            "hypothesis_text": "Then lock a sink to the lid and sink in the mushrooms.",
        }

        audited = audit_row(row)

        error_types = json.loads(audited["suspected_error_types"])
        candidates = json.loads(audited["candidate_terms_or_entities"])
        self.assertIn("proper_noun_or_named_entity", error_types)
        self.assertIn("Lakkha Singh", candidates)
        self.assertEqual(audited["term_eval_candidate"], "true")

    def test_audit_writes_sorted_csv(self) -> None:
        rows = [
            {
                "clip_id": "low",
                "dataset_name": "Google FLEURS",
                "language": "en",
                "model_name": "tiny",
                "metric_name": "WER",
                "error_rate": "0.1",
                "reference_text": "The system works.",
                "hypothesis_text": "The system works.",
            },
            {
                "clip_id": "high",
                "dataset_name": "AMI Meeting Corpus",
                "language": "en",
                "model_name": "tiny",
                "metric_name": "WER",
                "error_rate": "0.5",
                "cleaned_error_rate": "0.3",
                "reference_text": "Um this is the kick-off meeting for twenty five minutes.",
                "hypothesis_text": "This is the kickoff meeting.",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "asr.csv"
            output = root / "audit.csv"
            _write_csv(source, rows)

            audited = audit_asr_errors(source, output, min_error_rate=0.2)

            self.assertEqual([row["clip_id"] for row in audited], ["high"])
            self.assertTrue(output.exists())
            error_types = json.loads(audited[0]["suspected_error_types"])
            self.assertIn("meeting_disfluency_or_truncation", error_types)

    def test_cli_help_requires_no_data(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "experiments" / "audit_asr_errors.py"),
                "--help",
            ],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--min-error-rate", result.stdout)


class RealTermRescueAuditTests(unittest.TestCase):
    def test_rule_verifier_rejects_ambiguous_common_word_context(self) -> None:
        match = TermMatch(
            canonical="RAG",
            matched_form="rack",
            score=1.0,
            retrieval_method="exact_error_form",
            safe_to_apply=True,
            needs_review=False,
            context_reason="exact match",
            asr_error_forms=("rack",),
        )

        verification = verify_term_match(
            match,
            raw_text="Please put the rack on the table near the microphone.",
            context="meeting room setup",
        )

        self.assertEqual(verification.decision, "reject")
        self.assertFalse(verification.api_used)

    def test_llm_with_rule_fallback_runs_without_api_key(self) -> None:
        match = TermMatch(
            canonical="Lakkha Singh",
            matched_form="lock a sink",
            score=1.0,
            retrieval_method="exact_error_form",
            safe_to_apply=True,
            needs_review=False,
            context_reason="exact match",
            asr_error_forms=("lock a sink",),
        )

        verification = verify_term_match(
            match,
            raw_text="Then lock a sink led the bhajans.",
            terms_source="oracle_diagnostic",
            verifier="llm_with_rule_fallback",
            llm_config=None,
        )

        self.assertEqual(verification.decision, "accept")
        self.assertTrue(verification.fallback_used)
        self.assertFalse(verification.api_used)

    def test_llm_verifier_invalid_response_falls_back_when_allowed(self) -> None:
        match = TermMatch(
            canonical="Lakkha Singh",
            matched_form="lock a sink",
            score=1.0,
            retrieval_method="exact_error_form",
            safe_to_apply=True,
            needs_review=False,
            context_reason="exact match",
            asr_error_forms=("lock a sink",),
        )
        config = LLMConfig(
            provider="deepseek",
            api_key="test-key",
            model="deepseek-chat",
            base_url="https://api.deepseek.com",
        )

        with patch(
            "backend.term_verifier.request_json_completion",
            side_effect=ValueError("bad json"),
        ):
            verification = verify_term_match(
                match,
                raw_text="Then lock a sink led the bhajans.",
                terms_source="oracle_diagnostic",
                verifier="llm_with_rule_fallback",
                llm_config=config,
            )

        self.assertEqual(verification.decision, "accept")
        self.assertTrue(verification.fallback_used)
        self.assertFalse(verification.api_used)

    def test_llm_prompt_carries_explicit_language(self) -> None:
        match = TermMatch(
            canonical="巨龟",
            matched_form="聚会",
            score=1.0,
            retrieval_method="exact_error_form",
            safe_to_apply=True,
            needs_review=False,
            context_reason="exact match",
            asr_error_forms=("聚会",),
        )

        messages = build_verifier_messages(
            match,
            raw_text="使得聚会成为克隆群的主要食堂的物。",
            context="dataset=Google FLEURS; language=zh-CN",
            language="zh-CN",
            terms_source="oracle_diagnostic",
        )
        payload = json.loads(messages[1]["content"])

        self.assertEqual(payload["language"], "zh-CN")
        self.assertIn("Mandarin", payload["language_policy"])

    def test_markdown_glossary_loads_domain_terms_and_pairs(self) -> None:
        entries = load_glossary(ROOT / "docs" / "knowledge_base" / "domain_terms.md")
        canonicals = {entry.canonical for entry in entries}
        pyannote = next(entry for entry in entries if entry.canonical == "pyannote")

        self.assertIn("RAG", canonicals)
        self.assertIn("piano note", pyannote.asr_error_forms)

    def test_external_glossary_rescues_missing_terms(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "asr.csv"
            glossary = root / "terms.json"
            output = root / "term_audit.csv"
            summary = root / "summary.csv"
            _write_csv(
                source,
                [
                    {
                        "clip_id": "clip_001",
                        "dataset_name": "technical_demo",
                        "language": "en",
                        "model_name": "tiny",
                        "metric_name": "WER",
                        "error_rate": "0.4",
                        "reference_text": "We use pyannote for speaker diarization.",
                        "hypothesis_text": "We use piano note for speaker diary station.",
                    }
                ],
            )
            glossary.write_text(
                json.dumps(
                    [
                        {
                            "canonical": "pyannote",
                            "asr_error_forms": ["piano note"],
                            "allowed_contexts": ["speaker", "diarization"],
                        },
                        {
                            "canonical": "speaker diarization",
                            "spoken_forms": ["diarization"],
                            "asr_error_forms": ["diary station"],
                            "allowed_contexts": ["speaker"],
                        },
                    ]
                ),
                encoding="utf-8",
            )

            rows = evaluate_term_rescue_real_audit(
                input_path=source,
                glossary_path=glossary,
                output_path=output,
                summary_output_path=summary,
                terms_source="external_or_predefined",
            )

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["baseline_term_error_rate"], "1.000000")
            self.assertEqual(rows[0]["rescued_term_error_rate"], "0.000000")
            self.assertIn("piano note -> pyannote", rows[0]["applied_corrections"])
            self.assertEqual(rows[0]["verifier"], "rule")
            self.assertEqual(rows[0]["verifier_accept_count"], "2")
            self.assertTrue(output.exists())
            self.assertTrue(summary.exists())

    def test_oracle_source_is_labeled_as_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "asr.csv"
            glossary = root / "oracle.json"
            output = root / "term_audit.csv"
            _write_csv(
                source,
                [
                    {
                        "clip_id": "clip_002",
                        "dataset_name": "oracle_case",
                        "language": "en",
                        "model_name": "tiny",
                        "metric_name": "WER",
                        "error_rate": "0.5",
                        "reference_text": "Lakkha Singh led the bhajans.",
                        "hypothesis_text": "Lock a sink led the passions.",
                    }
                ],
            )
            glossary.write_text(
                json.dumps(
                    [
                        {
                            "canonical": "Lakkha Singh",
                            "asr_error_forms": ["Lock a sink"],
                        },
                        {
                            "canonical": "bhajans",
                            "asr_error_forms": ["passions"],
                        },
                    ]
                ),
                encoding="utf-8",
            )

            rows = evaluate_term_rescue_real_audit(
                input_path=source,
                glossary_path=glossary,
                output_path=output,
                terms_source="oracle_diagnostic",
                verifier="llm_with_rule_fallback",
            )

            self.assertIn("Oracle/custom-vocabulary", rows[0]["claim_scope"])
            self.assertIn("oracle/custom vocabulary diagnostic", rows[0]["notes"])
            self.assertGreaterEqual(
                int(rows[0]["verifier_fallback_used_count"]),
                1,
            )

    def test_cli_help_requires_no_data(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "experiments" / "evaluate_term_rescue_real_audit.py"),
                "--help",
            ],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--terms-source", result.stdout)
        self.assertIn("--verifier", result.stdout)


if __name__ == "__main__":
    unittest.main()
