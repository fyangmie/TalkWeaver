"""Offline tests for secure optional LLM correction configuration."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.constrained_correction import apply_constrained_correction
from backend.llm_config import LLMConfig, load_llm_config, mask_api_key
from backend.llm_correction import correct_segments
from backend.schemas import TemporalAnchor
from backend.term_rescue import retrieve_term_candidates


ROOT = Path(__file__).resolve().parents[1]
SEGMENT = {
    "start": 0.0,
    "end": 3.0,
    "speaker": "SPEAKER_00",
    "speakers": ["SPEAKER_00"],
    "raw_text": "we use piano note for diary station",
    "retrieved_terms": ["pyannote", "diarization"],
    "overlap": False,
    "confidence": 0.9,
}


class LLMConfigTests(unittest.TestCase):
    def test_masks_api_key_in_metadata_and_repr(self) -> None:
        config = LLMConfig(
            provider="deepseek",
            api_key="sk-secret-value-1234",
            model="deepseek-chat",
            base_url="https://api.deepseek.com",
        )

        self.assertNotIn("secret-value", repr(config))
        self.assertNotIn(
            "secret-value",
            str(config.safe_metadata()),
        )
        self.assertEqual(
            config.safe_metadata()["api_key"],
            mask_api_key("sk-secret-value-1234"),
        )

    def test_missing_api_key_fails_in_llm_mode(self) -> None:
        with self.assertRaisesRegex(
            RuntimeError,
            "LLM API configuration is incomplete",
        ):
            load_llm_config(
                correction_mode="llm",
                environment={
                    "LLM_PROVIDER": "deepseek",
                    "LLM_MODEL": "deepseek-chat",
                    "LLM_BASE_URL": "https://api.deepseek.com",
                },
                load_dotenv_file=False,
            )


class CorrectionModeTests(unittest.TestCase):
    def test_rule_fallback_requires_no_api(self) -> None:
        corrected = correct_segments(
            [SEGMENT],
            correction_mode="rule_fallback",
        )

        self.assertEqual(
            corrected[0]["corrected_text"],
            "we use pyannote for diarization",
        )
        self.assertEqual(
            corrected[0]["correction_mode"],
            "rule_fallback",
        )
        self.assertFalse(corrected[0]["api_used"])
        self.assertFalse(corrected[0]["fallback_used"])

    def test_llm_with_rule_fallback_marks_unavailable_api(self) -> None:
        config = load_llm_config(
            correction_mode="llm_with_rule_fallback",
            environment={
                "LLM_PROVIDER": "deepseek",
                "LLM_MODEL": "deepseek-chat",
                "LLM_BASE_URL": "https://api.deepseek.com",
            },
            load_dotenv_file=False,
        )

        corrected = correct_segments(
            [SEGMENT],
            correction_mode="llm_with_rule_fallback",
            llm_config=config,
        )

        self.assertEqual(
            corrected[0]["corrected_text"],
            "we use pyannote for diarization",
        )
        self.assertFalse(corrected[0]["api_used"])
        self.assertTrue(corrected[0]["fallback_used"])
        self.assertEqual(
            corrected[0]["correction_backend_mode"],
            "api_unconfigured_fallback_rule_based",
        )

    def test_correction_audit_records_execution_metadata(self) -> None:
        config = load_llm_config(
            correction_mode="llm_with_rule_fallback",
            environment={
                "LLM_PROVIDER": "deepseek",
                "LLM_MODEL": "deepseek-chat",
                "LLM_BASE_URL": "https://api.deepseek.com",
                "LLM_TEMPERATURE": "0",
            },
            load_dotenv_file=False,
        )
        anchor = TemporalAnchor(
            anchor_id="anchor",
            clip_id="clip",
            start=0.0,
            end=3.0,
            speaker="SPEAKER_00",
            speakers=["SPEAKER_00"],
            raw_text="we use piano note for diary station",
            language="en",
            confidence=0.9,
        )
        candidates = retrieve_term_candidates([anchor])

        _anchors, audits, mode = apply_constrained_correction(
            [anchor],
            candidates,
            [],
            llm_config={
                "correction_mode": "llm_with_rule_fallback",
                "runtime_config": config,
            },
        )

        self.assertEqual(mode, "llm_with_rule_fallback")
        self.assertEqual(audits[0].llm_provider, "deepseek")
        self.assertEqual(audits[0].llm_model, "deepseek-chat")
        self.assertEqual(
            audits[0].prompt_version,
            "talkweaver.correction.v1",
        )
        self.assertFalse(audits[0].api_used)
        self.assertTrue(audits[0].fallback_used)

    def test_llm_mode_does_not_silently_fallback_on_failure(self) -> None:
        config = LLMConfig(
            provider="deepseek",
            api_key="test-key",
            model="deepseek-chat",
            base_url="https://api.deepseek.com",
        )
        with patch(
            "backend.llm_correction._chat_completion",
            side_effect=RuntimeError("network unavailable"),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "LLM correction failed",
            ):
                correct_segments(
                    [SEGMENT],
                    correction_mode="llm",
                    llm_config=config,
                )


class SmokeCliTests(unittest.TestCase):
    def test_help_requires_no_api(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "run_llm_correction_smoke.py"),
                "--help",
            ],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--mode", result.stdout)


if __name__ == "__main__":
    unittest.main()
