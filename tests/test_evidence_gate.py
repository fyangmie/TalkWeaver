"""Network-free tests for the TalkWeaver EvidenceGate phase."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from backend.evidence_gate import (
    FEATURE_COLUMNS,
    extract_evidence_features,
    normalize_evidence_label,
)
from experiments.augment_evidence_gate_examples import (
    augment_evidence_gate_examples,
)
from experiments.build_evidence_gate_dataset import (
    build_evidence_gate_dataset,
)
from experiments.evaluate_evidence_gate import (
    compute_evidence_gate_metrics,
)
from experiments.plot_evidence_gate import plot_evidence_gate_results
from experiments.train_evidence_gate import (
    group_train_validation_test_split,
    train_evidence_gate_models,
)

ROOT_DIR = Path(__file__).resolve().parents[1]


def _training_frame(num_groups: int = 12) -> pd.DataFrame:
    rows = []
    for group_index in range(num_groups):
        for label in ("accept", "reject", "needs_review"):
            row = {
                "example_id": f"g{group_index}_{label}",
                "source_experiment": "test",
                "case_id": f"case_{group_index}",
                "variant": f"test_{label}",
                "template_group": f"group_{group_index}",
                "raw_text": "we use piano note",
                "corrected_text": (
                    "we use pyannote" if label == "accept" else "we use piano note"
                ),
                "reference_text": "we use pyannote",
                "language": "en",
                "is_augmented": False,
                "expected_label": label,
                "label_reason": f"test {label}",
                "safety_pass": label != "reject",
                "needs_review": label == "needs_review",
                "correction_rejected": label == "reject",
                "unsupported_change_count": int(label == "reject"),
                "forbidden_change_count": 0,
                "invented_content": False,
                "speaker_attribution_changed": False,
                "overlap": label == "needs_review",
                "uncertainty_level": (
                    "high" if label == "needs_review" else "low"
                ),
                "expected_terms": '["pyannote"]',
                "true_positive_terms": (
                    '["pyannote"]' if label == "accept" else "[]"
                ),
                "missed_terms": (
                    "[]" if label == "accept" else '["pyannote"]'
                ),
                "retrieved_candidates": '["pyannote"]',
                "applied_corrections": (
                    '["piano note -> pyannote"]' if label == "accept" else "[]"
                ),
                "text_error_before": 0.3,
                "text_error_after": 0.0 if label == "accept" else 0.3,
            }
            row.update(extract_evidence_features(row))
            rows.append(row)
    return pd.DataFrame(rows)


class EvidenceGateFeatureTests(unittest.TestCase):
    def test_feature_extraction_tracks_change_and_risk(self) -> None:
        features = extract_evidence_features(
            {
                "raw_text": "we use piano note",
                "corrected_text": "we use pyannote",
                "retrieved_candidates": '["pyannote"]',
                "applied_corrections": '["piano note -> pyannote"]',
                "expected_terms": '["pyannote"]',
                "true_positive_terms": '["pyannote"]',
                "overlap": True,
                "uncertainty_level": "high",
                "needs_review": True,
            }
        )

        self.assertEqual(features["retrieval_candidate_count"], 1)
        self.assertGreater(features["num_changed_tokens"], 0)
        self.assertEqual(features["heavy_overlap_flag"], 1)
        self.assertGreater(features["context_risk_score"], 0)

    def test_label_policy_is_conservative(self) -> None:
        label, _reason, _ambiguous = normalize_evidence_label(
            {
                "raw_text": "partial words",
                "corrected_text": "a complete unsupported claim",
                "invented_content": True,
                "safety_pass": False,
            }
        )
        review, _reason, _ambiguous = normalize_evidence_label(
            {
                "raw_text": "partial words",
                "corrected_text": "partial words",
                "needs_review": True,
                "overlap": True,
                "uncertainty_level": "high",
                "safety_pass": True,
            }
        )

        self.assertEqual(label, "reject")
        self.assertEqual(review, "needs_review")


class EvidenceGateDatasetTests(unittest.TestCase):
    def test_dataset_builder_normalizes_two_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            term = root / "term.csv"
            overlap = root / "overlap.csv"
            pd.DataFrame(
                [
                    {
                        "case_id": "term_1",
                        "variant": "fused_plus_rule_correction",
                        "raw_asr_text": "piano note",
                        "reference_text": "pyannote",
                        "corrected_text": "pyannote",
                        "expected_terms": '["pyannote"]',
                        "true_positive_terms": '["pyannote"]',
                        "missed_terms": "[]",
                        "unsupported_changes": "[]",
                        "needs_review": False,
                        "text_error_before": 1.0,
                        "text_error_after": 0.0,
                    }
                ]
            ).to_csv(term, index=False)
            pd.DataFrame(
                [
                    {
                        "case_id": "overlap_1",
                        "variant": "overlap_aware_rule",
                        "raw_asr_text": "partial",
                        "reference_text": "partial",
                        "corrected_text": "partial",
                        "overlap": True,
                        "uncertainty_level": "high",
                        "needs_review": True,
                        "correction_rejected": False,
                        "safety_pass": True,
                    }
                ]
            ).to_csv(overlap, index=False)
            result = build_evidence_gate_dataset(term, overlap)

        self.assertEqual(len(result), 2)
        self.assertEqual(set(result["source_experiment"]), {"term_rescue", "overlap_safety"})
        self.assertTrue(set(FEATURE_COLUMNS).issubset(result.columns))

    def test_augmentation_preserves_template_group(self) -> None:
        source = _training_frame(3)
        result = augment_evidence_gate_examples(
            source,
            augmentations_per_group=2,
        )

        self.assertEqual(result["template_group"].nunique(), 3)
        self.assertTrue(result["is_augmented"].astype(bool).any())
        self.assertTrue(set(result["expected_label"]).issubset(
            {"accept", "reject", "needs_review"}
        ))

    def test_group_split_has_no_template_leakage(self) -> None:
        train, validation, test = group_train_validation_test_split(
            _training_frame(),
            "template_group",
            random_seed=42,
        )
        groups = [
            set(part["template_group"])
            for part in (train, validation, test)
        ]

        self.assertFalse(groups[0] & groups[1])
        self.assertFalse(groups[0] & groups[2])
        self.assertFalse(groups[1] & groups[2])


class EvidenceGateTrainingTests(unittest.TestCase):
    def test_training_evaluation_and_plotting_on_tiny_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = train_evidence_gate_models(
                _training_frame(),
                root / "results",
                ["logistic_regression"],
                group_split_column="template_group",
                random_seed=42,
                models_dir=root / "models",
            )
            charts = plot_evidence_gate_results(
                result["metrics"],
                result["predictions"],
                result["feature_importance"],
                root / "charts",
            )

            self.assertTrue(Path(result["prediction_path"]).exists())
            self.assertTrue(
                (root / "models" / "evidence_gate_logistic_regression.joblib").exists()
            )
            self.assertEqual(len(charts), 5)
            self.assertTrue(all(path.exists() for path in charts))

    def test_safety_metrics_count_unsafe_accepts(self) -> None:
        metrics = compute_evidence_gate_metrics(
            ["accept", "reject", "needs_review"],
            ["accept", "accept", "needs_review"],
        )

        self.assertAlmostEqual(metrics["unsafe_accept_rate"], 1.0)
        self.assertAlmostEqual(metrics["false_accept_rate"], 0.5)
        self.assertAlmostEqual(metrics["needs_review_recall"], 1.0)

    def test_evidence_gate_cli_help(self) -> None:
        scripts = (
            "build_evidence_gate_dataset.py",
            "augment_evidence_gate_examples.py",
            "train_evidence_gate.py",
            "evaluate_evidence_gate.py",
            "plot_evidence_gate.py",
        )
        for script in scripts:
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT_DIR / "experiments" / script),
                    "--help",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
