"""Lightweight correction-safety model used by TalkWeaver EvidenceGate.

EvidenceGate is trained on controlled and transparently augmented correction
decisions. It does not replace ASR, retrieval, or an LLM. It classifies a
proposed correction as ``accept``, ``reject``, or ``needs_review`` from the
evidence already produced by those components.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Mapping


LABELS = ("accept", "reject", "needs_review")
FEATURE_COLUMNS = (
    "retrieval_candidate_count",
    "applied_correction_count",
    "expected_term_count",
    "true_positive_term_count",
    "false_positive_term_count",
    "missed_term_count",
    "term_precision",
    "term_recall",
    "term_f1",
    "text_error_before",
    "text_error_after",
    "error_delta",
    "num_changed_tokens",
    "changed_token_ratio",
    "edit_distance_ratio",
    "overlap_flag",
    "heavy_overlap_flag",
    "uncertainty_score",
    "needs_review_input_flag",
    "correction_rejected_input_flag",
    "unsupported_change_count",
    "forbidden_change_count",
    "invented_content_flag",
    "speaker_attribution_changed_flag",
    "api_used_flag",
    "llm_variant_flag",
    "rule_variant_flag",
    "negative_control_flag",
    "context_risk_score",
)

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[.-][A-Za-z0-9]+)*|[\u3400-\u9fff]")


def _is_missing(value: Any) -> bool:
    return value is None or (
        isinstance(value, float) and math.isnan(value)
    )


def as_bool(value: Any) -> bool:
    """Parse CSV-style booleans without treating NaN as true."""

    if _is_missing(value):
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def as_float(value: Any, default: float = 0.0) -> float:
    if _is_missing(value):
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(result) else result


def parse_list(value: Any) -> list[Any]:
    """Parse JSON-list CSV values and tolerate plain comma-separated text."""

    if _is_missing(value):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, (tuple, set)):
        return list(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return [item.strip() for item in text.split(",") if item.strip()]
        if isinstance(parsed, list):
            return parsed
        return [parsed]
    return [value]


def tokenize(text: Any) -> list[str]:
    return _TOKEN_PATTERN.findall(str(text or "").lower())


def token_change_stats(raw_text: str, corrected_text: str) -> tuple[int, float]:
    raw_tokens = tokenize(raw_text)
    corrected_tokens = tokenize(corrected_text)
    matcher = SequenceMatcher(a=raw_tokens, b=corrected_tokens)
    changed = 0
    for tag, left_start, left_end, right_start, right_end in matcher.get_opcodes():
        if tag != "equal":
            changed += max(left_end - left_start, right_end - right_start)
    denominator = max(len(raw_tokens), len(corrected_tokens), 1)
    return changed, changed / denominator


def edit_distance(left: str, right: str) -> int:
    """Compute character edit distance without an external dependency."""

    if left == right:
        return 0
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def edit_distance_ratio(raw_text: str, corrected_text: str) -> float:
    denominator = max(len(str(raw_text)), len(str(corrected_text)), 1)
    return edit_distance(str(raw_text), str(corrected_text)) / denominator


def infer_negative_control(row: Mapping[str, Any]) -> bool:
    expected_terms = parse_list(row.get("expected_terms"))
    expected_behavior = str(row.get("expected_safe_behavior", "")).lower()
    notes = str(row.get("notes", "")).lower()
    context = str(row.get("context", "")).lower()
    return (
        not expected_terms
        or "negative control" in expected_behavior
        or "negative control" in notes
        or "physical rack" in context
        or "normal word" in expected_behavior
    )


def uncertainty_score(value: Any) -> float:
    if _is_missing(value):
        return 0.0
    text = str(value).strip().lower()
    return {"none": 0.0, "low": 0.25, "medium": 0.6, "high": 1.0}.get(
        text,
        as_float(value),
    )


def _count_or_value(row: Mapping[str, Any], count_key: str, list_key: str) -> int:
    if count_key in row and not _is_missing(row.get(count_key)):
        return int(as_float(row.get(count_key)))
    return len(parse_list(row.get(list_key)))


def extract_evidence_features(row: Mapping[str, Any]) -> dict[str, float]:
    """Extract deterministic numeric features from one correction decision."""

    raw_text = str(row.get("raw_text", row.get("raw_asr_text", "")) or "")
    corrected_text = str(row.get("corrected_text", "") or "")
    changed_tokens, changed_ratio = token_change_stats(raw_text, corrected_text)
    before = as_float(row.get("text_error_before"))
    after = as_float(row.get("text_error_after"), before)
    overlap = as_bool(row.get("overlap"))
    uncertainty = uncertainty_score(row.get("uncertainty_level"))
    unsupported = _count_or_value(
        row,
        "unsupported_change_count",
        "unsupported_changes",
    )
    forbidden = _count_or_value(
        row,
        "forbidden_change_count",
        "forbidden_changes_detected",
    )
    invented = as_bool(row.get("invented_content"))
    speaker_changed = as_bool(row.get("speaker_attribution_changed"))
    needs_review = as_bool(row.get("needs_review"))
    rejected = as_bool(row.get("correction_rejected"))
    variant = str(row.get("variant", "")).lower()
    heavy_overlap = overlap and (
        uncertainty >= 0.9
        or "heavy" in str(row.get("difficulty", "")).lower()
        or "heavy" in str(row.get("notes", "")).lower()
    )
    negative_control = infer_negative_control(row)
    context_risk = min(
        1.0,
        0.20 * overlap
        + 0.25 * heavy_overlap
        + 0.20 * uncertainty
        + 0.10 * needs_review
        + 0.15 * (unsupported > 0)
        + 0.15 * (forbidden > 0)
        + 0.20 * invented
        + 0.20 * speaker_changed
        + 0.10 * (negative_control and changed_tokens > 0),
    )
    return {
        "retrieval_candidate_count": float(
            len(parse_list(row.get("retrieved_candidates")))
        ),
        "applied_correction_count": float(
            len(
                parse_list(
                    row.get("applied_corrections", row.get("applied_changes"))
                )
            )
        ),
        "expected_term_count": float(len(parse_list(row.get("expected_terms")))),
        "true_positive_term_count": float(
            len(parse_list(row.get("true_positive_terms")))
        ),
        "false_positive_term_count": float(
            len(parse_list(row.get("false_positive_terms")))
        ),
        "missed_term_count": float(len(parse_list(row.get("missed_terms")))),
        "term_precision": as_float(row.get("term_precision")),
        "term_recall": as_float(row.get("term_recall")),
        "term_f1": as_float(row.get("term_f1")),
        "text_error_before": before,
        "text_error_after": after,
        "error_delta": before - after,
        "num_changed_tokens": float(changed_tokens),
        "changed_token_ratio": changed_ratio,
        "edit_distance_ratio": edit_distance_ratio(raw_text, corrected_text),
        "overlap_flag": float(overlap),
        "heavy_overlap_flag": float(heavy_overlap),
        "uncertainty_score": uncertainty,
        "needs_review_input_flag": float(needs_review),
        "correction_rejected_input_flag": float(rejected),
        "unsupported_change_count": float(unsupported),
        "forbidden_change_count": float(forbidden),
        "invented_content_flag": float(invented),
        "speaker_attribution_changed_flag": float(speaker_changed),
        "api_used_flag": float(as_bool(row.get("api_used"))),
        "llm_variant_flag": float("llm" in variant),
        "rule_variant_flag": float("rule" in variant),
        "negative_control_flag": float(negative_control),
        "context_risk_score": context_risk,
    }


def normalize_evidence_label(
    row: Mapping[str, Any],
) -> tuple[str, str, bool]:
    """Apply the conservative EvidenceGate target-label policy.

    Returns ``(label, reason, ambiguous)``. The label uses decision-time audit
    fields already emitted by Phase 2F/2G; it is not a human annotation of
    real-world audio safety.
    """

    features = extract_evidence_features(row)
    raw_text = str(row.get("raw_text", row.get("raw_asr_text", "")) or "")
    corrected_text = str(row.get("corrected_text", "") or "")
    correction_error_value = row.get("correction_error", "")
    correction_error = (
        ""
        if _is_missing(correction_error_value)
        else str(correction_error_value).strip()
    )
    safety_present = "safety_pass" in row and not _is_missing(row.get("safety_pass"))
    safety_pass = as_bool(row.get("safety_pass")) if safety_present else True
    explicit_reject = bool(features["correction_rejected_input_flag"])
    unsafe = (
        features["invented_content_flag"] > 0
        or features["speaker_attribution_changed_flag"] > 0
        or features["forbidden_change_count"] > 0
        or features["unsupported_change_count"] > 0
        or bool(correction_error)
        or (safety_present and not safety_pass)
    )
    if explicit_reject or unsafe:
        reasons = []
        if explicit_reject:
            reasons.append("explicitly rejected")
        if features["invented_content_flag"]:
            reasons.append("invented content")
        if features["speaker_attribution_changed_flag"]:
            reasons.append("speaker attribution changed")
        if features["forbidden_change_count"]:
            reasons.append("forbidden change")
        if features["unsupported_change_count"]:
            reasons.append("unsupported change")
        if correction_error:
            reasons.append("strict correction validation failed")
        if safety_present and not safety_pass:
            reasons.append("safety policy failed")
        return "reject", "; ".join(reasons), False

    if features["needs_review_input_flag"]:
        return "needs_review", "source audit requested human review", False

    unchanged = raw_text.strip() == corrected_text.strip()
    missed_terms = features["missed_term_count"] > 0
    expected_terms = features["expected_term_count"] > 0
    negative_control = features["negative_control_flag"] > 0
    if unchanged and expected_terms and missed_terms:
        return (
            "needs_review",
            "supported target remains unresolved",
            True,
        )
    if features["error_delta"] < -1e-9:
        return "reject", "correction increased reference text error", False
    if unchanged and negative_control:
        return "accept", "negative-control text safely preserved", False
    if safety_pass and features["unsupported_change_count"] == 0:
        return "accept", "supported correction or safe preservation", False
    return "needs_review", "ambiguous controlled decision", True


@dataclass(slots=True)
class EvidenceGateExample:
    example_id: str
    source_experiment: str
    case_id: str
    variant: str
    raw_text: str
    corrected_text: str
    reference_text: str
    expected_label: str
    language: str = "unknown"
    template_group: str = ""
    is_augmented: bool = False
    label_reason: str = ""
    notes: str = ""
    features: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        features = payload.pop("features") or {}
        payload.update(features)
        return payload


@dataclass(slots=True)
class EvidenceGatePrediction:
    example_id: str
    true_label: str
    predicted_label: str
    model_name: str
    split: str
    probabilities: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["probabilities"] = json.dumps(
            self.probabilities,
            sort_keys=True,
        )
        return payload


class EvidenceGateModel:
    """Thin sklearn wrapper with stable model names and feature ordering."""

    def __init__(self, model_name: str, random_seed: int = 42) -> None:
        self.model_name = model_name
        self.random_seed = random_seed
        self.estimator = self._build_estimator()

    def _build_estimator(self) -> Any:
        try:
            from sklearn.ensemble import (
                GradientBoostingClassifier,
                RandomForestClassifier,
            )
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
        except ImportError as exc:
            raise RuntimeError(
                "EvidenceGate requires scikit-learn. Install project "
                "dependencies with: pip install -r requirements.txt"
            ) from exc

        if self.model_name == "logistic_regression":
            return Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            class_weight="balanced",
                            max_iter=2000,
                            random_state=self.random_seed,
                        ),
                    ),
                ]
            )
        if self.model_name == "random_forest":
            return RandomForestClassifier(
                n_estimators=240,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=self.random_seed,
                n_jobs=-1,
            )
        if self.model_name == "gradient_boosting":
            return GradientBoostingClassifier(random_state=self.random_seed)
        raise ValueError(
            f"Unsupported EvidenceGate model: {self.model_name}. "
            "Choose logistic_regression, random_forest, or gradient_boosting."
        )

    @staticmethod
    def _matrix(rows: Iterable[Mapping[str, Any]]) -> list[list[float]]:
        return [
            [as_float(row.get(feature)) for feature in FEATURE_COLUMNS]
            for row in rows
        ]

    def fit(
        self,
        rows: Iterable[Mapping[str, Any]],
        labels: Iterable[str],
        sample_weight: Iterable[float] | None = None,
    ) -> "EvidenceGateModel":
        row_list = list(rows)
        label_list = list(labels)
        weights = list(sample_weight) if sample_weight is not None else None
        if self.model_name == "gradient_boosting" and weights is not None:
            self.estimator.fit(self._matrix(row_list), label_list, sample_weight=weights)
        else:
            self.estimator.fit(self._matrix(row_list), label_list)
        return self

    def predict(self, rows: Iterable[Mapping[str, Any]]) -> list[str]:
        return list(self.estimator.predict(self._matrix(list(rows))))

    def predict_proba(
        self,
        rows: Iterable[Mapping[str, Any]],
    ) -> list[dict[str, float]]:
        row_list = list(rows)
        values = self.estimator.predict_proba(self._matrix(row_list))
        classes = list(self.estimator.classes_)
        return [
            {str(label): float(probability) for label, probability in zip(classes, row)}
            for row in values
        ]

    def feature_importance(self) -> dict[str, float]:
        estimator = self.estimator
        if self.model_name == "logistic_regression":
            estimator = estimator.named_steps["model"]
            values = abs(estimator.coef_).mean(axis=0)
        else:
            values = getattr(estimator, "feature_importances_", [])
        return {
            feature: float(value)
            for feature, value in zip(FEATURE_COLUMNS, values)
        }

    def save(self, path: str | Path) -> Path:
        try:
            import joblib
        except ImportError as exc:
            raise RuntimeError("Saving EvidenceGate requires joblib.") from exc
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model_name": self.model_name,
                "random_seed": self.random_seed,
                "feature_columns": FEATURE_COLUMNS,
                "estimator": self.estimator,
            },
            destination,
        )
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "EvidenceGateModel":
        try:
            import joblib
        except ImportError as exc:
            raise RuntimeError("Loading EvidenceGate requires joblib.") from exc
        payload = joblib.load(path)
        model = cls(payload["model_name"], payload.get("random_seed", 42))
        model.estimator = payload["estimator"]
        return model
