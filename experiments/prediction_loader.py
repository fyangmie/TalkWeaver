"""Stable loader for audited Phase 2C real-ASR prediction artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PredictionRecord:
    clip_id: str
    dataset_name: str
    language: str
    model_name: str
    reference_text: str
    hypothesis_text: str
    metric_name: str
    error_rate: float
    segments: list[dict[str, Any]]
    source_path: Path
    vad_filter: bool | None = None

    def as_asr_output(self) -> dict[str, Any]:
        """Return the backend ASR contract without relabeling the run."""

        return {
            "mode": "real_prediction_json",
            "segments": self.segments,
            "language": self.language,
            "source_path": str(self.source_path),
            "model": self.model_name,
            "vad_filter": self.vad_filter,
        }


def prediction_path(
    predictions_dir: str | Path,
    model_name: str,
    clip_id: str,
) -> Path:
    """Return the Phase 2C naming convention for one prediction."""

    return Path(predictions_dir) / f"{model_name}__{clip_id}.json"


def load_prediction_json(path: str | Path) -> PredictionRecord:
    """Load one real prediction and reject mock or malformed artifacts."""

    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if bool(payload.get("is_mock")):
        raise RuntimeError(
            f"Mock prediction cannot be used as real evidence: {source}"
        )
    asr = payload.get("asr")
    if not isinstance(asr, dict):
        raise ValueError(f"Prediction JSON has no ASR payload: {source}")
    if str(asr.get("asr_mode", "")).lower() != "real":
        raise RuntimeError(
            "Prediction JSON is not labeled as real ASR: "
            f"{source}"
        )
    segments = asr.get("segments")
    if not isinstance(segments, list):
        raise ValueError(
            f"Prediction JSON has no ASR segment list: {source}"
        )
    return PredictionRecord(
        clip_id=str(payload["clip_id"]),
        dataset_name=str(payload.get("dataset_name", "")),
        language=str(payload.get("language", "")),
        model_name=str(payload.get("model_name", "")),
        reference_text=str(payload.get("reference_text", "")),
        hypothesis_text=str(
            payload.get("hypothesis_text")
            or asr.get("hypothesis_text", "")
        ),
        metric_name=str(payload.get("metric_name", "")),
        error_rate=float(payload.get("error_rate", 0.0)),
        segments=[dict(segment) for segment in segments],
        source_path=source,
        vad_filter=payload.get("vad_filter"),
    )


def find_and_load_prediction(
    predictions_dir: str | Path,
    model_name: str,
    clip_id: str,
) -> PredictionRecord | None:
    """Load a matching prediction, returning ``None`` when absent."""

    source = prediction_path(predictions_dir, model_name, clip_id)
    return load_prediction_json(source) if source.is_file() else None
