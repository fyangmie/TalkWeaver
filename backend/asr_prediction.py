"""Load audited real-ASR prediction JSON into the backend ASR contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_asr_prediction_json(path: str | Path) -> dict[str, Any]:
    """Load Phase 2C prediction JSON and reject mock or malformed output."""

    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if bool(payload.get("is_mock")):
        raise RuntimeError(
            f"Prediction JSON is labeled mock and cannot be reused: {source}"
        )
    asr_payload = payload.get("asr", payload)
    if str(asr_payload.get("asr_mode", "")).startswith("mock"):
        raise RuntimeError(
            f"Prediction JSON contains mock ASR output: {source}"
        )
    segments = asr_payload.get("segments")
    if not isinstance(segments, list):
        raise ValueError(
            f"Prediction JSON has no ASR segment list: {source}"
        )
    return {
        "mode": "real_prediction_json",
        "segments": segments,
        "language": (
            payload.get("language")
            or asr_payload.get("language_detected")
            or asr_payload.get("language_requested")
        ),
        "source_path": str(source),
        "model": payload.get("model_name"),
        "vad_filter": payload.get(
            "vad_filter",
            asr_payload.get("vad_filter"),
        ),
    }
