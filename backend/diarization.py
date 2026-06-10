"""Speaker diarization interface with deterministic mock turns."""

from __future__ import annotations

from pathlib import Path
from typing import Any


MOCK_SPEAKER_TURNS: list[dict[str, Any]] = [
    {"start": 0.0, "end": 3.4, "speaker": "SPEAKER_00"},
    {"start": 3.0, "end": 6.5, "speaker": "SPEAKER_01"},
    {"start": 6.6, "end": 9.4, "speaker": "SPEAKER_00"},
]


def diarize(
    audio_path: str | Path | None = None,
    *,
    mock: bool = False,
    hf_token: str = "",
) -> list[dict[str, Any]]:
    """Diarize audio or return deterministic mock speaker turns."""

    if mock:
        return [dict(turn) for turn in MOCK_SPEAKER_TURNS]

    if audio_path is None:
        raise ValueError("An audio path is required outside mock mode.")
    if not Path(audio_path).exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if not hf_token:
        raise RuntimeError("HF_TOKEN is required for real pyannote diarization.")

    try:
        import pyannote.audio  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "pyannote.audio is not installed. Install Phase 3 dependencies or "
            "run with --mock."
        ) from exc

    raise RuntimeError("Real pyannote inference is scheduled for Phase 3.")
