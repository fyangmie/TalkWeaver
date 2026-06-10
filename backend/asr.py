"""ASR interface with deterministic Phase 1 mock output."""

from __future__ import annotations

from pathlib import Path
from typing import Any


MOCK_ASR_SEGMENTS: list[dict[str, Any]] = [
    {
        "start": 0.0,
        "end": 3.2,
        "text": "We use piano note for diary station.",
        "words": [
            {"word": "We", "start": 0.0, "end": 0.3},
            {"word": "use", "start": 0.35, "end": 0.65},
            {"word": "piano", "start": 0.7, "end": 1.2},
            {"word": "note", "start": 1.25, "end": 1.55},
            {"word": "for", "start": 1.65, "end": 1.9},
            {"word": "diary", "start": 2.0, "end": 2.45},
            {"word": "station.", "start": 2.5, "end": 3.2},
        ],
    },
    {
        "start": 3.0,
        "end": 6.5,
        "text": "The rack glossary can reduce term errors.",
        "words": [
            {"word": "The", "start": 3.0, "end": 3.2},
            {"word": "rack", "start": 3.25, "end": 3.65},
            {"word": "glossary", "start": 3.7, "end": 4.3},
            {"word": "can", "start": 4.35, "end": 4.6},
            {"word": "reduce", "start": 4.65, "end": 5.1},
            {"word": "term", "start": 5.2, "end": 5.55},
            {"word": "errors.", "start": 5.6, "end": 6.5},
        ],
    },
    {
        "start": 6.6,
        "end": 9.4,
        "text": "We should compare where and the ear.",
        "words": [
            {"word": "We", "start": 6.6, "end": 6.85},
            {"word": "should", "start": 6.9, "end": 7.2},
            {"word": "compare", "start": 7.25, "end": 7.7},
            {"word": "where", "start": 7.75, "end": 8.15},
            {"word": "and", "start": 8.2, "end": 8.4},
            {"word": "the", "start": 8.45, "end": 8.65},
            {"word": "ear.", "start": 8.7, "end": 9.4},
        ],
    },
]


def transcribe(
    audio_path: str | Path | None = None,
    *,
    mock: bool = False,
    model_size: str = "medium",
) -> list[dict[str, Any]]:
    """Transcribe audio or return deterministic mock segments."""

    if mock:
        return [dict(segment) for segment in MOCK_ASR_SEGMENTS]

    if audio_path is None:
        raise ValueError("An audio path is required outside mock mode.")

    if not Path(audio_path).exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    try:
        import faster_whisper  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "faster-whisper is not installed. Install the Phase 2 optional "
            "dependencies or run with --mock."
        ) from exc

    raise RuntimeError(
        f"Real faster-whisper inference ({model_size}) is scheduled for Phase 2."
    )
