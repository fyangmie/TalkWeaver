"""ASR baseline using faster-whisper with a deterministic mock fallback."""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger(__name__)

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


def _mock_result(
    *,
    audio_path: str | Path | None,
    model_size: str,
    mode: str,
    fallback_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "mode": mode,
        "audio_path": str(audio_path) if audio_path else None,
        "model": f"mock:{model_size}",
        "device": "none",
        "compute_type": "none",
        "language": "en",
        "language_probability": 1.0,
        "duration_seconds": 9.4,
        "word_timestamps": True,
        "segments": copy.deepcopy(MOCK_ASR_SEGMENTS),
        "fallback_reason": fallback_reason,
    }


def _optional_float(value: Any) -> float | None:
    return None if value is None else round(float(value), 3)


def _serialize_word(word: Any) -> dict[str, Any]:
    payload = {
        "word": str(word.word).strip(),
        "start": _optional_float(word.start),
        "end": _optional_float(word.end),
    }
    probability = getattr(word, "probability", None)
    if probability is not None:
        payload["probability"] = round(float(probability), 4)
    return payload


def _serialize_segment(segment: Any) -> dict[str, Any]:
    words = getattr(segment, "words", None) or []
    payload = {
        "start": round(float(segment.start), 3),
        "end": round(float(segment.end), 3),
        "text": str(segment.text).strip(),
        "words": [_serialize_word(word) for word in words],
    }
    average_log_probability = getattr(segment, "avg_logprob", None)
    if average_log_probability is not None:
        payload["avg_logprob"] = round(float(average_log_probability), 4)
    no_speech_probability = getattr(segment, "no_speech_prob", None)
    if no_speech_probability is not None:
        payload["no_speech_probability"] = round(
            float(no_speech_probability),
            4,
        )
    return payload


def transcribe_with_metadata(
    audio_path: str | Path | None = None,
    *,
    mock: bool = False,
    model_size: str = "medium",
    device: str = "auto",
    compute_type: str = "default",
    language: str | None = None,
    beam_size: int = 5,
    word_timestamps: bool = True,
    vad_filter: bool = True,
    fallback_to_mock: bool = True,
) -> dict[str, Any]:
    """Transcribe audio and return segments plus execution metadata."""

    if mock:
        return _mock_result(
            audio_path=audio_path,
            model_size=model_size,
            mode="mock_demo",
        )

    if audio_path is None:
        raise ValueError("An audio path is required outside mock mode.")

    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        message = (
            "faster-whisper is not installed; using deterministic mock ASR. "
            "Install it with `pip install faster-whisper` for real inference."
        )
        if not fallback_to_mock:
            raise RuntimeError(message) from exc
        LOGGER.warning(message)
        return _mock_result(
            audio_path=path,
            model_size=model_size,
            mode="mock_fallback",
            fallback_reason=message,
        )

    model = WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type,
    )
    segment_generator, info = model.transcribe(
        str(path),
        beam_size=beam_size,
        language=language,
        word_timestamps=word_timestamps,
        vad_filter=vad_filter,
    )
    segments = [_serialize_segment(segment) for segment in segment_generator]

    duration = getattr(info, "duration", None)
    if duration is None and segments:
        duration = segments[-1]["end"]

    return {
        "mode": "faster_whisper",
        "audio_path": str(path),
        "model": model_size,
        "device": device,
        "compute_type": compute_type,
        "language": getattr(info, "language", language),
        "language_probability": _optional_float(
            getattr(info, "language_probability", None)
        ),
        "duration_seconds": _optional_float(duration),
        "duration_after_vad_seconds": _optional_float(
            getattr(info, "duration_after_vad", None)
        ),
        "word_timestamps": word_timestamps,
        "segments": segments,
        "fallback_reason": None,
    }


def transcribe(
    audio_path: str | Path | None = None,
    *,
    mock: bool = False,
    model_size: str = "medium",
    device: str = "auto",
    compute_type: str = "default",
    language: str | None = None,
    fallback_to_mock: bool = True,
) -> list[dict[str, Any]]:
    """Compatibility wrapper returning only ASR segments."""

    result = transcribe_with_metadata(
        audio_path,
        mock=mock,
        model_size=model_size,
        device=device,
        compute_type=compute_type,
        language=language,
        fallback_to_mock=fallback_to_mock,
    )
    return result["segments"]
