"""Speaker diarization with pyannote support and deterministic fallback."""

from __future__ import annotations

from pathlib import Path
from typing import Any


DEFAULT_DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"
MOCK_DURATION_SECONDS = 9.4
MOCK_SPEAKER_TURNS: list[dict[str, Any]] = [
    {"start": 0.0, "end": 3.4, "speaker": "SPEAKER_00"},
    {"start": 3.0, "end": 6.5, "speaker": "SPEAKER_01"},
    {"start": 6.6, "end": 9.4, "speaker": "SPEAKER_00"},
]


def build_mock_speaker_turns(
    duration_seconds: float | None = None,
) -> list[dict[str, Any]]:
    """Return deterministic two-speaker turns with one deliberate overlap."""

    scale = 1.0
    if duration_seconds is not None and duration_seconds > 0:
        scale = duration_seconds / MOCK_DURATION_SECONDS
    return [
        {
            "start": round(float(turn["start"]) * scale, 3),
            "end": round(float(turn["end"]) * scale, 3),
            "speaker": str(turn["speaker"]),
        }
        for turn in MOCK_SPEAKER_TURNS
    ]


def _turns_from_annotation(annotation: Any) -> list[dict[str, Any]]:
    """Convert a pyannote Annotation-like object to serializable turns."""

    turns: list[dict[str, Any]] = []
    if hasattr(annotation, "itertracks"):
        iterator = annotation.itertracks(yield_label=True)
        for segment, _track, speaker in iterator:
            turns.append(
                {
                    "start": round(float(segment.start), 3),
                    "end": round(float(segment.end), 3),
                    "speaker": str(speaker),
                }
            )
    else:
        for item in annotation:
            if len(item) == 3:
                segment, _track, speaker = item
            else:
                segment, speaker = item
            turns.append(
                {
                    "start": round(float(segment.start), 3),
                    "end": round(float(segment.end), 3),
                    "speaker": str(speaker),
                }
            )
    return sorted(turns, key=lambda turn: (turn["start"], turn["end"]))


def _load_waveform_for_pyannote(source: Path) -> dict[str, Any]:
    """Load audio explicitly to avoid pyannote's optional torchcodec decoder."""

    import soundfile as sf
    import torch

    samples, sample_rate = sf.read(
        str(source),
        always_2d=True,
        dtype="float32",
    )
    waveform = torch.from_numpy(samples.T.copy())
    return {
        "uri": source.stem,
        "waveform": waveform,
        "sample_rate": int(sample_rate),
    }


def _mock_result(
    *,
    audio_path: str | Path | None,
    mode: str,
    fallback_reason: str | None = None,
    duration_seconds: float | None = None,
) -> dict[str, Any]:
    turns = build_mock_speaker_turns(duration_seconds)
    return {
        "mode": mode,
        "audio_path": str(audio_path) if audio_path is not None else None,
        "model": "deterministic_mock_diarization",
        "speaker_count": len({turn["speaker"] for turn in turns}),
        "turns": turns,
        "fallback_reason": fallback_reason,
        "is_mock": True,
    }


def diarize_with_metadata(
    audio_path: str | Path | None = None,
    *,
    mock: bool = False,
    hf_token: str = "",
    model_name: str = DEFAULT_DIARIZATION_MODEL,
    fallback_to_mock: bool = True,
    duration_seconds: float | None = None,
) -> dict[str, Any]:
    """Run pyannote diarization or return a labeled deterministic fallback."""

    if mock:
        return _mock_result(
            audio_path=audio_path,
            mode="mock_demo",
            duration_seconds=duration_seconds,
        )

    fallback_reason: str | None = None
    if not hf_token:
        fallback_reason = (
            "HF_TOKEN is not configured; "
            + (
                "deterministic mock diarization was used instead of "
                "pyannote.audio."
                if fallback_to_mock
                else "real diarization cannot run and mock fallback is disabled."
            )
        )

    if fallback_reason is None:
        try:
            from pyannote.audio import Pipeline
        except ImportError:
            fallback_reason = (
                "pyannote.audio is not installed; "
                + (
                    "deterministic mock diarization was used."
                    if fallback_to_mock
                    else "real diarization cannot run and mock fallback is disabled."
                )
            )

    if fallback_reason is not None:
        if not fallback_to_mock:
            raise RuntimeError(fallback_reason)
        return _mock_result(
            audio_path=audio_path,
            mode="mock_fallback",
            fallback_reason=fallback_reason,
            duration_seconds=duration_seconds,
        )

    if audio_path is None:
        raise ValueError("An audio path is required for pyannote diarization.")
    source = Path(audio_path)
    if not source.exists():
        raise FileNotFoundError(f"Audio file not found: {source}")

    try:
        try:
            pipeline = Pipeline.from_pretrained(model_name, token=hf_token)
        except TypeError:
            pipeline = Pipeline.from_pretrained(
                model_name,
                use_auth_token=hf_token,
            )
        output = pipeline(_load_waveform_for_pyannote(source))
        annotation = getattr(output, "speaker_diarization", output)
        turns = _turns_from_annotation(annotation)
    except Exception as exc:
        reason = (
            "pyannote diarization failed; deterministic mock diarization was "
            f"used. Cause: {exc}"
        )
        if not fallback_to_mock:
            raise RuntimeError(reason) from exc
        return _mock_result(
            audio_path=source,
            mode="mock_fallback",
            fallback_reason=reason,
            duration_seconds=duration_seconds,
        )

    return {
        "mode": "pyannote",
        "audio_path": str(source),
        "model": model_name,
        "speaker_count": len({turn["speaker"] for turn in turns}),
        "turns": turns,
        "fallback_reason": None,
        "is_mock": False,
    }


def diarize(
    audio_path: str | Path | None = None,
    *,
    mock: bool = False,
    hf_token: str = "",
    fallback_to_mock: bool = True,
    duration_seconds: float | None = None,
) -> list[dict[str, Any]]:
    """Compatibility wrapper that returns only speaker turns."""

    result = diarize_with_metadata(
        audio_path,
        mock=mock,
        hf_token=hf_token,
        fallback_to_mock=fallback_to_mock,
        duration_seconds=duration_seconds,
    )
    return result["turns"]
