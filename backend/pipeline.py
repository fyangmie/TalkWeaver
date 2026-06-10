"""TalkWeaver pipeline orchestration with a Phase 2 ASR baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.alignment import align_segments
from backend.asr import transcribe_with_metadata
from backend.config import get_settings
from backend.diarization import diarize
from backend.export import (
    export_raw_transcript,
    write_json,
    write_transcript_markdown,
)
from backend.llm_correction import correct_segments
from backend.overlap import detect_overlap_regions
from backend.preprocessing import preprocess_audio
from backend.summarizer import summarize_segments


def _stringify_paths(paths: dict[str, Path]) -> dict[str, str]:
    return {name: str(path) for name, path in paths.items()}


def _run_mock_pipeline(
    *,
    audio_path: str | Path | None,
    denoise: bool,
    model_size: str,
) -> dict[str, Any]:
    settings = get_settings()
    preprocessing = preprocess_audio(
        audio_path,
        mock=True,
        denoise=denoise,
    )
    asr_result = transcribe_with_metadata(
        audio_path,
        mock=True,
        model_size=model_size,
    )
    asr_segments = asr_result["segments"]
    raw_paths = export_raw_transcript(
        settings.output_dir / "transcripts",
        "mock_asr",
        asr_result,
    )

    speaker_turns = diarize(audio_path, mock=True)
    overlap_regions = detect_overlap_regions(speaker_turns)
    aligned = align_segments(asr_segments, speaker_turns, overlap_regions)
    corrected = correct_segments(aligned, mock=True)
    summary = summarize_segments(corrected)

    paths = {
        "raw_asr_json": raw_paths["json"],
        "raw_asr_markdown": raw_paths["markdown"],
        "raw_asr_metadata": raw_paths["metadata"],
        "diarization": write_json(
            settings.output_dir / "diarization" / "mock_diarization.json",
            speaker_turns,
        ),
        "corrected": write_json(
            settings.output_dir
            / "corrected_transcripts"
            / "mock_temporal_anchor.json",
            corrected,
        ),
        "transcript_markdown": write_transcript_markdown(
            settings.output_dir
            / "corrected_transcripts"
            / "mock_transcript.md",
            corrected,
        ),
        "summary": write_json(
            settings.output_dir / "summaries" / "mock_summary.json",
            summary,
        ),
    }

    result = {
        "mode": "mock_demo",
        "audio_path": str(audio_path) if audio_path else None,
        "preprocessing": preprocessing,
        "asr": {
            key: value
            for key, value in asr_result.items()
            if key != "segments"
        },
        "asr_segments": asr_segments,
        "speaker_turns": speaker_turns,
        "overlap_regions": overlap_regions,
        "transcript": corrected,
        "summary": summary,
        "artifacts": _stringify_paths(paths),
        "warning": "Mock/demo output is not a real experimental result.",
    }
    manifest = write_json(
        settings.output_dir / "exports" / "mock_pipeline_manifest.json",
        result,
    )
    result["artifacts"]["manifest"] = str(manifest)
    return result


def _run_phase2_pipeline(
    *,
    audio_path: str | Path | None,
    denoise: bool,
    model_size: str,
    language: str | None,
    device: str,
    compute_type: str,
) -> dict[str, Any]:
    if audio_path is None:
        raise ValueError("An audio path is required outside mock mode.")

    settings = get_settings()
    preprocessing = preprocess_audio(
        audio_path,
        mock=False,
        denoise=denoise,
    )
    processed_path = Path(preprocessing["output_path"])
    asr_result = transcribe_with_metadata(
        processed_path,
        model_size=model_size,
        language=language,
        device=device,
        compute_type=compute_type,
        fallback_to_mock=True,
    )
    stem = f"{Path(audio_path).stem}_raw_asr"
    raw_paths = export_raw_transcript(
        settings.output_dir / "transcripts",
        stem,
        asr_result,
    )

    fallback = asr_result["mode"] == "mock_fallback"
    warning = (
        "Audio preprocessing completed, but ASR used mock fallback because "
        "faster-whisper was unavailable."
        if fallback
        else (
            "Phase 2 preprocessing and ASR completed. Diarization and "
            "overlap-aware correction are separate later-phase stages."
        )
    )
    paths: dict[str, str] = {
        "processed_audio": str(processed_path),
        **_stringify_paths(
            {
                "raw_asr_json": raw_paths["json"],
                "raw_asr_markdown": raw_paths["markdown"],
                "raw_asr_metadata": raw_paths["metadata"],
            }
        ),
    }
    result = {
        "mode": (
            "phase2_mock_asr_fallback"
            if fallback
            else "phase2_faster_whisper"
        ),
        "audio_path": str(audio_path),
        "preprocessing": preprocessing,
        "asr": {
            key: value
            for key, value in asr_result.items()
            if key != "segments"
        },
        "asr_segments": asr_result["segments"],
        "speaker_turns": [],
        "overlap_regions": [],
        "transcript": [],
        "summary": None,
        "artifacts": paths,
        "warning": warning,
    }
    manifest = write_json(
        settings.output_dir / "exports" / f"{stem}_manifest.json",
        result,
    )
    result["artifacts"]["manifest"] = str(manifest)
    return result


def run_pipeline(
    *,
    audio_path: str | Path | None = None,
    mock: bool = False,
    denoise: bool = False,
    model_size: str | None = None,
    language: str | None = None,
    device: str = "auto",
    compute_type: str = "default",
) -> dict[str, Any]:
    """Run the deterministic demo or the Phase 2 preprocessing/ASR path."""

    settings = get_settings()
    selected_model = model_size or settings.asr_model_size
    if mock or settings.use_mock_asr:
        return _run_mock_pipeline(
            audio_path=audio_path,
            denoise=denoise,
            model_size=selected_model,
        )
    return _run_phase2_pipeline(
        audio_path=audio_path,
        denoise=denoise,
        model_size=selected_model,
        language=language,
        device=device,
        compute_type=compute_type,
    )
