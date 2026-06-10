"""TalkWeaver Phase 3 preprocessing, ASR, and diarization orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.alignment import align_segments
from backend.asr import transcribe_with_metadata
from backend.config import get_settings
from backend.diarization import diarize_with_metadata
from backend.export import (
    export_diarization,
    export_overlap_regions,
    export_raw_transcript,
    export_temporal_anchor_transcript,
    read_json,
    write_json,
)
from backend.overlap import detect_overlap_regions
from backend.preprocessing import preprocess_audio


def _artifact_paths(
    *,
    raw_paths: dict[str, Path],
    diarization_paths: dict[str, Path],
    overlap_paths: dict[str, Path],
    temporal_paths: dict[str, Path],
    processed_audio: str | None,
) -> dict[str, str]:
    artifacts = {
        "raw_asr_json": str(raw_paths["json"]),
        "raw_asr_markdown": str(raw_paths["markdown"]),
        "raw_asr_metadata": str(raw_paths["metadata"]),
        "diarization_json": str(diarization_paths["json"]),
        "diarization_markdown": str(diarization_paths["markdown"]),
        "diarization_metadata": str(diarization_paths["metadata"]),
        "overlap_regions_json": str(overlap_paths["json"]),
        "overlap_warnings_markdown": str(overlap_paths["markdown"]),
        "temporal_anchor_json": str(temporal_paths["json"]),
        "speaker_transcript_markdown": str(temporal_paths["markdown"]),
    }
    if processed_audio is not None:
        artifacts["processed_audio"] = processed_audio
    return artifacts


def _run_phase3_pipeline(
    *,
    audio_path: str | Path | None,
    mock: bool,
    denoise: bool,
    model_size: str,
    language: str | None,
    device: str,
    compute_type: str,
) -> dict[str, Any]:
    if not mock and audio_path is None:
        raise ValueError("An audio path is required outside mock mode.")

    settings = get_settings()
    preprocessing = preprocess_audio(
        audio_path,
        mock=mock,
        denoise=denoise,
    )
    processed_path = (
        None if mock else Path(str(preprocessing["output_path"]))
    )
    asr_source = audio_path if mock else processed_path
    asr_result = transcribe_with_metadata(
        asr_source,
        mock=mock,
        model_size=model_size,
        language=language,
        device=device,
        compute_type=compute_type,
        fallback_to_mock=True,
    )

    source_stem = "mock" if mock else Path(str(audio_path)).stem
    raw_paths = export_raw_transcript(
        settings.output_dir / "transcripts",
        f"{source_stem}_asr" if mock else f"{source_stem}_raw_asr",
        asr_result,
    )
    asr_segments = read_json(raw_paths["json"])

    duration = asr_result.get("duration_seconds")
    diarization_result = diarize_with_metadata(
        processed_path,
        mock=mock or settings.use_mock_diarization,
        hf_token=settings.hf_token,
        fallback_to_mock=True,
        duration_seconds=float(duration) if duration else None,
    )
    diarization_paths = export_diarization(
        settings.output_dir / "diarization",
        f"{source_stem}_diarization",
        diarization_result,
    )

    speaker_turns = diarization_result["turns"]
    overlap_regions = detect_overlap_regions(speaker_turns)
    overlap_paths = export_overlap_regions(
        settings.output_dir / "diarization",
        overlap_regions,
        mode=diarization_result["mode"],
    )
    transcript = align_segments(asr_segments, speaker_turns)
    temporal_paths = export_temporal_anchor_transcript(
        settings.output_dir / "transcripts",
        source_stem,
        transcript,
        mode=diarization_result["mode"],
    )

    has_mock_components = any(
        str(component_mode).startswith("mock")
        for component_mode in (
            asr_result["mode"],
            diarization_result["mode"],
        )
    )
    if mock:
        mode = "mock_demo"
        warning = (
            "Mock/demo ASR and diarization outputs are deterministic and are "
            "not real experimental results."
        )
    elif has_mock_components:
        mode = "phase3_with_mock_fallback"
        warning = (
            "Phase 3 completed with one or more clearly labeled mock "
            "fallbacks. Inspect artifact metadata before evaluation."
        )
    else:
        mode = "phase3_real"
        warning = (
            "Phase 3 completed. Temporal anchors are raw alignment output; "
            "LLM correction and RAG are not run in this phase."
        )

    result = {
        "mode": mode,
        "audio_path": str(audio_path) if audio_path else None,
        "preprocessing": preprocessing,
        "asr": {
            key: value
            for key, value in asr_result.items()
            if key != "segments"
        },
        "diarization": {
            key: value
            for key, value in diarization_result.items()
            if key != "turns"
        },
        "asr_segments": asr_segments,
        "speaker_turns": speaker_turns,
        "overlap_regions": overlap_regions,
        "transcript": transcript,
        "summary": None,
        "artifacts": _artifact_paths(
            raw_paths=raw_paths,
            diarization_paths=diarization_paths,
            overlap_paths=overlap_paths,
            temporal_paths=temporal_paths,
            processed_audio=(
                str(processed_path) if processed_path is not None else None
            ),
        ),
        "warning": warning,
    }
    manifest = write_json(
        settings.output_dir
        / "exports"
        / f"{source_stem}_phase3_manifest.json",
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
    """Run preprocessing, ASR, diarization, overlap, and alignment."""

    settings = get_settings()
    selected_model = model_size or settings.asr_model_size
    use_mock = mock or settings.use_mock_asr
    return _run_phase3_pipeline(
        audio_path=audio_path,
        mock=use_mock,
        denoise=denoise,
        model_size=selected_model,
        language=language,
        device=device,
        compute_type=compute_type,
    )
