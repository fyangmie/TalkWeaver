"""TalkWeaver end-to-end overlap-aware ASR correction pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.alignment import align_segments
from backend.asr import transcribe_with_metadata
from backend.config import get_settings
from backend.diarization import diarize_with_metadata
from backend.export import (
    export_corrected_transcript,
    export_diarization,
    export_overlap_regions,
    export_rag_transcript,
    export_raw_transcript,
    export_summary,
    export_temporal_anchor_transcript,
    read_json,
    write_json,
)
from backend.llm_correction import correct_segments
from backend.overlap import detect_overlap_regions
from backend.preprocessing import preprocess_audio
from backend.rag import enrich_segments_with_terms
from backend.summarizer import summarize_segments


def _artifact_paths(
    *,
    raw_paths: dict[str, Path],
    diarization_paths: dict[str, Path],
    overlap_paths: dict[str, Path],
    temporal_paths: dict[str, Path],
    rag_paths: dict[str, Path],
    corrected_paths: dict[str, Path],
    summary_paths: dict[str, Path],
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
        "rag_transcript_json": str(rag_paths["json"]),
        "rag_retrieval_markdown": str(rag_paths["markdown"]),
        "rag_metadata": str(rag_paths["metadata"]),
        "corrected_transcript_json": str(corrected_paths["json"]),
        "corrected_transcript_markdown": str(corrected_paths["markdown"]),
        "summary_json": str(summary_paths["json"]),
        "summary_markdown": str(summary_paths["markdown"]),
    }
    if processed_audio is not None:
        artifacts["processed_audio"] = processed_audio
    return artifacts


def _run_pipeline(
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
    temporal_transcript = align_segments(asr_segments, speaker_turns)
    temporal_paths = export_temporal_anchor_transcript(
        settings.output_dir / "transcripts",
        source_stem,
        temporal_transcript,
        mode=diarization_result["mode"],
    )

    rag_transcript, rag_metadata = enrich_segments_with_terms(
        temporal_transcript,
        directory=settings.knowledge_base_dir,
    )
    rag_paths = export_rag_transcript(
        settings.output_dir / "transcripts",
        source_stem,
        rag_transcript,
        metadata=rag_metadata,
    )

    correction_mock = mock or settings.use_mock_llm
    corrected = correct_segments(
        rag_transcript,
        mock=correction_mock,
        provider=settings.llm_provider,
        openai_api_key=settings.openai_api_key,
        deepseek_api_key=settings.deepseek_api_key,
        qwen_api_key=settings.qwen_api_key,
        openai_model=settings.openai_model,
        deepseek_model=settings.deepseek_model,
        qwen_model=settings.qwen_model,
        openai_base_url=settings.openai_base_url,
        deepseek_base_url=settings.deepseek_base_url,
        qwen_base_url=settings.qwen_base_url,
    )
    correction_mode = (
        "mock_rule_based"
        if correction_mock
        else (
            corrected[0].get("correction_mode", "unknown")
            if corrected
            else "no_segments"
        )
    )
    corrected_paths = export_corrected_transcript(
        settings.output_dir / "corrected_transcripts",
        source_stem,
        corrected,
        mode=correction_mode,
    )
    summary = summarize_segments(corrected)
    summary_paths = export_summary(
        settings.output_dir / "summaries",
        source_stem,
        summary,
    )

    has_mock_components = any(
        str(component_mode).startswith("mock")
        for component_mode in (
            asr_result["mode"],
            diarization_result["mode"],
        )
    )
    uses_rule_based_correction = any(
        "rule_based" in str(segment.get("correction_mode", ""))
        for segment in corrected
    )
    if mock:
        mode = "mock_demo"
        warning = (
            "Mock ASR, diarization, and rule-based correction are "
            "deterministic demo outputs, not real experimental results."
        )
    elif has_mock_components or uses_rule_based_correction:
        mode = "pipeline_with_mock_fallback"
        warning = (
            "The pipeline completed with one or more labeled mock or "
            "rule-based components. Inspect artifact metadata before "
            "evaluation."
        )
    else:
        mode = "full_pipeline"
        warning = (
            "Correction is constrained by transcript evidence and retrieved "
            "terms. Overlap segments remain marked for human review."
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
        "rag": rag_metadata,
        "asr_segments": asr_segments,
        "speaker_turns": speaker_turns,
        "overlap_regions": overlap_regions,
        "temporal_transcript": temporal_transcript,
        "transcript": corrected,
        "summary": summary,
        "artifacts": _artifact_paths(
            raw_paths=raw_paths,
            diarization_paths=diarization_paths,
            overlap_paths=overlap_paths,
            temporal_paths=temporal_paths,
            rag_paths=rag_paths,
            corrected_paths=corrected_paths,
            summary_paths=summary_paths,
            processed_audio=(
                str(processed_path) if processed_path is not None else None
            ),
        ),
        "warning": warning,
    }
    manifest = write_json(
        settings.output_dir
        / "exports"
        / f"{source_stem}_pipeline_manifest.json",
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
    """Run preprocessing through RAG correction and summarization."""

    settings = get_settings()
    selected_model = model_size or settings.asr_model_size
    use_mock = mock or settings.use_mock_asr
    return _run_pipeline(
        audio_path=audio_path,
        mock=use_mock,
        denoise=denoise,
        model_size=selected_model,
        language=language,
        device=device,
        compute_type=compute_type,
    )
