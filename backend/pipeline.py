"""TalkWeaver pipeline orchestration for Phase 1 mock mode."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.alignment import align_segments
from backend.asr import transcribe
from backend.config import get_settings
from backend.diarization import diarize
from backend.export import write_json, write_transcript_markdown
from backend.llm_correction import correct_segments
from backend.overlap import detect_overlap_regions
from backend.preprocessing import preprocess_audio
from backend.summarizer import summarize_segments


def run_pipeline(
    *,
    audio_path: str | Path | None = None,
    mock: bool = False,
) -> dict[str, Any]:
    """Run the available pipeline and persist intermediate artifacts."""

    settings = get_settings()
    effective_mock = (
        mock
        or settings.use_mock_asr
        or settings.use_mock_diarization
    )
    if not effective_mock:
        raise RuntimeError(
            "Phase 1 supports the complete pipeline in mock mode only. "
            "Use --mock; real integrations begin in Phase 2."
        )

    preprocessing = preprocess_audio(audio_path, mock=True)
    asr_segments = transcribe(audio_path, mock=True)
    speaker_turns = diarize(audio_path, mock=True)
    overlap_regions = detect_overlap_regions(speaker_turns)
    aligned = align_segments(asr_segments, speaker_turns, overlap_regions)
    corrected = correct_segments(aligned, mock=True)
    summary = summarize_segments(corrected)

    paths = {
        "asr": write_json(
            settings.output_dir / "transcripts" / "mock_asr.json",
            asr_segments,
        ),
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
        "asr_segments": asr_segments,
        "speaker_turns": speaker_turns,
        "overlap_regions": overlap_regions,
        "transcript": corrected,
        "summary": summary,
        "artifacts": {name: str(path) for name, path in paths.items()},
        "warning": "Mock/demo output is not a real experimental result.",
    }
    write_json(
        settings.output_dir / "exports" / "mock_pipeline_manifest.json",
        result,
    )
    return result
