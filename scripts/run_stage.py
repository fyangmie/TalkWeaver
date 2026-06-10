#!/usr/bin/env python3
"""Run one TalkWeaver pipeline stage."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

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
)
from backend.llm_correction import correct_segments
from backend.overlap import detect_overlap_regions
from backend.preprocessing import preprocess_audio
from backend.rag import enrich_segments_with_terms
from backend.summarizer import summarize_segments


STAGES = (
    "preprocess",
    "preprocessing",
    "asr",
    "diarization",
    "align",
    "overlap",
    "rag",
    "correction",
    "summarize",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument(
        "--denoise",
        action="store_true",
        help="Apply noisereduce when the optional package is installed.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output WAV path for the preprocessing stage.",
    )
    parser.add_argument(
        "--asr-json",
        type=Path,
        help="Existing raw ASR JSON for the align stage.",
    )
    parser.add_argument(
        "--diarization-json",
        type=Path,
        help="Existing speaker-turn JSON for align or overlap.",
    )
    parser.add_argument(
        "--transcript-json",
        type=Path,
        help="Existing temporal-anchor or corrected transcript JSON.",
    )
    parser.add_argument("--model-size", help="faster-whisper model size.")
    parser.add_argument("--language", help="Optional ASR language code.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--compute-type", default="default")
    return parser


def _asr_stage(args: argparse.Namespace) -> dict[str, Any]:
    settings = get_settings()
    model_size = args.model_size or settings.asr_model_size
    result = transcribe_with_metadata(
        args.audio,
        mock=args.mock,
        model_size=model_size,
        language=args.language,
        device=args.device,
        compute_type=args.compute_type,
        fallback_to_mock=True,
    )
    stem = (
        "mock_asr"
        if args.mock or args.audio is None
        else f"{args.audio.stem}_raw_asr"
    )
    paths = export_raw_transcript(
        settings.output_dir / "transcripts",
        stem,
        result,
    )
    return {
        **result,
        "artifacts": {name: str(path) for name, path in paths.items()},
    }


def _diarization_stage(args: argparse.Namespace) -> dict[str, Any]:
    settings = get_settings()
    result = diarize_with_metadata(
        args.audio,
        mock=args.mock or settings.use_mock_diarization,
        hf_token=settings.hf_token,
        fallback_to_mock=True,
    )
    stem = (
        "mock_diarization"
        if result["mode"].startswith("mock")
        else f"{args.audio.stem}_diarization"
    )
    paths = export_diarization(
        settings.output_dir / "diarization",
        stem,
        result,
    )
    return {
        **result,
        "artifacts": {name: str(path) for name, path in paths.items()},
    }


def _ensure_asr_json(args: argparse.Namespace) -> Path:
    if args.asr_json is not None:
        if not args.asr_json.exists():
            raise FileNotFoundError(f"ASR JSON not found: {args.asr_json}")
        return args.asr_json

    settings = get_settings()
    default_path = settings.output_dir / "transcripts" / "mock_asr.json"
    if args.mock:
        if not default_path.exists():
            payload = _asr_stage(args)
            return Path(payload["artifacts"]["json"])
        return default_path

    if args.audio is None:
        raise ValueError(
            "Provide --asr-json or --audio outside mock alignment mode."
        )
    inferred = (
        settings.output_dir
        / "transcripts"
        / f"{args.audio.stem}_raw_asr.json"
    )
    if not inferred.exists():
        raise FileNotFoundError(
            f"ASR JSON not found: {inferred}. Run the ASR stage first."
        )
    return inferred


def _ensure_diarization_json(
    args: argparse.Namespace,
) -> tuple[Path, str]:
    if args.diarization_json is not None:
        if not args.diarization_json.exists():
            raise FileNotFoundError(
                f"Diarization JSON not found: {args.diarization_json}"
            )
        return args.diarization_json, "artifact"

    settings = get_settings()
    default_path = (
        settings.output_dir / "diarization" / "mock_diarization.json"
    )
    if args.mock:
        if not default_path.exists():
            payload = _diarization_stage(args)
            return Path(payload["artifacts"]["json"]), payload["mode"]
        return default_path, "mock_demo"

    if args.audio is None:
        raise ValueError(
            "Provide --diarization-json or --audio outside mock mode."
        )
    inferred = (
        settings.output_dir
        / "diarization"
        / f"{args.audio.stem}_diarization.json"
    )
    if not inferred.exists():
        payload = _diarization_stage(args)
        return Path(payload["artifacts"]["json"]), payload["mode"]
    return inferred, "artifact"


def _overlap_stage(args: argparse.Namespace) -> dict[str, Any]:
    settings = get_settings()
    diarization_path, mode = _ensure_diarization_json(args)
    speaker_turns = read_json(diarization_path)
    regions = detect_overlap_regions(speaker_turns)
    paths = export_overlap_regions(
        settings.output_dir / "diarization",
        regions,
        mode=mode,
    )
    return {
        "mode": mode,
        "speaker_turns_path": str(diarization_path),
        "regions": regions,
        "artifacts": {name: str(path) for name, path in paths.items()},
    }


def _align_stage(args: argparse.Namespace) -> dict[str, Any]:
    settings = get_settings()
    asr_path = _ensure_asr_json(args)
    diarization_path, mode = _ensure_diarization_json(args)
    asr_segments = read_json(asr_path)
    speaker_turns = read_json(diarization_path)
    overlap_regions = detect_overlap_regions(speaker_turns)
    transcript = align_segments(asr_segments, speaker_turns)

    stem = "mock" if args.mock else asr_path.stem.removesuffix("_raw_asr")
    temporal_paths = export_temporal_anchor_transcript(
        settings.output_dir / "transcripts",
        stem,
        transcript,
        mode=mode,
    )
    overlap_paths = export_overlap_regions(
        settings.output_dir / "diarization",
        overlap_regions,
        mode=mode,
    )
    paths = {
        "temporal_anchor_json": temporal_paths["json"],
        "speaker_transcript_markdown": temporal_paths["markdown"],
        "overlap_regions_json": overlap_paths["json"],
        "overlap_warnings_markdown": overlap_paths["markdown"],
    }
    return {
        "mode": mode,
        "asr_path": str(asr_path),
        "diarization_path": str(diarization_path),
        "segments": transcript,
        "overlap_regions": overlap_regions,
        "artifacts": {name: str(path) for name, path in paths.items()},
    }


def _ensure_temporal_json(args: argparse.Namespace) -> Path:
    if args.transcript_json is not None:
        if not args.transcript_json.exists():
            raise FileNotFoundError(
                f"Transcript JSON not found: {args.transcript_json}"
            )
        return args.transcript_json

    settings = get_settings()
    default_path = (
        settings.output_dir / "transcripts" / "mock_temporal_anchor.json"
    )
    if args.mock:
        if not default_path.exists():
            payload = _align_stage(args)
            return Path(payload["artifacts"]["temporal_anchor_json"])
        return default_path

    if args.audio is None:
        raise ValueError(
            "Provide --transcript-json or --audio outside mock mode."
        )
    inferred = (
        settings.output_dir
        / "transcripts"
        / f"{args.audio.stem}_temporal_anchor.json"
    )
    if not inferred.exists():
        payload = _align_stage(args)
        return Path(payload["artifacts"]["temporal_anchor_json"])
    return inferred


def _source_stem(args: argparse.Namespace, source_path: Path) -> str:
    if args.mock:
        return "mock"
    stem = source_path.stem
    for suffix in (
        "_temporal_anchor",
        "_rag_enriched",
        "_corrected",
    ):
        stem = stem.removesuffix(suffix)
    return stem


def _rag_stage(args: argparse.Namespace) -> dict[str, Any]:
    settings = get_settings()
    transcript_path = _ensure_temporal_json(args)
    segments = read_json(transcript_path)
    enriched, metadata = enrich_segments_with_terms(
        segments,
        directory=settings.knowledge_base_dir,
    )
    stem = _source_stem(args, transcript_path)
    paths = export_rag_transcript(
        settings.output_dir / "transcripts",
        stem,
        enriched,
        metadata=metadata,
    )
    return {
        "mode": metadata["mode"],
        "transcript_path": str(transcript_path),
        "segments": enriched,
        "metadata": metadata,
        "artifacts": {name: str(path) for name, path in paths.items()},
    }


def _correction_stage(args: argparse.Namespace) -> dict[str, Any]:
    settings = get_settings()
    rag_result = _rag_stage(args)
    correction_mock = args.mock or settings.use_mock_llm
    corrected = correct_segments(
        rag_result["segments"],
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
    transcript_path = Path(rag_result["transcript_path"])
    stem = _source_stem(args, transcript_path)
    mode = (
        "mock_rule_based"
        if correction_mock
        else (
            corrected[0].get("correction_mode", "no_segments")
            if corrected
            else "no_segments"
        )
    )
    paths = export_corrected_transcript(
        settings.output_dir / "corrected_transcripts",
        stem,
        corrected,
        mode=mode,
    )
    return {
        "mode": mode,
        "segments": corrected,
        "rag": rag_result["metadata"],
        "artifacts": {name: str(path) for name, path in paths.items()},
    }


def _summarize_stage(args: argparse.Namespace) -> dict[str, Any]:
    settings = get_settings()
    if args.transcript_json is not None:
        if not args.transcript_json.exists():
            raise FileNotFoundError(
                f"Transcript JSON not found: {args.transcript_json}"
            )
        segments = read_json(args.transcript_json)
        stem = _source_stem(args, args.transcript_json)
        correction_artifacts: dict[str, str] = {}
    else:
        correction_result = _correction_stage(args)
        segments = correction_result["segments"]
        stem = "mock" if args.mock else Path(str(args.audio)).stem
        correction_artifacts = {
            "corrected_transcript_json": correction_result["artifacts"][
                "json"
            ],
            "corrected_transcript_markdown": correction_result["artifacts"][
                "markdown"
            ],
        }

    summary = summarize_segments(segments)
    paths = export_summary(
        settings.output_dir / "summaries",
        stem,
        summary,
    )
    return {
        **summary,
        "artifacts": {
            **correction_artifacts,
            "summary_json": str(paths["json"]),
            "summary_markdown": str(paths["markdown"]),
        },
    }


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    stage = "preprocess" if args.stage == "preprocessing" else args.stage

    try:
        if stage == "preprocess":
            payload = preprocess_audio(
                args.audio,
                mock=args.mock,
                denoise=args.denoise,
                output_path=args.output,
            )
        elif stage == "asr":
            payload = _asr_stage(args)
        elif stage == "diarization":
            payload = _diarization_stage(args)
        elif stage == "align":
            payload = _align_stage(args)
        elif stage == "overlap":
            payload = _overlap_stage(args)
        elif stage == "rag":
            payload = _rag_stage(args)
        elif stage == "correction":
            payload = _correction_stage(args)
        else:
            payload = _summarize_stage(args)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"TalkWeaver stage error: {exc}", file=sys.stderr)
        return 2

    print(json.dumps({"stage": stage, "output": payload}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
