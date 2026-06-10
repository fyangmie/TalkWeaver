#!/usr/bin/env python3
"""Benchmark TalkWeaver pipeline stages with labeled mock support."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, Callable


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.alignment import align_segments
from backend.asr import transcribe_with_metadata
from backend.config import get_settings
from backend.diarization import diarize_with_metadata
from backend.llm_correction import correct_segments
from backend.overlap import detect_overlap_regions
from backend.preprocessing import preprocess_audio
from backend.rag import enrich_segments_with_terms
from backend.summarizer import summarize_segments


DEFAULT_OUTPUT = (
    ROOT_DIR / "experiments" / "results" / "latency_results.csv"
)


def _timed(function: Callable[[], Any]) -> tuple[Any, float]:
    start = perf_counter()
    result = function()
    return result, perf_counter() - start


def benchmark_pipeline_stages(
    *,
    audio_path: str | Path | None = None,
    mock: bool = False,
    denoise: bool = False,
) -> list[dict[str, Any]]:
    """Measure each pipeline stage using actual elapsed wall-clock time."""

    if not mock and audio_path is None:
        raise ValueError("Real latency benchmarking requires --audio.")
    settings = get_settings()
    rows: list[dict[str, Any]] = []

    preprocessing, elapsed = _timed(
        lambda: preprocess_audio(audio_path, mock=mock, denoise=denoise)
    )
    rows.append(("preprocessing", elapsed, preprocessing["mode"]))
    processed_path = None if mock else Path(str(preprocessing["output_path"]))

    asr, elapsed = _timed(
        lambda: transcribe_with_metadata(
            audio_path if mock else processed_path,
            mock=mock,
            model_size=settings.asr_model_size,
            fallback_to_mock=True,
        )
    )
    rows.append(("asr", elapsed, asr["mode"]))

    diarization, elapsed = _timed(
        lambda: diarize_with_metadata(
            processed_path,
            mock=mock or settings.use_mock_diarization,
            hf_token=settings.hf_token,
            fallback_to_mock=True,
            duration_seconds=float(asr.get("duration_seconds") or 0) or None,
        )
    )
    rows.append(("diarization", elapsed, diarization["mode"]))

    aligned, elapsed = _timed(
        lambda: align_segments(asr["segments"], diarization["turns"])
    )
    rows.append(("alignment", elapsed, "deterministic_timestamp_midpoint"))

    _overlap, elapsed = _timed(
        lambda: detect_overlap_regions(diarization["turns"])
    )
    rows.append(("overlap_detection", elapsed, "interval_intersection"))

    enriched_result, elapsed = _timed(
        lambda: enrich_segments_with_terms(
            aligned,
            directory=settings.knowledge_base_dir,
        )
    )
    enriched, _metadata = enriched_result
    rows.append(("rag_retrieval", elapsed, "local_tfidf"))

    correction_mock = mock or settings.use_mock_llm
    corrected, elapsed = _timed(
        lambda: correct_segments(
            enriched,
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
    )
    correction_mode = (
        corrected[0].get("correction_mode", "no_segments")
        if corrected
        else "no_segments"
    )
    rows.append(("llm_correction", elapsed, correction_mode))

    _summary, elapsed = _timed(lambda: summarize_segments(corrected))
    rows.append(("summary", elapsed, "deterministic_extractive"))

    return [
        {
            "stage": stage,
            "latency_seconds": round(seconds, 6),
            "is_mock": (
                "true"
                if mock or str(mode).startswith("mock")
                else "false"
            ),
            "mode": mode,
            "notes": (
                "Measured elapsed time on deterministic mock/demo data."
                if mock
                else "Measured elapsed time; inspect mode for fallbacks."
            ),
        }
        for stage, seconds, mode in rows
    ]


def write_latency_results(
    rows: list[dict[str, Any]],
    output_path: str | Path = DEFAULT_OUTPUT,
) -> Path:
    """Write per-stage latency rows to CSV."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--denoise", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    try:
        rows = benchmark_pipeline_stages(
            audio_path=args.audio,
            mock=args.mock,
            denoise=args.denoise,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    output = write_latency_results(rows, args.output)
    for row in rows:
        print(
            f"{row['stage']}={row['latency_seconds']:.6f}s "
            f"mode={row['mode']}"
        )
    print(f"Wrote latency results: {output}")
    if args.mock:
        print("These timings are mock/demo execution measurements.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
