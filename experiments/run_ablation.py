#!/usr/bin/env python3
"""Run the TalkWeaver ablation study on explicit reference data."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.llm_correction import correct_segments
from backend.pipeline import run_pipeline
from experiments.evaluate_latency import (
    benchmark_pipeline_stages,
    write_latency_results,
)
from experiments.evaluate_terms import compare_term_recovery
from experiments.evaluate_wder import overlap_flag_error, simplified_wder
from experiments.evaluate_wer import extract_text, word_error_rate


RESULT_PATH = ROOT_DIR / "experiments" / "results" / "ablation_results.csv"
TERM_RESULT_PATH = (
    ROOT_DIR / "experiments" / "results" / "term_error_results.csv"
)
LATENCY_RESULT_PATH = (
    ROOT_DIR / "experiments" / "results" / "latency_results.csv"
)

MOCK_REFERENCE_TEXT = (
    "We use pyannote for diarization. "
    "The RAG glossary can reduce term errors. "
    "We should compare WER and DER."
)
MOCK_REFERENCE_SEGMENTS = [
    {
        "start": 0.0,
        "end": 3.2,
        "speaker": "SPEAKER_00",
        "speakers": ["SPEAKER_00"],
        "overlap": False,
    },
    {
        "start": 3.0,
        "end": 3.2,
        "speaker": "OVERLAP",
        "speakers": ["SPEAKER_00", "SPEAKER_01"],
        "overlap": True,
    },
    {
        "start": 3.25,
        "end": 6.5,
        "speaker": "SPEAKER_01",
        "speakers": ["SPEAKER_01"],
        "overlap": False,
    },
    {
        "start": 6.6,
        "end": 9.4,
        "speaker": "SPEAKER_00",
        "speakers": ["SPEAKER_00"],
        "overlap": False,
    },
]

PIPELINES = (
    "Whisper only",
    "+ preprocessing",
    "+ diarization + alignment",
    "+ structured LLM correction",
    "+ RAG glossary",
    "+ overlap-aware correction",
)


def _raw_asr_anchors(
    segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "start": segment["start"],
            "end": segment["end"],
            "speaker": "UNKNOWN",
            "speakers": [],
            "overlap": False,
            "raw_text": segment["text"],
        }
        for segment in segments
    ]


def _count_hallucinated_corrections(
    segments: list[dict[str, Any]],
) -> int:
    """Count segments whose correction adds unsupported lexical material."""

    from backend.llm_correction import validate_corrected_text

    count = 0
    for segment in segments:
        valid, _reason = validate_corrected_text(
            str(segment.get("raw_text", "")),
            str(segment.get("corrected_text", "")),
            [str(term) for term in segment.get("retrieved_terms", [])],
        )
        count += not valid
    return count


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def generate_mock_ablation(
    *,
    result_path: str | Path = RESULT_PATH,
    term_result_path: str | Path = TERM_RESULT_PATH,
    latency_result_path: str | Path = LATENCY_RESULT_PATH,
) -> list[dict[str, Any]]:
    """Compute deterministic demonstration metrics from the mock pipeline."""

    result = run_pipeline(mock=True)
    raw_text = extract_text(result["asr_segments"], field="text")
    raw_anchors = _raw_asr_anchors(result["asr_segments"])
    aligned = result["temporal_transcript"]
    structured_without_rag = correct_segments(aligned, mock=True)
    structured_text = extract_text(
        structured_without_rag,
        field="corrected_text",
    )
    corrected = result["transcript"]
    corrected_text = extract_text(corrected, field="corrected_text")

    latency_rows = benchmark_pipeline_stages(mock=True)
    write_latency_results(latency_rows, latency_result_path)
    latency = {
        row["stage"]: float(row["latency_seconds"])
        for row in latency_rows
    }
    cumulative = {
        PIPELINES[0]: latency["asr"],
        PIPELINES[1]: latency["preprocessing"] + latency["asr"],
        PIPELINES[2]: sum(
            latency[name]
            for name in (
                "preprocessing",
                "asr",
                "diarization",
                "alignment",
                "overlap_detection",
            )
        ),
        PIPELINES[3]: 0.0,
        PIPELINES[4]: 0.0,
        PIPELINES[5]: 0.0,
    }
    cumulative[PIPELINES[3]] = (
        cumulative[PIPELINES[2]] + latency["llm_correction"]
    )
    cumulative[PIPELINES[4]] = (
        cumulative[PIPELINES[2]]
        + latency["rag_retrieval"]
        + latency["llm_correction"]
    )
    cumulative[PIPELINES[5]] = cumulative[PIPELINES[4]]

    variants = [
        (PIPELINES[0], raw_text, raw_anchors, []),
        (PIPELINES[1], raw_text, raw_anchors, []),
        (PIPELINES[2], raw_text, aligned, []),
        (
            PIPELINES[3],
            structured_text,
            structured_without_rag,
            structured_without_rag,
        ),
        (PIPELINES[4], corrected_text, corrected, corrected),
        (PIPELINES[5], corrected_text, corrected, corrected),
    ]
    rows: list[dict[str, Any]] = []
    for pipeline, text, anchors, corrected_segments in variants:
        term_metrics = compare_term_recovery(
            MOCK_REFERENCE_TEXT,
            {pipeline: text},
        )[0]
        rows.append(
            {
                "pipeline": pipeline,
                "wer": round(
                    word_error_rate(
                        MOCK_REFERENCE_TEXT,
                        text,
                        prefer_jiwer=False,
                    ),
                    4,
                ),
                "speaker_error_or_wder": round(
                    simplified_wder(MOCK_REFERENCE_SEGMENTS, anchors),
                    4,
                ),
                "term_error_rate": round(
                    float(term_metrics["term_error_rate"]),
                    4,
                ),
                "overlap_error": round(
                    overlap_flag_error(MOCK_REFERENCE_SEGMENTS, anchors),
                    4,
                ),
                "hallucinated_corrections": (
                    _count_hallucinated_corrections(corrected_segments)
                    if corrected_segments
                    else 0
                ),
                "latency_seconds": round(cumulative[pipeline], 6),
                "is_mock": "true",
                "notes": (
                    "Deterministic mock/demo metric calculated from the "
                    "built-in reference; not a real model-performance claim. "
                    "Speaker error is a simplified temporal-anchor WDER "
                    "approximation."
                ),
            }
        )

    term_rows = compare_term_recovery(
        MOCK_REFERENCE_TEXT,
        {
            "Whisper only": raw_text,
            "Structured LLM correction": structured_text,
            "Structured LLM correction + RAG glossary": corrected_text,
        },
    )
    serializable_term_rows = [
        {
            "pipeline": row["pipeline"],
            "term_error_rate": round(row["term_error_rate"], 4),
            "precision": round(row["precision"], 4),
            "recall": round(row["recall"], 4),
            "required_terms": "; ".join(row["required_terms"]),
            "missing_terms": "; ".join(row["missing_terms"]),
            "is_mock": "true",
            "notes": "Deterministic mock/demo domain-term comparison.",
        }
        for row in term_rows
    ]
    _write_csv(Path(result_path), rows)
    _write_csv(Path(term_result_path), serializable_term_rows)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    args = parser.parse_args()
    if not args.mock:
        parser.error(
            "Real ablation requires a frozen manifest and reference "
            "annotations. Use the individual evaluators, then replace the "
            "mock CSV with reference-backed rows."
        )

    rows = generate_mock_ablation(result_path=args.output)
    print(f"Wrote mock ablation results: {args.output}")
    print(f"Wrote term comparison: {TERM_RESULT_PATH}")
    print(f"Wrote stage latency: {LATENCY_RESULT_PATH}")
    print(f"Rows={len(rows)} is_mock=true")
    print(
        "All values are deterministic demonstration metrics and must not be "
        "reported as real experimental performance."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
