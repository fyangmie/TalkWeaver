#!/usr/bin/env python3
"""Run real faster-whisper ASR over a small formal evaluation manifest."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from time import perf_counter
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.asr import REAL_ASR_DEPENDENCY_ERROR  # noqa: E402
from experiments.metrics.text_metrics import evaluate_text  # noqa: E402
from experiments.metrics.text_normalization import (  # noqa: E402
    canonical_language,
    is_mandarin_language,
)


OUTPUT_COLUMNS = [
    "clip_id",
    "dataset_name",
    "language",
    "model_name",
    "device",
    "compute_type",
    "duration_seconds",
    "runtime_seconds",
    "rtf",
    "metric_name",
    "error_rate",
    "reference_text",
    "hypothesis_text",
    "normalized_reference",
    "normalized_hypothesis",
    "prediction_json_path",
    "prediction_txt_path",
    "notes",
]


def project_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def model_language(language: str | None) -> str | None:
    """Map manifest language identifiers to faster-whisper identifiers."""

    if is_mandarin_language(language):
        return "zh"
    normalized = canonical_language(language)
    if normalized.startswith("en"):
        return "en"
    if normalized.startswith("fr"):
        return "fr"
    return normalized or None


def load_faster_whisper_model(
    model_name: str,
    *,
    device: str,
    compute_type: str,
) -> Any:
    """Load a real faster-whisper model without any mock fallback."""

    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(REAL_ASR_DEPENDENCY_ERROR) from exc
    return WhisperModel(
        model_name,
        device=device,
        compute_type=compute_type,
    )


def _optional_float(value: Any) -> float | None:
    return None if value is None else round(float(value), 6)


def _serialize_word(word: Any) -> dict[str, Any]:
    payload = {
        "word": str(word.word).strip(),
        "start": _optional_float(word.start),
        "end": _optional_float(word.end),
    }
    probability = getattr(word, "probability", None)
    if probability is not None:
        payload["probability"] = round(float(probability), 6)
    return payload


def _serialize_segment(segment: Any) -> dict[str, Any]:
    payload = {
        "start": round(float(segment.start), 6),
        "end": round(float(segment.end), 6),
        "text": str(segment.text).strip(),
        "words": [
            _serialize_word(word)
            for word in (getattr(segment, "words", None) or [])
        ],
    }
    for source_name, output_name in (
        ("avg_logprob", "avg_logprob"),
        ("no_speech_prob", "no_speech_probability"),
    ):
        value = getattr(segment, source_name, None)
        if value is not None:
            payload[output_name] = round(float(value), 6)
    return payload


def transcribe_real_clip(
    model: Any,
    audio_path: Path,
    *,
    language: str | None,
) -> tuple[dict[str, Any], float]:
    """Run one real inference and include generator materialization in time."""

    started = perf_counter()
    segment_generator, info = model.transcribe(
        str(audio_path),
        beam_size=5,
        language=model_language(language),
        word_timestamps=True,
        vad_filter=True,
    )
    segments = [_serialize_segment(segment) for segment in segment_generator]
    runtime_seconds = perf_counter() - started
    hypothesis = " ".join(
        segment["text"] for segment in segments if segment["text"]
    ).strip()
    result = {
        "asr_mode": "real",
        "backend": "faster_whisper",
        "language_requested": model_language(language),
        "language_detected": getattr(info, "language", None),
        "language_probability": _optional_float(
            getattr(info, "language_probability", None)
        ),
        "duration_seconds_backend": _optional_float(
            getattr(info, "duration", None)
        ),
        "duration_after_vad_seconds": _optional_float(
            getattr(info, "duration_after_vad", None)
        ),
        "hypothesis_text": hypothesis,
        "segments": segments,
    }
    return result, runtime_seconds


def _safe_stem(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def _write_prediction(
    prediction_dir: Path,
    *,
    model_name: str,
    row: dict[str, str],
    reference_text: str,
    inference: dict[str, Any],
    metrics: dict[str, str | float],
    runtime_seconds: float,
    model_load_seconds: float,
    device: str,
    compute_type: str,
) -> tuple[Path, Path]:
    prediction_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{_safe_stem(model_name)}__{_safe_stem(row['clip_id'])}"
    json_path = prediction_dir / f"{stem}.json"
    txt_path = prediction_dir / f"{stem}.txt"
    payload = {
        "benchmark_scope": "small-subset formal evaluation",
        "is_mock": False,
        "clip_id": row["clip_id"],
        "dataset_name": row["dataset_name"],
        "language": row["language"],
        "audio_path": row["audio_path"],
        "transcript_path": row["transcript_path"],
        "model_name": model_name,
        "device": device,
        "compute_type": compute_type,
        "model_load_seconds": round(model_load_seconds, 6),
        "runtime_seconds": round(runtime_seconds, 6),
        "reference_text": reference_text,
        "hypothesis_text": inference["hypothesis_text"],
        "normalized_reference": metrics["normalized_reference"],
        "normalized_hypothesis": metrics["normalized_hypothesis"],
        "metric_name": metrics["metric_name"],
        "error_rate": round(float(metrics["error_rate"]), 6),
        "asr": inference,
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    txt_path.write_text(
        inference["hypothesis_text"].strip() + "\n",
        encoding="utf-8",
    )
    return json_path, txt_path


def load_manifest_rows(manifest: Path) -> list[dict[str, str]]:
    with manifest.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run_benchmark(
    *,
    manifest: str | Path,
    models: list[str],
    device: str,
    compute_type: str,
    output: str | Path,
    predictions_dir: str | Path,
) -> list[dict[str, Any]]:
    """Run real ASR for each valid manifest row and write audit artifacts."""

    manifest_path = project_path(manifest)
    output_path = project_path(output)
    prediction_root = project_path(predictions_dir)
    rows = load_manifest_rows(manifest_path)
    valid_rows: list[tuple[dict[str, str], Path, Path]] = []
    for row in rows:
        audio_path = project_path(row.get("audio_path", ""))
        transcript_path = project_path(row.get("transcript_path", ""))
        if not audio_path.is_file() or not transcript_path.is_file():
            print(
                f"Skipping {row.get('clip_id', '<unknown>')}: missing "
                "audio_path or transcript_path.",
                file=sys.stderr,
            )
            continue
        valid_rows.append((row, audio_path, transcript_path))
    if not valid_rows:
        raise RuntimeError("No manifest rows have real audio and transcripts.")

    results: list[dict[str, Any]] = []
    for model_name in models:
        load_started = perf_counter()
        model = load_faster_whisper_model(
            model_name,
            device=device,
            compute_type=compute_type,
        )
        model_load_seconds = perf_counter() - load_started
        print(
            f"Loaded model={model_name} in {model_load_seconds:.3f}s "
            f"for {len(valid_rows)} clips."
        )
        for index, (row, audio_path, transcript_path) in enumerate(
            valid_rows,
            start=1,
        ):
            reference_text = transcript_path.read_text(
                encoding="utf-8"
            ).strip()
            inference, runtime_seconds = transcribe_real_clip(
                model,
                audio_path,
                language=row.get("language"),
            )
            if inference.get("asr_mode") != "real":
                raise RuntimeError(
                    "ASR benchmark received a non-real inference result."
                )
            metrics = evaluate_text(
                reference_text,
                str(inference["hypothesis_text"]),
                row.get("language"),
            )
            duration_seconds = float(row["duration_seconds"])
            rtf = (
                runtime_seconds / duration_seconds
                if duration_seconds > 0
                else 0.0
            )
            json_path, txt_path = _write_prediction(
                prediction_root,
                model_name=model_name,
                row=row,
                reference_text=reference_text,
                inference=inference,
                metrics=metrics,
                runtime_seconds=runtime_seconds,
                model_load_seconds=model_load_seconds,
                device=device,
                compute_type=compute_type,
            )
            result = {
                "clip_id": row["clip_id"],
                "dataset_name": row["dataset_name"],
                "language": row["language"],
                "model_name": model_name,
                "device": device,
                "compute_type": compute_type,
                "duration_seconds": round(duration_seconds, 6),
                "runtime_seconds": round(runtime_seconds, 6),
                "rtf": round(rtf, 6),
                "metric_name": metrics["metric_name"],
                "error_rate": round(float(metrics["error_rate"]), 6),
                "reference_text": reference_text,
                "hypothesis_text": inference["hypothesis_text"],
                "normalized_reference": metrics["normalized_reference"],
                "normalized_hypothesis": metrics["normalized_hypothesis"],
                "prediction_json_path": display_path(json_path),
                "prediction_txt_path": display_path(txt_path),
                "notes": (
                    "Real faster-whisper inference on the small-subset "
                    "formal evaluation manifest; model initialization "
                    f"({model_load_seconds:.3f}s) excluded from per-clip "
                    "runtime and RTF."
                ),
            }
            results.append(result)
            print(
                f"[{model_name} {index}/{len(valid_rows)}] "
                f"{row['clip_id']} {metrics['metric_name']}="
                f"{float(metrics['error_rate']):.4f} "
                f"RTF={rtf:.4f}"
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=OUTPUT_COLUMNS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(results)
    print(f"Wrote {len(results)} real ASR rows: {output_path}")
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        run_benchmark(
            manifest=args.manifest,
            models=args.models,
            device=args.device,
            compute_type=args.compute_type,
            output=args.output,
            predictions_dir=args.predictions_dir,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"ASR benchmark failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
