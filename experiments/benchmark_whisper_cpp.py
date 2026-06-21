#!/usr/bin/env python3
"""Run a local whisper.cpp Level-1 ASR benchmark when available."""

from __future__ import annotations

import argparse
import csv
import shutil
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from time import perf_counter
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.metrics.text_metrics import evaluate_cleaned_wer, evaluate_text  # noqa: E402
from scripts.dataset_utils import resolve_repo_path  # noqa: E402


OUTPUT_COLUMNS = [
    "clip_id",
    "dataset_name",
    "language",
    "model_name",
    "model_path",
    "executable",
    "status",
    "duration_seconds",
    "runtime_seconds",
    "rtf",
    "metric_name",
    "error_rate",
    "cleaned_metric_name",
    "cleaned_error_rate",
    "reference_text",
    "hypothesis_text",
    "claim_level",
    "notes",
]

SUMMARY_COLUMNS = [
    "model_name",
    "dataset_name",
    "language",
    "status",
    "num_rows",
    "mean_error_rate",
    "mean_cleaned_error_rate",
    "mean_rtf",
    "notes",
]


DEFAULT_MODEL_SPECS = (
    "tiny=models/whisper.cpp/ggml-tiny.bin",
    "base=models/whisper.cpp/ggml-base.bin",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def find_executable(explicit: str | None = None) -> str:
    if explicit:
        resolved = shutil.which(explicit) or explicit
        return resolved if Path(resolved).exists() or shutil.which(resolved) else ""
    for candidate in ("whisper-cli", "whisper-cpp", "whisper.cpp", "main"):
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    return ""


def parse_model_specs(values: list[str] | None) -> list[tuple[str, Path]]:
    specs = values or list(DEFAULT_MODEL_SPECS)
    parsed: list[tuple[str, Path]] = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(
                f"Model spec {spec!r} must be formatted as name=path."
            )
        name, path = spec.split("=", 1)
        parsed.append((name.strip(), resolve_repo_path(path.strip())))
    return parsed


def _read_text(path: str) -> str:
    return resolve_repo_path(path).read_text(encoding="utf-8").strip()


def _skipped_row(
    manifest_row: dict[str, str],
    *,
    model_name: str,
    model_path: Path,
    executable: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "clip_id": manifest_row.get("clip_id", ""),
        "dataset_name": manifest_row.get("dataset_name", ""),
        "language": manifest_row.get("language", ""),
        "model_name": model_name,
        "model_path": str(model_path),
        "executable": executable,
        "status": "skipped",
        "duration_seconds": manifest_row.get("duration_seconds", ""),
        "runtime_seconds": "",
        "rtf": "",
        "metric_name": "",
        "error_rate": "",
        "cleaned_metric_name": "",
        "cleaned_error_rate": "",
        "reference_text": "",
        "hypothesis_text": "",
        "claim_level": "whisper_cpp_level1_local_or_skipped",
        "notes": reason,
    }


def _run_one(
    manifest_row: dict[str, str],
    *,
    model_name: str,
    model_path: Path,
    executable: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    audio_path = resolve_repo_path(manifest_row["audio_path"])
    reference_text = _read_text(manifest_row["transcript_path"])
    duration = float(manifest_row.get("duration_seconds") or 0.0)
    with tempfile.TemporaryDirectory() as directory:
        output_prefix = Path(directory) / "hypothesis"
        command = [
            executable,
            "-m",
            str(model_path),
            "-f",
            str(audio_path),
            "-otxt",
            "-of",
            str(output_prefix),
            "-nt",
        ]
        started = perf_counter()
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        runtime = perf_counter() - started
        hypothesis_path = output_prefix.with_suffix(".txt")
        if completed.returncode != 0:
            stderr = completed.stderr.strip().replace("\n", " ")[:500]
            return _skipped_row(
                manifest_row,
                model_name=model_name,
                model_path=model_path,
                executable=executable,
                reason=f"whisper.cpp returned {completed.returncode}: {stderr}",
            )
        if not hypothesis_path.exists():
            return _skipped_row(
                manifest_row,
                model_name=model_name,
                model_path=model_path,
                executable=executable,
                reason="whisper.cpp completed but did not write a TXT hypothesis.",
            )
        hypothesis = hypothesis_path.read_text(encoding="utf-8").strip()
    score = evaluate_text(reference_text, hypothesis, manifest_row.get("language"))
    cleaned = (
        evaluate_cleaned_wer(reference_text, hypothesis)
        if score["metric_name"] == "WER"
        else {"cleaned_metric_name": "", "cleaned_error_rate": ""}
    )
    return {
        "clip_id": manifest_row.get("clip_id", ""),
        "dataset_name": manifest_row.get("dataset_name", ""),
        "language": manifest_row.get("language", ""),
        "model_name": model_name,
        "model_path": str(model_path),
        "executable": executable,
        "status": "ok",
        "duration_seconds": f"{duration:.6f}",
        "runtime_seconds": f"{runtime:.6f}",
        "rtf": f"{runtime / duration:.6f}" if duration > 0 else "",
        "metric_name": score["metric_name"],
        "error_rate": f"{float(score['error_rate']):.6f}",
        "cleaned_metric_name": cleaned.get("cleaned_metric_name", ""),
        "cleaned_error_rate": (
            f"{float(cleaned['cleaned_error_rate']):.6f}"
            if cleaned.get("cleaned_error_rate") != ""
            else ""
        ),
        "reference_text": reference_text,
        "hypothesis_text": hypothesis,
        "claim_level": "whisper_cpp_level1_local",
        "notes": "Local whisper.cpp benchmark; not a phone-device measurement.",
    }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["model_name"]),
            str(row["dataset_name"]),
            str(row["language"]),
            str(row["status"]),
        )
        grouped.setdefault(key, []).append(row)
    summaries: list[dict[str, str]] = []
    for (model_name, dataset_name, language, status), group in sorted(grouped.items()):
        ok = [row for row in group if row["status"] == "ok"]

        def mean(column: str) -> str:
            values = [
                float(row[column])
                for row in ok
                if str(row.get(column, "")).strip()
            ]
            return f"{statistics.fmean(values):.6f}" if values else ""

        summaries.append(
            {
                "model_name": model_name,
                "dataset_name": dataset_name,
                "language": language,
                "status": status,
                "num_rows": str(len(group)),
                "mean_error_rate": mean("error_rate"),
                "mean_cleaned_error_rate": mean("cleaned_error_rate"),
                "mean_rtf": mean("rtf"),
                "notes": "Means include only status=ok rows.",
            }
        )
    return summaries


def run_benchmark(
    *,
    manifest: str | Path,
    output: str | Path,
    summary_output: str | Path | None = None,
    executable: str | None = None,
    model_specs: list[str] | None = None,
    max_clips: int | None = None,
    timeout_seconds: int = 120,
) -> list[dict[str, Any]]:
    manifest_rows = _read_csv(resolve_repo_path(str(manifest)))
    if max_clips is not None:
        manifest_rows = manifest_rows[:max_clips]
    resolved_executable = find_executable(executable)
    models = parse_model_specs(model_specs)
    rows: list[dict[str, Any]] = []
    for model_name, model_path in models:
        for manifest_row in manifest_rows:
            if not resolved_executable:
                rows.append(
                    _skipped_row(
                        manifest_row,
                        model_name=model_name,
                        model_path=model_path,
                        executable="",
                        reason="whisper.cpp executable was not found.",
                    )
                )
                continue
            if not model_path.exists():
                rows.append(
                    _skipped_row(
                        manifest_row,
                        model_name=model_name,
                        model_path=model_path,
                        executable=resolved_executable,
                        reason=(
                            "model file is missing; download a whisper.cpp "
                            "ggml/gguf model before running the true benchmark."
                        ),
                    )
                )
                continue
            rows.append(
                _run_one(
                    manifest_row,
                    model_name=model_name,
                    model_path=model_path,
                    executable=resolved_executable,
                    timeout_seconds=timeout_seconds,
                )
            )

    output_path = resolve_repo_path(str(output))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    if summary_output is not None:
        summary_path = resolve_repo_path(str(summary_output))
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=SUMMARY_COLUMNS,
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(summarize(rows))
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--executable")
    parser.add_argument(
        "--model-spec",
        action="append",
        dest="model_specs",
        help="Model spec formatted as name=path. Can be repeated.",
    )
    parser.add_argument("--max-clips", type=int)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        rows = run_benchmark(
            manifest=args.manifest,
            output=args.output,
            summary_output=args.summary_output,
            executable=args.executable,
            model_specs=args.model_specs,
            max_clips=args.max_clips,
            timeout_seconds=args.timeout_seconds,
        )
    except (FileNotFoundError, ValueError, subprocess.TimeoutExpired) as exc:
        print(f"whisper.cpp benchmark failed: {exc}", file=sys.stderr)
        return 2
    ok = sum(row["status"] == "ok" for row in rows)
    skipped = sum(row["status"] == "skipped" for row in rows)
    print(f"Wrote {len(rows)} whisper.cpp rows: {args.output} (ok={ok}, skipped={skipped})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
