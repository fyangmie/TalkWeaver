#!/usr/bin/env python3
"""Create a Level-1 mobile ASR trade-off table from local ASR runs.

The current implementation intentionally supports a conservative proxy mode:
it reuses the same faster-whisper CPU int8 benchmark rows that feed the formal
ASR evaluation and marks every row as ``mobile_style_proxy``. This gives the
paper a reproducible speed/accuracy trade-off table without pretending that a
desktop CPU run is a true phone or whisper.cpp measurement.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

OUTPUT_COLUMNS = [
    "clip_id",
    "dataset_name",
    "language",
    "model_name",
    "backend",
    "model_family",
    "quantization",
    "device",
    "runtime_environment",
    "compute_type",
    "vad_filter",
    "duration_seconds",
    "runtime_seconds",
    "rtf",
    "metric_name",
    "error_rate",
    "cleaned_metric_name",
    "cleaned_error_rate",
    "cold_model_load_seconds",
    "model_size_mb",
    "model_size_source",
    "hardware_label",
    "true_mobile_device",
    "whisper_cpp_executable",
    "claim_level",
    "source_result_path",
    "notes",
]


def project_path(path: str | Path) -> Path:
    """Resolve a repository-relative path."""

    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def display_path(path: Path) -> str:
    """Return a stable repository-relative display path when possible."""

    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def find_whisper_cpp_executable() -> str:
    """Locate a local whisper.cpp command if one is already installed."""

    candidates = (
        "whisper-cli",
        "whisper-cpp",
        "whisper.cpp",
        "main",
    )
    for candidate in candidates:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    return ""


def default_hardware_label() -> str:
    """Build a concise local hardware/runtime label."""

    processor = platform.processor() or platform.machine() or "unknown-cpu"
    system = platform.system() or "unknown-os"
    return f"{system} {processor}".strip()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_mobile_proxy_rows(
    source_rows: list[dict[str, str]],
    *,
    source_result_path: Path,
    hardware_label: str,
    whisper_cpp_executable: str,
) -> list[dict[str, str]]:
    """Convert real ASR benchmark rows into explicit mobile-style proxy rows."""

    output_rows: list[dict[str, str]] = []
    for row in source_rows:
        compute_type = row.get("compute_type", "").strip()
        quantization = compute_type or "unknown"
        output_rows.append(
            {
                "clip_id": row.get("clip_id", ""),
                "dataset_name": row.get("dataset_name", ""),
                "language": row.get("language", ""),
                "model_name": row.get("model_name", ""),
                "backend": "faster-whisper",
                "model_family": "Whisper",
                "quantization": quantization,
                "device": row.get("device", ""),
                "runtime_environment": "local_cpu_proxy",
                "compute_type": compute_type,
                "vad_filter": row.get("vad_filter", ""),
                "duration_seconds": row.get("duration_seconds", ""),
                "runtime_seconds": row.get("runtime_seconds", ""),
                "rtf": row.get("rtf", ""),
                "metric_name": row.get("metric_name", ""),
                "error_rate": row.get("error_rate", ""),
                "cleaned_metric_name": row.get("cleaned_metric_name", ""),
                "cleaned_error_rate": row.get("cleaned_error_rate", ""),
                "cold_model_load_seconds": row.get(
                    "cold_model_load_seconds",
                    "",
                ),
                "model_size_mb": "",
                "model_size_source": "not_measured_for_proxy",
                "hardware_label": hardware_label,
                "true_mobile_device": "false",
                "whisper_cpp_executable": whisper_cpp_executable,
                "claim_level": "mobile_style_proxy",
                "source_result_path": display_path(source_result_path),
                "notes": (
                    "Derived from faster-whisper CPU int8 warm runtime. "
                    "This is a reproducible trade-off proxy, not a true "
                    "phone or whisper.cpp benchmark."
                ),
            }
        )
    return output_rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    """Write mobile trade-off rows to CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=OUTPUT_COLUMNS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_metadata(
    path: Path,
    *,
    mode_requested: str,
    source_result_path: Path,
    output_csv_path: Path,
    row_count: int,
    hardware_label: str,
    whisper_cpp_executable: str,
) -> None:
    """Write runtime metadata that defines the claim level."""

    payload: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "mode_requested": mode_requested,
        "mode_effective": "proxy",
        "claim_level": "mobile_style_proxy",
        "row_count": row_count,
        "source_result_path": display_path(source_result_path),
        "output_csv_path": display_path(output_csv_path),
        "hardware_label": hardware_label,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "whisper_cpp_available": bool(whisper_cpp_executable),
        "whisper_cpp_executable": whisper_cpp_executable,
        "true_mobile_device": False,
        "notes": [
            (
                "The current Level-1 artifact is a proxy derived from local "
                "faster-whisper CPU int8 benchmark rows."
            ),
            (
                "Do not report these rows as true mobile, on-device, or "
                "whisper.cpp performance."
            ),
            (
                "Replace or extend this CSV with device-measured whisper.cpp "
                "rows after approving and running a real mobile benchmark."
            ),
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def build_mobile_proxy_report(
    source_asr_results: str | Path,
    output_csv: str | Path,
    metadata_output: str | Path,
    *,
    mode: str = "auto",
    hardware_label: str | None = None,
) -> list[dict[str, str]]:
    """Build the mobile-style proxy table and metadata sidecar."""

    if mode not in {"auto", "proxy"}:
        raise ValueError("mode must be 'auto' or 'proxy'.")

    source_path = project_path(source_asr_results)
    output_path = project_path(output_csv)
    metadata_path = project_path(metadata_output)
    source_rows = _read_csv(source_path)
    if not source_rows:
        raise ValueError("source ASR result CSV contains no rows.")

    executable = find_whisper_cpp_executable()
    resolved_hardware_label = hardware_label or default_hardware_label()
    rows = build_mobile_proxy_rows(
        source_rows,
        source_result_path=source_path,
        hardware_label=resolved_hardware_label,
        whisper_cpp_executable=executable,
    )
    write_csv(output_path, rows)
    write_metadata(
        metadata_path,
        mode_requested=mode,
        source_result_path=source_path,
        output_csv_path=output_path,
        row_count=len(rows),
        hardware_label=resolved_hardware_label,
        whisper_cpp_executable=executable,
    )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-asr-results",
        type=Path,
        default=Path("experiments/results/asr_benchmark_real.csv"),
        help="Existing faster-whisper ASR benchmark CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/results/v1/mobile_asr.csv"),
        help="Output mobile-style trade-off CSV.",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=Path("experiments/results/v1/mobile_device_metadata.json"),
        help="Output metadata JSON documenting claim level and hardware.",
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "proxy"),
        default="auto",
        help="Current implementation always emits explicit proxy rows.",
    )
    parser.add_argument(
        "--hardware-label",
        default=None,
        help="Optional human-readable hardware label for this local run.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        rows = build_mobile_proxy_report(
            args.source_asr_results,
            args.output,
            args.metadata_output,
            mode=args.mode,
            hardware_label=args.hardware_label,
        )
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"Mobile ASR trade-off report failed: {exc}", file=sys.stderr)
        return 2

    print(
        "Wrote "
        f"{len(rows)} mobile-style proxy rows to {args.output}; "
        f"metadata: {args.metadata_output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
