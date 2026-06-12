"""Shared helpers for small, reproducible public-dataset subsets."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import requests


REPO_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_COLUMNS = [
    "clip_id",
    "audio_path",
    "source_type",
    "dataset_name",
    "dataset_version",
    "split",
    "language",
    "duration_seconds",
    "speaker_count",
    "has_overlap",
    "has_interruptions",
    "has_domain_terms",
    "recording_device",
    "noise_condition",
    "consent_status",
    "redistribution_status",
    "license_or_access",
    "transcript_path",
    "anchors_path",
    "terms_path",
    "events_path",
    "download_status",
    "notes",
]

PREPARED_STATUSES = {"downloaded", "prepared"}


def repo_relative(path: Path) -> str:
    """Return a stable POSIX path relative to the repository root."""

    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def resolve_repo_path(value: str, *, repo_root: Path = REPO_ROOT) -> Path:
    """Resolve a manifest path without allowing empty values."""

    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def ensure_directory_markers(paths: Iterable[Path]) -> None:
    """Create data directories and their tracked empty-directory markers."""

    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
        marker = path / ".gitkeep"
        if not any(child for child in path.iterdir() if child.name != ".gitkeep"):
            marker.touch(exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_manifest(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write rows using the Phase 2A-REAL manifest contract."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=MANIFEST_COLUMNS,
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    column: _csv_value(row.get(column, ""))
                    for column in MANIFEST_COLUMNS
                }
            )


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [column for column in MANIFEST_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(
                f"{path} is missing required manifest columns: {', '.join(missing)}"
            )
        return list(reader)


def _csv_value(value: Any) -> Any:
    if isinstance(value, bool):
        return str(value).lower()
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_checksums(path: Path, files: Iterable[Path]) -> None:
    rows = []
    for file_path in sorted(set(files)):
        if file_path.exists() and file_path.is_file():
            rows.append(
                {
                    "path": repo_relative(file_path),
                    "sha256": sha256_file(file_path),
                    "bytes": file_path.stat().st_size,
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["path", "sha256", "bytes"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def request_json(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    timeout: int = 60,
) -> dict[str, Any]:
    response = requests.get(url, params=params, timeout=timeout)
    if response.status_code != 200:
        detail = response.text[:500].replace("\n", " ")
        raise RuntimeError(
            f"GET {response.url} returned HTTP {response.status_code}: {detail}"
        )
    return response.json()


def download_file(
    url: str,
    destination: Path,
    *,
    max_bytes: int = 100 * 1024 * 1024,
    timeout: int = 120,
) -> Path:
    """Stream one file while enforcing a strict size ceiling."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    try:
        with requests.get(url, stream=True, timeout=timeout) as response:
            response.raise_for_status()
            declared = int(response.headers.get("content-length", "0") or 0)
            if declared and declared > max_bytes:
                raise RuntimeError(
                    f"Refusing {url}: declared size {declared} exceeds "
                    f"{max_bytes} bytes."
                )
            written = 0
            with partial.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    written += len(chunk)
                    if written > max_bytes:
                        raise RuntimeError(
                            f"Refusing {url}: streamed size exceeds {max_bytes} bytes."
                        )
                    handle.write(chunk)
        partial.replace(destination)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return destination


def write_reference_bundle(
    reference_dir: Path,
    *,
    transcript: str,
    anchors: list[dict[str, Any]],
    terms: list[dict[str, Any]] | list[str] | None = None,
    events: list[dict[str, Any]] | None = None,
) -> dict[str, Path]:
    reference_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = reference_dir / "reference_transcript.txt"
    anchors_path = reference_dir / "reference_anchors.json"
    terms_path = reference_dir / "reference_terms.json"
    events_path = reference_dir / "reference_events.json"
    transcript_path.write_text(transcript.strip() + "\n", encoding="utf-8")
    write_json(anchors_path, anchors)
    write_json(terms_path, terms or [])
    write_json(events_path, events or [])
    return {
        "transcript_path": transcript_path,
        "anchors_path": anchors_path,
        "terms_path": terms_path,
        "events_path": events_path,
    }
