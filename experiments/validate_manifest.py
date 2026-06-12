#!/usr/bin/env python3
"""Validate TalkWeaver Phase 2A-REAL manifest files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.dataset_utils import (  # noqa: E402
    PREPARED_STATUSES,
    REPO_ROOT,
    read_manifest,
    resolve_repo_path,
)


JSON_PATH_FIELDS = ("anchors_path", "terms_path", "events_path")


def validate_manifest(
    manifest: Path,
    *,
    require_real_files: bool = False,
    repo_root: Path = REPO_ROOT,
) -> list[str]:
    errors: list[str] = []
    try:
        rows = read_manifest(manifest)
    except Exception as exc:
        return [str(exc)]
    if not rows:
        errors.append(f"{manifest} contains no data rows.")
        return errors

    seen: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        clip_id = row["clip_id"].strip()
        label = clip_id or f"row {row_number}"
        if not clip_id:
            errors.append(f"row {row_number}: clip_id is blank.")
        elif clip_id in seen:
            errors.append(f"{label}: duplicate clip_id.")
        seen.add(clip_id)

        status = row["download_status"].strip().lower()
        if require_real_files and status not in PREPARED_STATUSES:
            errors.append(
                f"{label}: download_status must be downloaded or prepared, got "
                f"{status or 'blank'}."
            )
        for field in ("audio_path", "transcript_path"):
            value = row[field].strip()
            if require_real_files and not value:
                errors.append(f"{label}: {field} is required.")
            elif value and not resolve_repo_path(value, repo_root=repo_root).is_file():
                errors.append(f"{label}: {field} does not exist: {value}")
        for field in JSON_PATH_FIELDS:
            value = row[field].strip()
            if not value:
                continue
            path = resolve_repo_path(value, repo_root=repo_root)
            if not path.is_file():
                errors.append(f"{label}: {field} does not exist: {value}")
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                errors.append(f"{label}: invalid JSON in {field}: {exc}")
                continue
            if not isinstance(payload, list):
                errors.append(f"{label}: {field} must contain a JSON list.")
        if require_real_files:
            planned = "planned" in status or row["source_type"].strip().lower() == "planned"
            if planned:
                errors.append(f"{label}: planned rows are not allowed.")
    return errors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--require-real-files", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    errors = validate_manifest(
        args.manifest,
        require_real_files=args.require_real_files,
    )
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 2
    row_count = len(read_manifest(args.manifest))
    print(f"Manifest valid: {args.manifest} ({row_count} real rows).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
