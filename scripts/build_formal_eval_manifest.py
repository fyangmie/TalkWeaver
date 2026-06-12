#!/usr/bin/env python3
"""Combine only locally complete real-data manifest rows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.dataset_utils import (  # noqa: E402
    PREPARED_STATUSES,
    REPO_ROOT,
    read_manifest,
    resolve_repo_path,
    write_manifest,
)


def build_combined_manifest(
    inputs: Iterable[Path],
    output: Path,
    *,
    repo_root: Path = REPO_ROOT,
) -> tuple[list[dict[str, str]], list[str]]:
    included: list[dict[str, str]] = []
    skipped: list[str] = []
    seen: set[str] = set()
    for input_path in inputs:
        if not input_path.exists():
            skipped.append(f"{input_path}: manifest missing")
            continue
        for row in read_manifest(input_path):
            clip_id = row["clip_id"].strip()
            status = row["download_status"].strip().lower()
            if not clip_id:
                skipped.append(f"{input_path}: blank clip_id")
                continue
            if status not in PREPARED_STATUSES:
                skipped.append(f"{clip_id}: status={status or 'blank'}")
                continue
            missing = [
                field
                for field in ("audio_path", "transcript_path")
                if not row[field].strip()
                or not resolve_repo_path(row[field], repo_root=repo_root).is_file()
            ]
            if missing:
                skipped.append(f"{clip_id}: missing {', '.join(missing)}")
                continue
            if clip_id in seen:
                raise ValueError(f"Duplicate clip_id across manifests: {clip_id}")
            seen.add(clip_id)
            included.append(row)
    write_manifest(output, included)
    return included, skipped


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        rows, skipped = build_combined_manifest(args.inputs, args.output)
    except Exception as exc:
        print(f"Combined manifest failed: {exc}", file=sys.stderr)
        return 2
    for message in skipped:
        print(f"Skipped: {message}", file=sys.stderr)
    print(f"Wrote {len(rows)} complete real rows to {args.output}.")
    return 0 if rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
