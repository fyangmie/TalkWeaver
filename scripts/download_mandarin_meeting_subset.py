#!/usr/bin/env python3
"""Attempt a size-capped official Mandarin meeting subset acquisition."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.dataset_utils import ensure_directory_markers, write_manifest  # noqa: E402


MAX_AUTOMATIC_BYTES = 500 * 1024 * 1024
AISHELL4_ARCHIVES = {
    "test.tar.gz": "https://openslr.trmal.net/resources/111/test.tar.gz",
    "train_L.tar.gz": "https://openslr.trmal.net/resources/111/train_L.tar.gz",
}
ALIMEETING_PAGE = "https://www.modelscope.cn/datasets/modelscope/AliMeeting"


def _head_size(url: str) -> tuple[int, str]:
    response = requests.head(url, allow_redirects=True, timeout=45)
    response.raise_for_status()
    return int(response.headers.get("content-length", "0") or 0), response.url


def assess_mandarin_sources(dataset: str) -> list[str]:
    reasons: list[str] = []
    if dataset in {"auto", "aishell4", "aishell4_or_alimeeting_or_auto"}:
        for name, url in AISHELL4_ARCHIVES.items():
            try:
                size, final_url = _head_size(url)
            except Exception as exc:
                reasons.append(f"AISHELL-4 {name}: HEAD {url} failed: {exc}")
                continue
            if size > MAX_AUTOMATIC_BYTES:
                reasons.append(
                    f"AISHELL-4 {name}: {size} bytes at {final_url}, exceeding "
                    f"the {MAX_AUTOMATIC_BYTES}-byte automatic-download ceiling."
                )
            else:
                reasons.append(
                    f"AISHELL-4 {name}: archive is small enough, but no "
                    "file-level extraction path was verified."
                )

    if dataset in {"auto", "alimeeting", "aishell4_or_alimeeting_or_auto"}:
        try:
            response = requests.get(ALIMEETING_PAGE, timeout=45)
            response.raise_for_status()
        except Exception as exc:
            reasons.append(f"AliMeeting page GET {ALIMEETING_PAGE} failed: {exc}")
        else:
            reasons.append(
                "AliMeeting official ModelScope page is reachable, but no "
                "verified file-level API for 1-3 audio clips with matching "
                "annotations was found; manual dataset access/preparation is required."
            )
    return reasons


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="auto",
        choices=[
            "auto",
            "aishell4",
            "alimeeting",
            "aishell4_or_alimeeting_or_auto",
        ],
    )
    parser.add_argument("--max-clips", type=int, default=3)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/raw/public/mandarin_meeting"),
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=Path("data/reference/public/mandarin_meeting"),
    )
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=Path("data/manifests/mandarin_meeting_real.csv"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    ensure_directory_markers(
        [args.output_root, args.reference_root, args.manifest_out.parent]
    )
    write_manifest(args.manifest_out, [])
    reasons = assess_mandarin_sources(args.dataset)
    print(
        "No Mandarin meeting rows were prepared. The empty manifest contains "
        "no placeholders and is safe for the combined-manifest builder.",
        file=sys.stderr,
    )
    for reason in reasons:
        print(f"- {reason}", file=sys.stderr)
    print(
        "See docs/manual_dataset_steps.md for approved manual acquisition steps.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
