#!/usr/bin/env python3
"""Download and prepare a small multi-file Earnings-22 evaluation subset.

The script downloads public Earnings-22 audio into ignored local storage,
downloads small force-aligned transcript files, then creates one reliable
short diagnostic slice per source file. Raw audio and generated WAV clips are
not intended to be committed.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_earnings22_eval_slices import (  # noqa: E402
    MANIFEST_COLUMNS,
    prepare_slices,
)


GITHUB_RAW_BASE = (
    "https://raw.githubusercontent.com/revdotcom/speech-datasets/main"
)
GITHUB_MEDIA_BASE = (
    "https://media.githubusercontent.com/media/revdotcom/speech-datasets/main"
)
METADATA_URL = f"{GITHUB_RAW_BASE}/earnings22/metadata.csv"
GITHUB_CONTENTS_API_BASE = (
    "https://api.github.com/repos/revdotcom/speech-datasets/contents"
)
DEFAULT_FILE_IDS = (
    "4453225",  # South Africa, English
    "4467434",  # South Africa, English
    "4481221",  # India, Asian region
    "4462231",  # United States, English
)
DEFAULT_HELDOUT_FILE_IDS = (
    "4474955",
    "4483046",
    "4468919",
    "4475604",
    "4471586",
    "4482968",
    "4482110",
    "4469075",
    "4446796",
    "4483623",
    "4485206",
    "4480850",
)
DEFAULT_HELDOUT_FALLBACK_FILE_IDS = (
    "4470290",
    "4469528",
    "4470010",
    "4470570",
    "4329526",
    "4450779",
)
DEFAULT_DEV_EXCLUDE_FILE_IDS = (
    "4453225",
    "4467434",
    "4481221",
    "4462231",
)


def project_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def media_url(file_id: str) -> str:
    return f"{GITHUB_MEDIA_BASE}/earnings22/media/{file_id}.mp3"


def aligned_reference_url(file_id: str) -> str:
    return (
        f"{GITHUB_RAW_BASE}/earnings22/transcripts/"
        f"force_aligned_nlp_references/{file_id}.aligned.nlp"
    )


def nlp_reference_url(file_id: str) -> str:
    return (
        f"{GITHUB_RAW_BASE}/earnings22/transcripts/"
        f"nlp_references/{file_id}.nlp"
    )


def _looks_like_lfs_pointer(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size > 512:
        return False
    return path.read_bytes().startswith(b"version https://git-lfs.github.com/spec")


def _github_contents_api_url(raw_url: str) -> str | None:
    prefix = f"{GITHUB_RAW_BASE}/"
    if not raw_url.startswith(prefix):
        return None
    relative = raw_url.removeprefix(prefix)
    quoted = urllib.parse.quote(relative)
    return f"{GITHUB_CONTENTS_API_BASE}/{quoted}?ref=main"


def _download_github_contents(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    content = str(payload.get("content", ""))
    encoding = str(payload.get("encoding", ""))
    if encoding != "base64" or not content:
        raise RuntimeError(f"Unexpected GitHub contents response for {url}")
    return base64.b64decode(content)


def download_url(
    url: str,
    destination: Path,
    *,
    refresh: bool = False,
    retries: int = 3,
) -> bool:
    """Download one URL if missing; return whether a new file was written."""

    if destination.is_file() and not refresh and not _looks_like_lfs_pointer(destination):
        print(f"Using existing {repo_relative(destination)}", flush=True)
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            contents_url = _github_contents_api_url(url)
            effective_url = contents_url or url
            print(f"Downloading {effective_url}", flush=True)
            if contents_url:
                payload = _download_github_contents(contents_url)
            else:
                with urllib.request.urlopen(url, timeout=60) as response:
                    payload = response.read()
            destination.write_bytes(payload)
            print(
                f"Wrote {repo_relative(destination)} "
                f"({destination.stat().st_size} bytes)",
                flush=True,
            )
            return True
        except (RuntimeError, urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(1.5 * attempt)
    raise RuntimeError(f"Could not download {url}: {last_error}")


def load_metadata(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {row["File ID"]: row for row in rows if row.get("File ID")}


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _unique_file_ids(values: list[str] | tuple[str, ...]) -> list[str]:
    return list(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))


def prepare_subset(
    *,
    file_ids: list[str],
    fallback_file_ids: list[str] | None = None,
    exclude_file_ids: list[str] | None = None,
    target_count: int | None = None,
    output_manifest: str | Path,
    raw_dir: str | Path,
    reference_root: str | Path,
    slice_seconds: float,
    start_seconds: float,
    split_name: str = "multi-file diagnostic subset",
    refresh: bool = False,
    min_reference_words: int = 80,
    max_reference_words_per_second: float = 3.5,
    min_aligned_anchor_ratio: float = 0.05,
    download_full_reference: bool = False,
) -> list[dict[str, str]]:
    raw_root = project_path(raw_dir)
    ref_root = project_path(reference_root)
    manifest_path = project_path(output_manifest)
    metadata_path = ref_root / "metadata.csv"
    download_url(METADATA_URL, metadata_path, refresh=refresh)
    metadata = load_metadata(metadata_path)
    excluded = set(_unique_file_ids(exclude_file_ids or []))
    requested = _unique_file_ids(
        [
            *file_ids,
            *(fallback_file_ids or []),
        ]
    )
    selected = [file_id for file_id in requested if file_id not in excluded]

    rows: list[dict[str, str]] = []
    with tempfile.TemporaryDirectory(prefix="earnings22_manifest_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        for file_id in selected:
            if target_count is not None and len(rows) >= target_count:
                break
            print(f"Preparing Earnings-22 file_id={file_id}", flush=True)
            item = metadata.get(file_id, {})
            audio_path = raw_root / f"{file_id}.mp3"
            reference_dir = ref_root / file_id
            aligned_path = reference_dir / "reference_aligned.nlp"
            full_path = reference_dir / "reference_full.nlp"
            download_url(media_url(file_id), audio_path, refresh=refresh)
            download_url(
                aligned_reference_url(file_id),
                aligned_path,
                refresh=refresh,
            )
            if download_full_reference:
                download_url(nlp_reference_url(file_id), full_path, refresh=refresh)

            tmp_manifest = tmp_root / f"{file_id}.csv"
            prepared = prepare_slices(
                source_audio=audio_path,
                reference_aligned=aligned_path,
                output_manifest=tmp_manifest,
                output_audio_dir=raw_root,
                output_reference_root=ref_root,
                clip_id_prefix=f"earnings22_{file_id}",
                slice_seconds=slice_seconds,
                num_slices=1,
                start_offsets=[start_seconds],
                min_reference_words=min_reference_words,
                max_reference_words_per_second=max_reference_words_per_second,
                min_aligned_anchor_ratio=min_aligned_anchor_ratio,
            )
            print(
                f"Prepared {len(prepared)} reliable slice(s) for file_id={file_id}",
                flush=True,
            )
            country = item.get("Country by Ticker", "unknown")
            region = item.get("Language Family + Area Based", "unknown")
            ticker = item.get("Ticker Symbol", "unknown")
            for row in prepared:
                row["dataset_version"] = (
                    "Rev.com speech-datasets main; "
                    f"file_id={file_id}; ticker={ticker}; "
                    f"country={country}; accent_region={region}"
                )
                row["split"] = split_name
                row["notes"] = (
                    row["notes"]
                    + f" Source file metadata: ticker={ticker}, "
                    f"country={country}, accent_region={region}."
                )
            rows.extend(prepared)

    if target_count is not None and len(rows) < target_count:
        raise RuntimeError(
            f"Only prepared {len(rows)} reliable row(s); target_count={target_count}."
        )
    write_manifest(manifest_path, rows)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--file-ids",
        nargs="+",
        default=list(DEFAULT_FILE_IDS),
        help="Earnings-22 file IDs to download and prepare.",
    )
    parser.add_argument(
        "--heldout-defaults",
        action="store_true",
        help=(
            "Use the frozen 12-file held-out candidate list and fallback list "
            "instead of the four-file diagnostic default."
        ),
    )
    parser.add_argument(
        "--fallback-file-ids",
        nargs="*",
        default=[],
        help="Extra file IDs to try if target-count has not been reached.",
    )
    parser.add_argument(
        "--exclude-file-ids",
        nargs="*",
        default=[],
        help="File IDs to exclude from this manifest, such as dev-set files.",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        help="Stop after this many reliable slices and fail if fewer are prepared.",
    )
    parser.add_argument(
        "--split-name",
        default="multi-file diagnostic subset",
        help="Manifest split label to write for prepared rows.",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=Path("data/manifests/earnings22_multi_file_4x180.csv"),
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw/public/earnings22"),
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=Path("data/reference/public/earnings22"),
    )
    parser.add_argument("--slice-seconds", type=float, default=180.0)
    parser.add_argument("--start-seconds", type=float, default=0.0)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument(
        "--download-full-reference",
        action="store_true",
        help=(
            "Also download non-aligned NLP references. The default only "
            "downloads force-aligned references needed for slice preparation."
        ),
    )
    parser.add_argument("--min-reference-words", type=int, default=80)
    parser.add_argument(
        "--max-reference-words-per-second",
        type=float,
        default=3.5,
    )
    parser.add_argument("--min-aligned-anchor-ratio", type=float, default=0.05)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    file_ids = args.file_ids
    fallback_file_ids = args.fallback_file_ids
    exclude_file_ids = args.exclude_file_ids
    if args.heldout_defaults:
        file_ids = list(DEFAULT_HELDOUT_FILE_IDS)
        fallback_file_ids = [
            *DEFAULT_HELDOUT_FALLBACK_FILE_IDS,
            *fallback_file_ids,
        ]
        exclude_file_ids = [
            *DEFAULT_DEV_EXCLUDE_FILE_IDS,
            *exclude_file_ids,
        ]
    try:
        rows = prepare_subset(
            file_ids=file_ids,
            fallback_file_ids=fallback_file_ids,
            exclude_file_ids=exclude_file_ids,
            target_count=args.target_count,
            output_manifest=args.output_manifest,
            raw_dir=args.raw_dir,
            reference_root=args.reference_root,
            slice_seconds=args.slice_seconds,
            start_seconds=args.start_seconds,
            split_name=args.split_name,
            refresh=args.refresh,
            min_reference_words=args.min_reference_words,
            max_reference_words_per_second=args.max_reference_words_per_second,
            min_aligned_anchor_ratio=args.min_aligned_anchor_ratio,
            download_full_reference=args.download_full_reference,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Earnings-22 subset download failed: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote {len(rows)} prepared Earnings-22 rows to {args.output_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
