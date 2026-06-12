#!/usr/bin/env python3
"""Download a tiny multilingual Common Voice subset or legal FLEURS fallback."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.dataset_utils import (  # noqa: E402
    download_file,
    ensure_directory_markers,
    repo_relative,
    request_json,
    write_checksums,
    write_manifest,
    write_reference_bundle,
)


DATASET_ROWS_API = "https://datasets-server.huggingface.co/rows"
COMMON_VOICE_DATASET = "mozilla-foundation/common_voice_17_0"
FLEURS_DATASET = "google/fleurs"
LANGUAGE_CONFIGS = {
    "en": {"common_voice": "en", "fleurs": "en_us", "name": "English"},
    "fr": {"common_voice": "fr", "fleurs": "fr_fr", "name": "French"},
    "zh-CN": {
        "common_voice": "zh-CN",
        "fleurs": "cmn_hans_cn",
        "name": "Mandarin Chinese",
    },
}


def _audio_url(row: dict[str, Any]) -> str:
    audio = row.get("audio")
    if isinstance(audio, list) and audio and isinstance(audio[0], dict):
        return str(audio[0].get("src", ""))
    if isinstance(audio, dict):
        return str(audio.get("src") or audio.get("path") or "")
    return ""


def _fetch_rows(dataset: str, config: str, length: int) -> list[dict[str, Any]]:
    payload = request_json(
        DATASET_ROWS_API,
        params={
            "dataset": dataset,
            "config": config,
            "split": "validation",
            "offset": 0,
            "length": length,
        },
    )
    return [item["row"] | {"_row_idx": item["row_idx"]} for item in payload["rows"]]


def _select_source(language: str, count: int) -> tuple[str, str, list[dict[str, Any]], str]:
    config = LANGUAGE_CONFIGS[language]
    try:
        rows = _fetch_rows(COMMON_VOICE_DATASET, config["common_voice"], count)
        return (
            "Mozilla Common Voice",
            "17.0",
            rows,
            "Common Voice official Hugging Face dataset; release-specific terms apply.",
        )
    except Exception as common_voice_error:
        print(
            f"[common-voice] {language}: official partial access unavailable: "
            f"{common_voice_error}",
            file=sys.stderr,
        )
        print(
            f"[fallback] {language}: using official Google FLEURS validation rows "
            "through the Hugging Face dataset viewer API.",
            file=sys.stderr,
        )
        rows = _fetch_rows(FLEURS_DATASET, config["fleurs"], count)
        return (
            "Google FLEURS",
            "Hugging Face main revision",
            rows,
            "CC BY 4.0; used as the documented multilingual fallback.",
        )


def download_multilingual_subset(
    *,
    languages: list[str],
    max_clips_per_language: int,
    output_root: Path,
    reference_root: Path,
    manifest_out: Path,
) -> list[dict[str, Any]]:
    ensure_directory_markers(
        [output_root, reference_root, manifest_out.parent]
    )
    manifest_rows: list[dict[str, Any]] = []
    checksum_files: list[Path] = []

    for language in languages:
        if language not in LANGUAGE_CONFIGS:
            raise ValueError(
                f"Unsupported language {language!r}; choose from "
                f"{', '.join(LANGUAGE_CONFIGS)}."
            )
        dataset_name, dataset_version, rows, license_note = _select_source(
            language, max_clips_per_language
        )
        language_audio_dir = output_root / language
        language_reference_dir = reference_root / language
        language_audio_dir.mkdir(parents=True, exist_ok=True)
        language_reference_dir.mkdir(parents=True, exist_ok=True)

        for item in rows[:max_clips_per_language]:
            row_id = item.get("id", item["_row_idx"])
            prefix = "cv" if dataset_name == "Mozilla Common Voice" else "fleurs"
            clip_id = f"{prefix}_{language.lower().replace('-', '_')}_{row_id}"
            audio_path = language_audio_dir / f"{clip_id}.wav"
            source_url = _audio_url(item)
            if not source_url:
                raise RuntimeError(f"No downloadable audio URL for {clip_id}.")
            download_file(source_url, audio_path, max_bytes=20 * 1024 * 1024)

            sample_count = int(item.get("num_samples", 0) or 0)
            duration = sample_count / 16000 if sample_count else 0.0
            transcript = str(
                item.get("raw_transcription")
                or item.get("transcription")
                or item.get("sentence")
                or ""
            ).strip()
            if not transcript:
                raise RuntimeError(f"No transcript was returned for {clip_id}.")
            references = write_reference_bundle(
                language_reference_dir / clip_id,
                transcript=transcript,
                anchors=[
                    {
                        "start": 0.0,
                        "end": round(duration, 3),
                        "speaker": "SPEAKER_00",
                        "speakers": ["SPEAKER_00"],
                        "text": transcript,
                        "overlap": False,
                        "annotation_source": "generated_single_speaker_anchor",
                    }
                ],
            )
            checksum_files.extend([audio_path, *references.values()])
            manifest_rows.append(
                {
                    "clip_id": clip_id,
                    "audio_path": repo_relative(audio_path),
                    "source_type": "public_dataset",
                    "dataset_name": dataset_name,
                    "dataset_version": dataset_version,
                    "split": "validation",
                    "language": language,
                    "duration_seconds": f"{duration:.3f}",
                    "speaker_count": 1,
                    "has_overlap": False,
                    "has_interruptions": False,
                    "has_domain_terms": False,
                    "recording_device": "dataset_provided",
                    "noise_condition": "dataset_provided",
                    "consent_status": "dataset_terms",
                    "redistribution_status": "raw_audio_not_committed",
                    "license_or_access": license_note,
                    **{
                        key: repo_relative(value)
                        for key, value in references.items()
                    },
                    "download_status": "downloaded",
                    "notes": (
                        "Official Common Voice partial access."
                        if dataset_name == "Mozilla Common Voice"
                        else "Common Voice was unavailable; real FLEURS fallback sample."
                    ),
                }
            )

    write_manifest(manifest_out, manifest_rows)
    write_checksums(
        manifest_out.with_name(f"{manifest_out.stem}_checksums.csv"),
        checksum_files,
    )
    return manifest_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--languages", nargs="+", default=["en", "fr", "zh-CN"])
    parser.add_argument("--max-clips-per-language", type=int, default=10)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/raw/public/common_voice"),
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=Path("data/reference/public/common_voice"),
    )
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=Path("data/manifests/common_voice_multilingual_real.csv"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.max_clips_per_language < 1:
        raise SystemExit("--max-clips-per-language must be positive.")
    try:
        rows = download_multilingual_subset(
            languages=args.languages,
            max_clips_per_language=args.max_clips_per_language,
            output_root=args.output_root,
            reference_root=args.reference_root,
            manifest_out=args.manifest_out,
        )
    except Exception as exc:
        print(f"Multilingual subset acquisition failed: {exc}", file=sys.stderr)
        return 2
    print(f"Prepared {len(rows)} real multilingual clips at {args.manifest_out}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
