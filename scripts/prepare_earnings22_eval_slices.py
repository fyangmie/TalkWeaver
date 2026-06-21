#!/usr/bin/env python3
"""Prepare local Earnings-22 diagnostic slices from an existing audio file.

This script does not download data. It expects the public Earnings-22 audio
and aligned reference transcript to already exist locally, then creates short
WAV clips plus reference text files and a manifest for repeatable ASR audits.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.preprocessing import TARGET_SAMPLE_RATE, load_audio  # noqa: E402


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


EARNINGS22_GLOSSARY: list[dict[str, Any]] = [
    {
        "canonical": "Aspen",
        "aliases": ["Aspen's"],
        "spoken_forms": ["Aspen"],
        "asr_error_forms": ["aspen", "aspens"],
        "language": "en",
        "category": "company_name",
        "allowed_contexts": ["results presentation", "revenue", "earnings"],
    },
    {
        "canonical": "Regional Brands",
        "aliases": ["Regional brands"],
        "spoken_forms": ["Regional Brands"],
        "asr_error_forms": ["regional brand"],
        "language": "en",
        "category": "business_segment",
        "allowed_contexts": ["segmental revenue", "reported terms"],
    },
    {
        "canonical": "Sterile Focus Brands",
        "aliases": ["Sterile focus brands"],
        "spoken_forms": ["Sterile Focus Brands"],
        "asr_error_forms": ["several focus brands"],
        "language": "en",
        "category": "business_segment",
        "allowed_contexts": ["constant exchange rate", "growth"],
    },
    {
        "canonical": "continuing operations",
        "spoken_forms": ["continuing operations"],
        "asr_error_forms": ["continuing operation", "continued operations"],
        "language": "en",
        "category": "financial_term",
        "allowed_contexts": ["financial highlights", "revenue", "earnings"],
    },
    {
        "canonical": "organic growth",
        "spoken_forms": ["organic growth"],
        "asr_error_forms": ["organ growth"],
        "language": "en",
        "category": "financial_term",
        "allowed_contexts": ["revenue", "double-digit", "portfolio"],
    },
    {
        "canonical": "double-digit",
        "aliases": ["double digit"],
        "spoken_forms": ["double digit"],
        "asr_error_forms": ["double dig it"],
        "language": "en",
        "category": "financial_term",
        "allowed_contexts": ["growth", "revenue"],
    },
    {
        "canonical": "debt",
        "spoken_forms": ["debt"],
        "asr_error_forms": ["dead"],
        "language": "en",
        "category": "financial_term",
        "allowed_contexts": ["leverage", "cash", "financial"],
    },
    {
        "canonical": "leverage",
        "spoken_forms": ["leverage"],
        "asr_error_forms": ["lever edge"],
        "language": "en",
        "category": "financial_term",
        "allowed_contexts": ["debt", "levels"],
    },
    {
        "canonical": "dividend",
        "spoken_forms": ["dividend"],
        "asr_error_forms": ["divide end"],
        "language": "en",
        "category": "financial_term",
        "allowed_contexts": ["cents a share", "reinstatement"],
    },
    {
        "canonical": "cents a share",
        "aliases": ["cents per share"],
        "spoken_forms": ["cents a share"],
        "asr_error_forms": ["sense a share", "seems to share"],
        "language": "en",
        "category": "financial_term",
        "allowed_contexts": ["dividend", "share"],
    },
    {
        "canonical": "constant exchange rate",
        "aliases": ["constant exchange rates"],
        "spoken_forms": ["constant exchange rate"],
        "asr_error_forms": [],
        "language": "en",
        "category": "financial_term",
        "allowed_contexts": ["reported terms", "year-on-year"],
    },
    {
        "canonical": "oncology",
        "spoken_forms": ["oncology"],
        "asr_error_forms": ["on college"],
        "language": "en",
        "category": "domain_term",
        "allowed_contexts": ["prices", "products", "market"],
    },
    {
        "canonical": "COVID",
        "aliases": ["Covid"],
        "spoken_forms": ["COVID"],
        "asr_error_forms": ["covered"],
        "language": "en",
        "category": "domain_term",
        "allowed_contexts": ["circumstances", "market", "products"],
    },
]


@dataclass(frozen=True)
class ReferenceToken:
    token: str
    speaker: str
    time: float
    end_time: float
    punctuation: str = ""
    prepunctuation: str = ""
    is_aligned: bool = False


def project_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def _parse_float(value: str | None) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _read_raw_reference_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="|")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Aligned reference contains no rows: {path}")
    return rows


def _interpolated_times(rows: list[dict[str, str]]) -> list[tuple[float, bool]]:
    known: list[tuple[int, float]] = []
    for index, row in enumerate(rows):
        timestamp = _parse_float(row.get("ts"))
        if timestamp is not None:
            known.append((index, timestamp))
    if not known:
        raise ValueError("Aligned reference contains no word timestamps.")

    times = [math.nan] * len(rows)
    for index, timestamp in known:
        times[index] = timestamp

    first_index, first_time = known[0]
    for index in range(first_index):
        times[index] = max(0.0, first_time * (index + 1) / (first_index + 1))

    for (left_index, left_time), (right_index, right_time) in zip(
        known,
        known[1:],
    ):
        gap = right_index - left_index
        if gap <= 1:
            continue
        for index in range(left_index + 1, right_index):
            fraction = (index - left_index) / gap
            times[index] = left_time + (right_time - left_time) * fraction

    last_index, last_time = known[-1]
    for index in range(last_index + 1, len(rows)):
        times[index] = last_time + 0.35 * (index - last_index)

    known_indices = {index for index, _timestamp in known}
    return [
        (float(value), index in known_indices)
        for index, value in enumerate(times)
        if not math.isnan(value)
    ]


def load_reference_tokens(path: str | Path) -> list[ReferenceToken]:
    """Load aligned Earnings-22 rows with interpolated per-token times."""

    rows = _read_raw_reference_rows(project_path(path))
    times = _interpolated_times(rows)
    tokens: list[ReferenceToken] = []
    for row, (timestamp, is_aligned) in zip(rows, times):
        token = str(row.get("token", "")).strip()
        if not token:
            continue
        end_time = _parse_float(row.get("endTs"))
        if end_time is None or end_time < timestamp:
            end_time = timestamp
        tokens.append(
            ReferenceToken(
                token=token,
                speaker=str(row.get("speaker", "")).strip() or "0",
                time=timestamp,
                end_time=end_time,
                punctuation=str(row.get("punctuation", "") or ""),
                prepunctuation=str(row.get("prepunctuation", "") or ""),
                is_aligned=is_aligned,
            )
        )
    return tokens


def tokens_for_window(
    tokens: list[ReferenceToken],
    *,
    start: float,
    end: float,
) -> list[ReferenceToken]:
    return [token for token in tokens if start <= token.time < end]


def render_reference_text(tokens: list[ReferenceToken]) -> str:
    pieces: list[str] = []
    for token in tokens:
        piece = f"{token.prepunctuation}{token.token}{token.punctuation}"
        pieces.append(piece)
    return " ".join(" ".join(pieces).split())


def _contains_term(text: str, entry: dict[str, Any]) -> bool:
    normalized = f" {text.casefold()} "
    forms = [
        str(entry.get("canonical", "")),
        *[str(value) for value in entry.get("aliases", [])],
        *[str(value) for value in entry.get("spoken_forms", [])],
    ]
    return any(f" {form.casefold()} " in normalized for form in forms if form)


def glossary_for_reference(text: str) -> list[dict[str, Any]]:
    return [
        entry
        for entry in EARNINGS22_GLOSSARY
        if _contains_term(text, entry)
    ]


def _resample_linear(audio: np.ndarray, input_rate: int, target_rate: int) -> np.ndarray:
    if input_rate == target_rate:
        return audio.astype(np.float32, copy=False)
    output_length = max(1, int(round(len(audio) * target_rate / input_rate)))
    source_positions = np.arange(len(audio), dtype=np.float64)
    target_positions = np.linspace(0, max(0, len(audio) - 1), output_length)
    return np.interp(target_positions, source_positions, audio).astype(np.float32)


def _write_pcm_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.round(np.clip(audio, -1.0, 1.0) * 32_767.0).astype("<i2")
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())


def _mono_audio(path: Path, target_sample_rate: int) -> tuple[np.ndarray, int, str]:
    samples, sample_rate, _channels, loader = load_audio(path)
    mono = samples.mean(axis=1).astype(np.float32)
    mono = _resample_linear(mono, sample_rate, target_sample_rate)
    return mono, target_sample_rate, loader


def _slice_audio(
    audio: np.ndarray,
    *,
    sample_rate: int,
    start: float,
    duration: float,
) -> np.ndarray:
    start_index = int(round(start * sample_rate))
    end_index = int(round((start + duration) * sample_rate))
    return audio[start_index:end_index]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _parse_offsets(value: list[float] | None, *, count: int, duration: float) -> list[float]:
    if value:
        return [float(item) for item in value]
    return [index * duration for index in range(count)]


def prepare_slices(
    *,
    source_audio: str | Path,
    reference_aligned: str | Path,
    output_manifest: str | Path,
    output_audio_dir: str | Path,
    output_reference_root: str | Path,
    clip_id_prefix: str = "earnings22_4453225",
    slice_seconds: float = 180.0,
    num_slices: int = 5,
    start_offsets: list[float] | None = None,
    target_sample_rate: int = TARGET_SAMPLE_RATE,
    min_reference_words: int = 80,
    max_reference_words_per_second: float = 3.5,
    min_aligned_anchor_ratio: float = 0.05,
) -> list[dict[str, str]]:
    audio_path = project_path(source_audio)
    aligned_path = project_path(reference_aligned)
    manifest_path = project_path(output_manifest)
    audio_dir = project_path(output_audio_dir)
    reference_root = project_path(output_reference_root)

    tokens = load_reference_tokens(aligned_path)
    audio, sample_rate, loader = _mono_audio(audio_path, target_sample_rate)
    audio_duration = len(audio) / sample_rate
    offsets = _parse_offsets(
        start_offsets,
        count=num_slices,
        duration=slice_seconds,
    )

    rows: list[dict[str, str]] = []
    for start in offsets:
        end = min(start + slice_seconds, audio_duration)
        if end - start < 30:
            continue
        start_label = int(round(start))
        duration = end - start
        clip_id = f"{clip_id_prefix}_{start_label:04d}_{int(duration)}s"
        clip_audio_path = audio_dir / f"{clip_id}.wav"
        reference_dir = reference_root / clip_id
        transcript_path = reference_dir / "reference_transcript.txt"
        anchors_path = reference_dir / "reference_anchors.json"
        terms_path = reference_dir / "reference_terms.json"
        events_path = reference_dir / "reference_events.json"

        audio_slice = _slice_audio(
            audio,
            sample_rate=sample_rate,
            start=start,
            duration=duration,
        )
        _write_pcm_wav(clip_audio_path, audio_slice, sample_rate)

        reference_tokens = tokens_for_window(tokens, start=start, end=end)
        aligned_count = sum(token.is_aligned for token in reference_tokens)
        word_rate = len(reference_tokens) / duration if duration else 0.0
        anchor_ratio = aligned_count / len(reference_tokens) if reference_tokens else 0.0
        if (
            len(reference_tokens) < min_reference_words
            or word_rate > max_reference_words_per_second
            or anchor_ratio < min_aligned_anchor_ratio
        ):
            print(
                "Skipping "
                f"{clip_id}: reference quality guard failed "
                f"(words={len(reference_tokens)}, "
                f"word_rate={word_rate:.2f}/s, "
                f"aligned_anchor_ratio={anchor_ratio:.3f}).",
                file=sys.stderr,
            )
            continue
        reference_text = render_reference_text(reference_tokens)
        transcript_path.parent.mkdir(parents=True, exist_ok=True)
        transcript_path.write_text(reference_text + "\n", encoding="utf-8")
        _write_json(
            anchors_path,
            {
                "source_audio": repo_relative(audio_path),
                "source_reference": repo_relative(aligned_path),
                "start_seconds": round(start, 3),
                "end_seconds": round(end, 3),
                "timestamp_policy": (
                    "Missing Earnings-22 word timestamps are linearly "
                    "interpolated between neighboring aligned words."
                ),
                "tokens": [
                    {
                        "token": token.token,
                        "speaker": token.speaker,
                        "start": round(token.time - start, 3),
                        "end": round(token.end_time - start, 3),
                    }
                    for token in reference_tokens
                ],
            },
        )
        _write_json(terms_path, glossary_for_reference(reference_text))
        _write_json(
            events_path,
            {
                "overlap_events": [],
                "interruption_events": [],
                "notes": (
                    "No human overlap/interruption labels are provided for "
                    "this Earnings-22 diagnostic slice."
                ),
            },
        )
        rows.append(
            {
                "clip_id": clip_id,
                "audio_path": repo_relative(clip_audio_path),
                "source_type": "public_dataset",
                "dataset_name": "Earnings-22",
                "dataset_version": "Rev.com speech-datasets main; local file 4453225",
                "split": "local expanded diagnostic slices",
                "language": "en",
                "duration_seconds": f"{duration:.3f}",
                "speaker_count": "unknown",
                "has_overlap": "false",
                "has_interruptions": "false",
                "has_domain_terms": str(bool(glossary_for_reference(reference_text))).lower(),
                "recording_device": "earnings_call_audio",
                "noise_condition": "teleconference",
                "consent_status": "public_dataset_terms",
                "redistribution_status": "raw_audio_not_committed",
                "license_or_access": (
                    "Earnings-22 public speech-datasets; generated WAV "
                    "slices are kept local and ignored by git."
                ),
                "transcript_path": repo_relative(transcript_path),
                "anchors_path": repo_relative(anchors_path),
                "terms_path": repo_relative(terms_path),
                "events_path": repo_relative(events_path),
                "download_status": "prepared",
                "notes": (
                    f"Expanded diagnostic slice start={start:.1f}s; "
                    f"audio decoded via {loader}; reference from aligned NLP; "
                    f"words={len(reference_tokens)}, "
                    f"aligned_anchor_ratio={anchor_ratio:.3f}."
                ),
            }
        )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-audio",
        type=Path,
        default=Path("data/raw/public/earnings22/4453225.mp3"),
    )
    parser.add_argument(
        "--reference-aligned",
        type=Path,
        default=Path("data/reference/public/earnings22/4453225/reference_aligned.nlp"),
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=Path("data/manifests/earnings22_expanded_5x180.csv"),
    )
    parser.add_argument(
        "--output-audio-dir",
        type=Path,
        default=Path("data/raw/public/earnings22"),
    )
    parser.add_argument(
        "--output-reference-root",
        type=Path,
        default=Path("data/reference/public/earnings22"),
    )
    parser.add_argument("--clip-id-prefix", default="earnings22_4453225")
    parser.add_argument("--slice-seconds", type=float, default=180.0)
    parser.add_argument("--num-slices", type=int, default=5)
    parser.add_argument(
        "--start-offsets",
        type=float,
        nargs="*",
        help="Optional explicit slice start times in seconds.",
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
    try:
        rows = prepare_slices(
            source_audio=args.source_audio,
            reference_aligned=args.reference_aligned,
            output_manifest=args.output_manifest,
            output_audio_dir=args.output_audio_dir,
            output_reference_root=args.output_reference_root,
            clip_id_prefix=args.clip_id_prefix,
            slice_seconds=args.slice_seconds,
            num_slices=args.num_slices,
            start_offsets=args.start_offsets,
            min_reference_words=args.min_reference_words,
            max_reference_words_per_second=args.max_reference_words_per_second,
            min_aligned_anchor_ratio=args.min_aligned_anchor_ratio,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Earnings-22 slice preparation failed: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote {len(rows)} Earnings-22 rows to {args.output_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
