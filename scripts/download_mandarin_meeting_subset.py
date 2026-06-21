#!/usr/bin/env python3
"""Attempt a size-capped official Mandarin meeting subset acquisition."""

from __future__ import annotations

import argparse
import re
import sys
import tarfile
import wave
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.dataset_utils import (  # noqa: E402
    download_file,
    ensure_directory_markers,
    repo_relative,
    write_manifest,
    write_reference_bundle,
)


MAX_AUTOMATIC_BYTES = 500 * 1024 * 1024
AISHELL4_ARCHIVES = {
    "test.tar.gz": "https://openslr.trmal.net/resources/111/test.tar.gz",
    "train_L.tar.gz": "https://openslr.trmal.net/resources/111/train_L.tar.gz",
}
AISHELL4_SPLIT_ARCHIVES = {
    "test": "test.tar.gz",
    "train_L": "train_L.tar.gz",
}
ALIMEETING_PAGE = "https://www.modelscope.cn/datasets/modelscope/AliMeeting"
AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a"}
DEFAULT_LARGE_DOWNLOAD_BYTES = 12 * 1024 * 1024 * 1024


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


def _audio_duration_seconds(path: Path) -> float:
    if path.suffix.lower() == ".wav":
        with wave.open(str(path), "rb") as handle:
            frames = handle.getnframes()
            rate = handle.getframerate()
            return frames / rate if rate else 0.0
    return 0.0


def _audio_duration_any(path: Path) -> float:
    try:
        import soundfile as sf

        info = sf.info(str(path))
        return float(info.frames / info.samplerate) if info.samplerate else 0.0
    except Exception:
        return _audio_duration_seconds(path)


def _path_for_manifest(path: Path) -> str:
    try:
        return repo_relative(path)
    except ValueError:
        return str(path.resolve())


def _find_transcript(
    audio_path: Path,
    *,
    audio_root: Path,
    transcript_root: Path | None,
) -> Path | None:
    relative = audio_path.relative_to(audio_root)
    candidates = [
        audio_path.with_suffix(".txt"),
        audio_path.with_name(audio_path.stem + ".transcript.txt"),
    ]
    if transcript_root is not None:
        candidates.extend(
            [
                transcript_root / relative.with_suffix(".txt"),
                transcript_root / f"{audio_path.stem}.txt",
                transcript_root / f"{audio_path.stem}.transcript.txt",
            ]
        )
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _read_kaldi_text(path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                rows[parts[0]] = parts[1].strip()
    return rows


def _read_kaldi_segments(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 4 or parts[0].startswith("#"):
                continue
            try:
                start = float(parts[2])
                end = float(parts[3])
            except ValueError:
                continue
            if end <= start:
                continue
            rows.append(
                {
                    "utt_id": parts[0],
                    "recording_id": parts[1],
                    "start": start,
                    "end": end,
                }
            )
    return rows


def _clean_aishell_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    return re.sub(r"\s+", "", text).strip()


def _read_textgrid_segments(path: Path, *, recording_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    speaker = "UNKNOWN"
    current: dict[str, Any] | None = None
    interval_index = 0
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if line.startswith("name ="):
            speaker = line.split("=", 1)[1].strip().strip('"') or "UNKNOWN"
        elif line.startswith("intervals ["):
            current = {"speaker": speaker}
        elif current is not None and line.startswith("xmin ="):
            try:
                current["start"] = float(line.split("=", 1)[1].strip())
            except ValueError:
                current["start"] = 0.0
        elif current is not None and line.startswith("xmax ="):
            try:
                current["end"] = float(line.split("=", 1)[1].strip())
            except ValueError:
                current["end"] = 0.0
        elif current is not None and line.startswith("text ="):
            text = _clean_aishell_text(line.split("=", 1)[1].strip().strip('"'))
            start = float(current.get("start", 0.0))
            end = float(current.get("end", 0.0))
            if text and end > start:
                interval_index += 1
                rows.append(
                    {
                        "utt_id": f"{recording_id}_{interval_index:05d}",
                        "recording_id": recording_id,
                        "start": start,
                        "end": end,
                        "speaker": str(current.get("speaker") or "UNKNOWN"),
                        "text": text,
                    }
                )
            current = None
    return rows


def _read_utt2spk(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    rows: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                rows[parts[0]] = parts[1]
    return rows


def _find_kaldi_bundle(root: Path) -> tuple[Path, Path, Path | None] | None:
    for segments_path in sorted(root.rglob("segments")):
        parent = segments_path.parent
        text_path = parent / "text"
        if text_path.exists():
            utt2spk_path = parent / "utt2spk"
            return segments_path, text_path, utt2spk_path if utt2spk_path.exists() else None
    return None


def _audio_index(root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
            index.setdefault(path.stem, path)
            index.setdefault(path.name, path)
    return index


def _match_audio(recording_id: str, index: dict[str, Path]) -> Path | None:
    if recording_id in index:
        return index[recording_id]
    normalized = recording_id.replace("/", "_")
    if normalized in index:
        return index[normalized]
    for stem, path in index.items():
        if stem == recording_id or stem.startswith(recording_id) or recording_id.startswith(stem):
            return path
    return None


def _compute_overlap_events(anchors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for left_index, left in enumerate(anchors):
        for right in anchors[left_index + 1 :]:
            if left.get("speaker") == right.get("speaker"):
                continue
            start = max(float(left["start"]), float(right["start"]))
            end = min(float(left["end"]), float(right["end"]))
            if end > start:
                events.append(
                    {
                        "start": round(start, 3),
                        "end": round(end, 3),
                        "type": "overlap",
                        "speakers": sorted({left["speaker"], right["speaker"]}),
                        "source": "aishell4_segments_intersection",
                    }
                )
    return events


def _write_audio_slice(
    source: Path,
    destination: Path,
    *,
    start_seconds: float,
    duration_seconds: float,
) -> float:
    import numpy as np
    import soundfile as sf

    info = sf.info(str(source))
    start_frame = max(0, int(round(start_seconds * info.samplerate)))
    stop_frame = int(round((start_seconds + duration_seconds) * info.samplerate))
    data, sample_rate = sf.read(
        str(source),
        start=start_frame,
        stop=stop_frame,
        always_2d=True,
        dtype="float32",
    )
    if data.size == 0:
        raise ValueError(f"No audio samples found in slice {source} @ {start_seconds}")
    mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(destination), np.asarray(mono, dtype="float32"), sample_rate)
    return float(len(mono) / sample_rate) if sample_rate else 0.0


def _safe_extract_tar(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            target = (destination / member.name).resolve()
            if not str(target).startswith(str(root)):
                raise RuntimeError(f"Unsafe archive member path: {member.name}")
        archive.extractall(destination)


def _ensure_aishell4_split(
    *,
    split: str,
    output_root: Path,
    allow_large_download: bool,
    max_download_bytes: int,
) -> Path:
    archive_name = AISHELL4_SPLIT_ARCHIVES[split]
    url = AISHELL4_ARCHIVES[archive_name]
    source_root = output_root / "_source"
    archive_path = source_root / archive_name
    extract_root = source_root / f"aishell4_{split}"
    marker = extract_root / ".extracted"
    source_root.mkdir(parents=True, exist_ok=True)
    size, final_url = _head_size(url)
    limit = max_download_bytes if allow_large_download else MAX_AUTOMATIC_BYTES
    if size > limit:
        raise RuntimeError(
            f"AISHELL-4 {archive_name} is {size} bytes at {final_url}; pass "
            "--allow-large-download to permit this archive."
        )
    if not archive_path.exists() or archive_path.stat().st_size == 0:
        download_file(url, archive_path, max_bytes=limit, timeout=3600)
    if not marker.exists():
        _safe_extract_tar(archive_path, extract_root)
        marker.write_text("ok\n", encoding="utf-8")
    return extract_root


def prepare_aishell4_subset(
    *,
    split: str,
    output_root: Path,
    reference_root: Path,
    manifest_out: Path,
    max_clips: int,
    max_clips_per_recording: int,
    clip_duration_seconds: float,
    allow_large_download: bool,
    max_download_bytes: int,
    extracted_root: Path | None = None,
) -> list[dict[str, Any]]:
    extract_root = (
        extracted_root
        if extracted_root is not None
        else _ensure_aishell4_split(
            split=split,
            output_root=output_root,
            allow_large_download=allow_large_download,
            max_download_bytes=max_download_bytes,
        )
    )
    bundle = _find_kaldi_bundle(extract_root)
    audio_by_recording = _audio_index(extract_root)
    by_recording: dict[str, list[dict[str, Any]]] = {}
    reference_source_note = ""
    text_by_utt: dict[str, str] = {}
    utt2spk: dict[str, str] = {}
    if bundle is not None:
        segments_path, text_path, utt2spk_path = bundle
        text_by_utt = {
            utt: _clean_aishell_text(text)
            for utt, text in _read_kaldi_text(text_path).items()
        }
        utt2spk = _read_utt2spk(utt2spk_path)
        for segment in _read_kaldi_segments(segments_path):
            if text_by_utt.get(segment["utt_id"]):
                by_recording.setdefault(segment["recording_id"], []).append(segment)
        reference_source_note = (
            f"References parsed from {segments_path.relative_to(extract_root)} "
            f"and {text_path.relative_to(extract_root)}."
        )
    else:
        for textgrid_path in sorted(extract_root.rglob("*.TextGrid")):
            recording_id = textgrid_path.stem
            if _match_audio(recording_id, audio_by_recording) is None:
                continue
            segments = _read_textgrid_segments(textgrid_path, recording_id=recording_id)
            for segment in segments:
                text_by_utt[segment["utt_id"]] = segment["text"]
                utt2spk[segment["utt_id"]] = segment["speaker"]
                by_recording.setdefault(recording_id, []).append(segment)
        reference_source_note = "References parsed from AISHELL-4 TextGrid files."

    if not by_recording:
        write_manifest(manifest_out, [])
        raise RuntimeError(
            f"No usable AISHELL-4 audio+transcript references found under {extract_root}."
        )

    rows: list[dict[str, Any]] = []
    for recording_id, segments in sorted(by_recording.items()):
        if len(rows) >= max_clips:
            break
        recording_rows = 0
        audio_path = _match_audio(recording_id, audio_by_recording)
        if audio_path is None:
            continue
        segments = sorted(segments, key=lambda item: (item["start"], item["end"]))
        audio_duration = _audio_duration_any(audio_path)
        segment_index = 0
        while segment_index < len(segments) and len(rows) < max_clips:
            if max_clips_per_recording > 0 and recording_rows >= max_clips_per_recording:
                break
            window_start = max(0.0, float(segments[segment_index]["start"]))
            if audio_duration:
                window_start = min(
                    window_start,
                    max(0.0, audio_duration - clip_duration_seconds),
                )
            window_end = window_start + clip_duration_seconds
            window_segments = [
                segment
                for segment in segments
                if float(segment["end"]) > window_start
                and float(segment["start"]) < window_end
            ]
            if not window_segments:
                segment_index += 1
                continue
            clip_id = f"aishell4_{split}_{len(rows) + 1:03d}"
            clip_path = output_root / f"{clip_id}.wav"
            duration = _write_audio_slice(
                audio_path,
                clip_path,
                start_seconds=window_start,
                duration_seconds=clip_duration_seconds,
            )
            anchors: list[dict[str, Any]] = []
            for segment in window_segments:
                utt_id = segment["utt_id"]
                start = max(0.0, float(segment["start"]) - window_start)
                end = min(duration, float(segment["end"]) - window_start)
                if end <= start:
                    continue
                anchors.append(
                    {
                        "start": round(start, 3),
                        "end": round(end, 3),
                        "speaker": utt2spk.get(utt_id, segment.get("speaker", "UNKNOWN")),
                        "text": text_by_utt[utt_id],
                        "utterance_id": utt_id,
                    }
                )
            if not anchors:
                segment_index += 1
                continue
            transcript = " ".join(anchor["text"] for anchor in anchors)
            events = _compute_overlap_events(anchors)
            refs = write_reference_bundle(
                reference_root / clip_id,
                transcript=transcript,
                anchors=anchors,
                terms=[],
                events=events,
            )
            speakers = {
                anchor["speaker"]
                for anchor in anchors
                if anchor["speaker"] != "UNKNOWN"
            }
            rows.append(
                {
                    "clip_id": clip_id,
                    "audio_path": _path_for_manifest(clip_path),
                    "source_type": "public_dataset",
                    "dataset_name": "AISHELL-4",
                    "dataset_version": "OpenSLR 111",
                    "split": split,
                    "language": "zh-CN",
                    "duration_seconds": f"{duration:.3f}",
                    "speaker_count": str(len(speakers)) if speakers else "",
                    "has_overlap": bool(events),
                    "has_interruptions": "",
                    "has_domain_terms": "false",
                    "recording_device": "AISHELL-4 meeting recording",
                    "noise_condition": "meeting_room",
                    "consent_status": "dataset_terms",
                    "redistribution_status": "raw_audio_not_committed",
                    "license_or_access": "AISHELL-4 OpenSLR 111; preserve official terms and citation.",
                    "transcript_path": _path_for_manifest(refs["transcript_path"]),
                    "anchors_path": _path_for_manifest(refs["anchors_path"]),
                    "terms_path": _path_for_manifest(refs["terms_path"]),
                    "events_path": _path_for_manifest(refs["events_path"]),
                    "download_status": "prepared",
                    "notes": (
                        f"Source recording {recording_id}, {window_start:.3f}-"
                        f"{window_start + duration:.3f}s. {reference_source_note}"
                    ),
                }
            )
            recording_rows += 1
            while (
                segment_index < len(segments)
                and float(segments[segment_index]["start"]) < window_end
            ):
                segment_index += 1
    write_manifest(manifest_out, rows)
    return rows


def prepare_local_mandarin_meeting_subset(
    *,
    local_audio_root: Path,
    local_transcript_root: Path | None,
    reference_root: Path,
    manifest_out: Path,
    max_clips: int,
    dataset_name: str = "Mandarin meeting local subset",
    dataset_version: str = "local_user_prepared",
    license_or_access: str = "local user-prepared subset; verify source license before sharing",
) -> list[dict[str, Any]]:
    audio_root = local_audio_root.resolve()
    transcript_root = local_transcript_root.resolve() if local_transcript_root else None
    rows: list[dict[str, Any]] = []
    for audio_path in sorted(audio_root.rglob("*")):
        if len(rows) >= max_clips:
            break
        if not audio_path.is_file() or audio_path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        transcript_path = _find_transcript(
            audio_path,
            audio_root=audio_root,
            transcript_root=transcript_root,
        )
        if transcript_path is None:
            continue
        transcript = transcript_path.read_text(encoding="utf-8").strip()
        if not transcript:
            continue
        duration = _audio_duration_seconds(audio_path)
        clip_id = f"mandarin_meeting_local_{len(rows) + 1:03d}"
        refs = write_reference_bundle(
            reference_root / clip_id,
            transcript=transcript,
            anchors=[
                {
                    "start": 0.0,
                    "end": duration,
                    "speaker": "UNKNOWN",
                    "text": transcript,
                }
            ],
            terms=[],
            events=[],
        )
        rows.append(
            {
                "clip_id": clip_id,
                "audio_path": _path_for_manifest(audio_path),
                "source_type": "public_dataset_local_import",
                "dataset_name": dataset_name,
                "dataset_version": dataset_version,
                "split": "local prepared subset",
                "language": "zh-CN",
                "duration_seconds": f"{duration:.3f}" if duration else "",
                "speaker_count": "",
                "has_overlap": "",
                "has_interruptions": "",
                "has_domain_terms": "false",
                "recording_device": "unknown",
                "noise_condition": "meeting_room",
                "consent_status": "dataset_terms_or_local_permission",
                "redistribution_status": "raw_audio_not_committed",
                "license_or_access": license_or_access,
                "transcript_path": _path_for_manifest(refs["transcript_path"]),
                "anchors_path": _path_for_manifest(refs["anchors_path"]),
                "terms_path": _path_for_manifest(refs["terms_path"]),
                "events_path": _path_for_manifest(refs["events_path"]),
                "download_status": "prepared",
                "notes": (
                    "Local Mandarin meeting import. Transcript is real local "
                    "reference; speaker/overlap fields are not diarization gold "
                    "unless separate annotations are added."
                ),
            }
        )
    write_manifest(manifest_out, rows)
    return rows


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
        "--max-clips-per-recording",
        type=int,
        default=0,
        help=(
            "Maximum AISHELL-4 clips to slice from one recording. Use a "
            "positive value for balanced multi-meeting benchmark subsets; "
            "0 preserves the original uncapped behavior."
        ),
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=sorted(AISHELL4_SPLIT_ARCHIVES),
        help="AISHELL-4 split/archive to use when --dataset aishell4 is selected.",
    )
    parser.add_argument(
        "--clip-duration-seconds",
        type=float,
        default=20.0,
    )
    parser.add_argument(
        "--allow-large-download",
        action="store_true",
        help="Allow AISHELL-4 archives larger than the default 500 MB ceiling.",
    )
    parser.add_argument(
        "--max-download-bytes",
        type=int,
        default=DEFAULT_LARGE_DOWNLOAD_BYTES,
        help="Safety ceiling used with --allow-large-download.",
    )
    parser.add_argument(
        "--aishell4-extracted-root",
        type=Path,
        help=(
            "Optional already-extracted AISHELL-4 directory. Useful when an "
            "external downloader or partial archive has produced usable files."
        ),
    )
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
    parser.add_argument(
        "--local-audio-root",
        type=Path,
        help=(
            "Optional local directory containing Mandarin meeting audio files. "
            "When provided, the script imports complete audio+transcript pairs "
            "instead of attempting network acquisition."
        ),
    )
    parser.add_argument(
        "--local-transcript-root",
        type=Path,
        help=(
            "Optional local transcript directory. Matching uses relative path "
            "with .txt extension or <audio_stem>.txt."
        ),
    )
    parser.add_argument(
        "--local-dataset-name",
        default="Mandarin meeting local subset",
    )
    parser.add_argument(
        "--local-dataset-version",
        default="local_user_prepared",
    )
    parser.add_argument(
        "--local-license-or-access",
        default="local user-prepared subset; verify source license before sharing",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    ensure_directory_markers(
        [args.output_root, args.reference_root, args.manifest_out.parent]
    )
    if args.local_audio_root:
        rows = prepare_local_mandarin_meeting_subset(
            local_audio_root=args.local_audio_root,
            local_transcript_root=args.local_transcript_root,
            reference_root=args.reference_root,
            manifest_out=args.manifest_out,
            max_clips=args.max_clips,
            dataset_name=args.local_dataset_name,
            dataset_version=args.local_dataset_version,
            license_or_access=args.local_license_or_access,
        )
        if rows:
            print(
                f"Prepared {len(rows)} local Mandarin meeting rows: "
                f"{args.manifest_out}"
            )
            return 0
        print(
            "No complete local Mandarin meeting audio+transcript pairs were found; "
            "wrote an empty manifest.",
            file=sys.stderr,
        )
        return 2
    if args.dataset == "aishell4":
        rows = prepare_aishell4_subset(
            split=args.split,
            output_root=args.output_root,
            reference_root=args.reference_root,
            manifest_out=args.manifest_out,
            max_clips=args.max_clips,
            max_clips_per_recording=args.max_clips_per_recording,
            clip_duration_seconds=args.clip_duration_seconds,
            allow_large_download=args.allow_large_download,
            max_download_bytes=args.max_download_bytes,
            extracted_root=args.aishell4_extracted_root,
        )
        if rows:
            print(f"Prepared {len(rows)} AISHELL-4 rows: {args.manifest_out}")
            return 0
        print("No AISHELL-4 rows were prepared.", file=sys.stderr)
        return 2
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
