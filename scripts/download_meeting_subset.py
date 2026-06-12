#!/usr/bin/env python3
"""Download and prepare a tiny official English meeting/overlap subset."""

from __future__ import annotations

import argparse
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.dataset_utils import (  # noqa: E402
    download_file,
    ensure_directory_markers,
    repo_relative,
    write_checksums,
    write_manifest,
    write_reference_bundle,
)


AMI_MEETING_ID = "ES2002a"
AMI_AUDIO_URL = (
    "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/"
    f"{AMI_MEETING_ID}/audio/{AMI_MEETING_ID}.Mix-Headset.wav"
)
AMI_ANNOTATION_URL = (
    "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/"
    "ami_public_manual_1.6.2.zip"
)


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _read_ami_words(
    annotation_zip: Path,
    meeting_id: str,
) -> list[dict[str, Any]]:
    words: list[dict[str, Any]] = []
    with zipfile.ZipFile(annotation_zip) as archive:
        names = [
            name
            for name in archive.namelist()
            if f"/words/{meeting_id}." in f"/{name}"
            and name.endswith(".words.xml")
        ]
        if not names:
            raise RuntimeError(
                f"No word annotations for {meeting_id} in {annotation_zip}."
            )
        for name in names:
            filename = Path(name).name
            parts = filename.split(".")
            speaker = parts[1] if len(parts) >= 3 else "UNKNOWN"
            root = ElementTree.fromstring(archive.read(name))
            for element in root.iter():
                if _local_name(element.tag) != "w":
                    continue
                text = (element.text or "").strip()
                start = element.attrib.get("starttime")
                end = element.attrib.get("endtime")
                if not text or start is None or end is None:
                    continue
                words.append(
                    {
                        "start": float(start),
                        "end": float(end),
                        "speaker": f"SPEAKER_{speaker}",
                        "text": text,
                    }
                )
    return sorted(words, key=lambda item: (item["start"], item["end"]))


def _window_has_overlap(words: list[dict[str, Any]]) -> bool:
    by_speaker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for word in words:
        by_speaker[word["speaker"]].append(word)
    speakers = sorted(by_speaker)
    for index, speaker in enumerate(speakers):
        for other in speakers[index + 1 :]:
            for left in by_speaker[speaker]:
                for right in by_speaker[other]:
                    if min(left["end"], right["end"]) > max(
                        left["start"], right["start"]
                    ):
                        return True
    return False


def _select_windows(
    words: list[dict[str, Any]],
    max_clips: int,
    *,
    duration: float = 20.0,
) -> list[tuple[float, float]]:
    candidates: list[tuple[bool, float, float]] = []
    for word in words:
        start = max(0.0, word["start"] - 2.0)
        end = start + duration
        selected = [item for item in words if item["start"] < end and item["end"] > start]
        if len({item["speaker"] for item in selected}) < 2:
            continue
        candidates.append((_window_has_overlap(selected), start, end))
    candidates.sort(key=lambda item: (not item[0], item[1]))
    windows: list[tuple[float, float]] = []
    for _, start, end in candidates:
        if any(min(end, old_end) - max(start, old_start) > 2 for old_start, old_end in windows):
            continue
        windows.append((start, end))
        if len(windows) >= max_clips:
            break
    if not windows:
        raise RuntimeError("No multi-speaker AMI windows were found.")
    return windows


def _group_anchors(
    words: list[dict[str, Any]],
    window_start: float,
    window_end: float,
) -> list[dict[str, Any]]:
    clipped = []
    for word in words:
        if word["start"] >= window_end or word["end"] <= window_start:
            continue
        clipped.append(
            {
                **word,
                "start": max(0.0, word["start"] - window_start),
                "end": min(window_end, word["end"]) - window_start,
            }
        )
    by_speaker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for word in clipped:
        by_speaker[word["speaker"]].append(word)
    anchors: list[dict[str, Any]] = []
    for speaker, speaker_words in by_speaker.items():
        current: list[dict[str, Any]] = []
        for word in speaker_words:
            if current and word["start"] - current[-1]["end"] > 0.8:
                anchors.append(_anchor_from_words(speaker, current))
                current = []
            current.append(word)
        if current:
            anchors.append(_anchor_from_words(speaker, current))
    return sorted(anchors, key=lambda item: (item["start"], item["end"]))


def _anchor_from_words(
    speaker: str,
    words: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "start": round(words[0]["start"], 3),
        "end": round(words[-1]["end"], 3),
        "speaker": speaker,
        "speakers": [speaker],
        "text": " ".join(word["text"] for word in words),
        "overlap": False,
        "annotation_source": "AMI manual word annotations",
    }


def _overlap_events(anchors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for index, left in enumerate(anchors):
        for right in anchors[index + 1 :]:
            if left["speaker"] == right["speaker"]:
                continue
            start = max(left["start"], right["start"])
            end = min(left["end"], right["end"])
            if end - start < 0.05:
                continue
            left["overlap"] = True
            right["overlap"] = True
            events.append(
                {
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "duration": round(end - start, 3),
                    "speakers": sorted([left["speaker"], right["speaker"]]),
                    "type": "overlap",
                    "annotation_source": "derived_from_AMI_manual_word_intervals",
                }
            )
    return events


def prepare_ami_subset(
    *,
    max_clips: int,
    output_root: Path,
    reference_root: Path,
    manifest_out: Path,
) -> list[dict[str, Any]]:
    ensure_directory_markers([output_root, reference_root, manifest_out.parent])
    source_dir = output_root / "_source"
    source_audio = source_dir / f"{AMI_MEETING_ID}.Mix-Headset.wav"
    annotation_zip = source_dir / "ami_public_manual_1.6.2.zip"
    if not source_audio.exists():
        download_file(AMI_AUDIO_URL, source_audio, max_bytes=60 * 1024 * 1024)
    if not annotation_zip.exists():
        download_file(
            AMI_ANNOTATION_URL,
            annotation_zip,
            max_bytes=40 * 1024 * 1024,
        )

    words = _read_ami_words(annotation_zip, AMI_MEETING_ID)
    windows = _select_windows(words, max_clips)
    rows: list[dict[str, Any]] = []
    checksum_files: list[Path] = []

    with sf.SoundFile(source_audio) as audio:
        sample_rate = audio.samplerate
        for index, (start, end) in enumerate(windows, start=1):
            audio.seek(int(start * sample_rate))
            samples = audio.read(int((end - start) * sample_rate), dtype="float32")
            clip_id = f"ami_{AMI_MEETING_ID.lower()}_{index:02d}"
            clip_path = output_root / f"{clip_id}.wav"
            sf.write(clip_path, samples, sample_rate)

            window_words = [
                item
                for item in words
                if item["start"] < end and item["end"] > start
            ]
            transcript = " ".join(item["text"] for item in window_words)
            anchors = _group_anchors(words, start, end)
            events = _overlap_events(anchors)
            references = write_reference_bundle(
                reference_root / clip_id,
                transcript=transcript,
                anchors=anchors,
                events=events,
            )
            checksum_files.extend([clip_path, *references.values()])
            rows.append(
                {
                    "clip_id": clip_id,
                    "audio_path": repo_relative(clip_path),
                    "source_type": "public_dataset",
                    "dataset_name": "AMI Meeting Corpus",
                    "dataset_version": "manual annotations 1.6.2",
                    "split": "official meeting excerpt",
                    "language": "en",
                    "duration_seconds": f"{end - start:.3f}",
                    "speaker_count": len({item["speaker"] for item in anchors}),
                    "has_overlap": bool(events),
                    "has_interruptions": False,
                    "has_domain_terms": False,
                    "recording_device": "AMI Mix-Headset",
                    "noise_condition": "meeting_room",
                    "consent_status": "dataset_terms",
                    "redistribution_status": "raw_audio_not_committed",
                    "license_or_access": (
                        "AMI Corpus CC BY 4.0; preserve attribution and source IDs."
                    ),
                    **{
                        key: repo_relative(value)
                        for key, value in references.items()
                    },
                    "download_status": "prepared",
                    "notes": (
                        f"Source {AMI_MEETING_ID}, {start:.3f}-{end:.3f}s. "
                        "Overlap events derived from manual word intervals; "
                        "interruption labels need_annotation."
                    ),
                }
            )

    write_manifest(manifest_out, rows)
    write_checksums(
        manifest_out.with_name(f"{manifest_out.stem}_checksums.csv"),
        checksum_files,
    )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="auto",
        choices=["auto", "ami", "libricss", "ami_or_librics_or_auto"],
    )
    parser.add_argument("--max-clips", type=int, default=3)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/raw/public/english_meeting"),
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=Path("data/reference/public/english_meeting"),
    )
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=Path("data/manifests/english_meeting_real.csv"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.max_clips < 1:
        raise SystemExit("--max-clips must be positive.")
    if args.dataset == "libricss":
        print(
            "LibriCSS official releases are distributed as session archives; "
            "the small downloader currently uses AMI file-level downloads.",
            file=sys.stderr,
        )
        return 2
    try:
        rows = prepare_ami_subset(
            max_clips=args.max_clips,
            output_root=args.output_root,
            reference_root=args.reference_root,
            manifest_out=args.manifest_out,
        )
    except Exception as exc:
        print(f"English meeting subset acquisition failed: {exc}", file=sys.stderr)
        return 2
    print(f"Prepared {len(rows)} real AMI meeting clips at {args.manifest_out}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
