"""JSON and Markdown export helpers."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def read_json(path: str | Path) -> Any:
    """Read a UTF-8 JSON artifact."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> Path:
    """Write UTF-8 JSON and create parent directories."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return output


def write_raw_transcript_markdown(
    path: str | Path,
    segments: list[dict[str, Any]],
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write a timestamped raw ASR transcript with optional word timing."""

    mode = (metadata or {}).get("mode", "unknown")
    lines = [
        "# TalkWeaver Raw ASR Transcript",
        "",
        f"**Execution mode:** `{mode}`",
        "",
    ]
    fallback_reason = (metadata or {}).get("fallback_reason")
    if fallback_reason:
        lines.extend(
            [
                f"> **Fallback notice:** {fallback_reason}",
                "",
            ]
        )
    if mode.startswith("mock"):
        lines.extend(
            [
                "> This is deterministic mock/demo output, not real ASR.",
                "",
            ]
        )

    if not segments:
        lines.extend(["No speech segments were produced.", ""])

    for index, segment in enumerate(segments, start=1):
        lines.extend(
            [
                (
                    f"## Segment {index}: "
                    f"{float(segment['start']):.2f}-"
                    f"{float(segment['end']):.2f}"
                ),
                "",
                str(segment["text"]),
                "",
            ]
        )
        words = segment.get("words", [])
        if words:
            lines.extend(
                [
                    "| Word | Start | End |",
                    "| --- | ---: | ---: |",
                ]
            )
            for word in words:
                start = word.get("start")
                end = word.get("end")
                start_text = "" if start is None else f"{float(start):.2f}"
                end_text = "" if end is None else f"{float(end):.2f}"
                safe_word = str(word.get("word", "")).replace("|", "\\|")
                lines.append(f"| {safe_word} | {start_text} | {end_text} |")
            lines.append("")

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def _safe_stem(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return normalized or "transcript"


def export_raw_transcript(
    output_dir: str | Path,
    stem: str,
    result: dict[str, Any],
) -> dict[str, Path]:
    """Export raw ASR segments as JSON and Markdown under one directory."""

    directory = Path(output_dir)
    safe_stem = _safe_stem(stem)
    segments = result["segments"]
    metadata = {key: value for key, value in result.items() if key != "segments"}
    paths = {
        "json": write_json(directory / f"{safe_stem}.json", segments),
        "markdown": write_raw_transcript_markdown(
            directory / f"{safe_stem}.md",
            segments,
            metadata=result,
        ),
        "metadata": write_json(
            directory / f"{safe_stem}.metadata.json",
            metadata,
        ),
    }
    return paths


def write_transcript_markdown(
    path: str | Path,
    segments: list[dict[str, Any]],
    *,
    title: str = "TalkWeaver Speaker-Attributed Transcript",
    mode: str = "unknown",
) -> Path:
    """Write a readable speaker-attributed transcript."""

    lines = [
        f"# {title}",
        "",
        f"**Execution mode:** `{mode}`",
        "",
    ]
    if mode.startswith("mock"):
        lines.extend(
            [
                "> This is deterministic mock/demo diarization output, not "
                "a real speaker analysis.",
                "",
            ]
        )
    for segment in segments:
        warning = " [OVERLAP - REVIEW REQUIRED]" if segment["overlap"] else ""
        speakers = ", ".join(segment.get("speakers", [])) or "none"
        lines.extend(
            [
                (
                    f"## {segment['start']:.2f}-{segment['end']:.2f} "
                    f"{segment['speaker']}{warning}"
                ),
                "",
                f"**Speakers:** {speakers}",
                "",
                f"**Confidence:** {float(segment['confidence']):.2f}",
                "",
                f"**Raw:** {segment['raw_text']}",
                "",
            ]
        )
        if segment["overlap"]:
            lines.extend(
                [
                    "> Overlapping speech was detected. This segment should "
                    "be reviewed before later LLM correction.",
                    "",
                ]
            )
    if not segments:
        lines.extend(["No aligned transcript segments were produced.", ""])
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def write_diarization_markdown(
    path: str | Path,
    turns: list[dict[str, Any]],
    *,
    metadata: dict[str, Any],
) -> Path:
    """Write diarization turns as a readable review table."""

    mode = str(metadata.get("mode", "unknown"))
    lines = [
        "# TalkWeaver Speaker Diarization",
        "",
        f"**Execution mode:** `{mode}`",
        "",
    ]
    fallback_reason = metadata.get("fallback_reason")
    if fallback_reason:
        lines.extend([f"> **Fallback notice:** {fallback_reason}", ""])
    if mode.startswith("mock"):
        lines.extend(
            [
                "> These speaker turns are deterministic mock/demo output.",
                "",
            ]
        )
    lines.extend(
        [
            "| Start | End | Speaker |",
            "| ---: | ---: | --- |",
        ]
    )
    for turn in turns:
        lines.append(
            f"| {float(turn['start']):.2f} | {float(turn['end']):.2f} "
            f"| {turn['speaker']} |"
        )
    lines.append("")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def write_overlap_markdown(
    path: str | Path,
    regions: list[dict[str, Any]],
    *,
    mode: str,
) -> Path:
    """Write overlap regions and review warnings."""

    lines = [
        "# TalkWeaver Overlap Warnings",
        "",
        f"**Execution mode:** `{mode}`",
        "",
    ]
    if mode.startswith("mock"):
        lines.extend(
            [
                "> Mock/demo overlap is deliberate and must not be reported "
                "as a real diarization result.",
                "",
            ]
        )
    if not regions:
        lines.extend(["No overlapping speaker turns were detected.", ""])
    else:
        lines.extend(
            [
                "> Overlapping speech can reduce speaker attribution and ASR "
                "confidence. Review these intervals before correction.",
                "",
                "| Start | End | Duration | Speakers |",
                "| ---: | ---: | ---: | --- |",
            ]
        )
        for region in regions:
            speakers = ", ".join(region["speakers"])
            lines.append(
                f"| {float(region['start']):.2f} "
                f"| {float(region['end']):.2f} "
                f"| {float(region['duration']):.2f} "
                f"| {speakers} |"
            )
        lines.append("")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def export_diarization(
    output_dir: str | Path,
    stem: str,
    result: dict[str, Any],
) -> dict[str, Path]:
    """Export diarization turns, metadata, and review Markdown."""

    directory = Path(output_dir)
    safe_stem = _safe_stem(stem)
    turns = result["turns"]
    metadata = {key: value for key, value in result.items() if key != "turns"}
    return {
        "json": write_json(directory / f"{safe_stem}.json", turns),
        "markdown": write_diarization_markdown(
            directory / f"{safe_stem}.md",
            turns,
            metadata=result,
        ),
        "metadata": write_json(
            directory / f"{safe_stem}.metadata.json",
            metadata,
        ),
    }


def export_overlap_regions(
    output_dir: str | Path,
    regions: list[dict[str, Any]],
    *,
    mode: str,
) -> dict[str, Path]:
    """Export overlap regions to stable JSON and Markdown filenames."""

    directory = Path(output_dir)
    return {
        "json": write_json(directory / "overlap_regions.json", regions),
        "markdown": write_overlap_markdown(
            directory / "overlap_warnings.md",
            regions,
            mode=mode,
        ),
    }


def export_temporal_anchor_transcript(
    output_dir: str | Path,
    stem: str,
    segments: list[dict[str, Any]],
    *,
    mode: str,
) -> dict[str, Path]:
    """Export Phase 3 temporal anchors as JSON and review Markdown."""

    directory = Path(output_dir)
    safe_stem = _safe_stem(stem)
    return {
        "json": write_json(
            directory / f"{safe_stem}_temporal_anchor.json",
            segments,
        ),
        "markdown": write_transcript_markdown(
            directory / f"{safe_stem}_speaker_transcript.md",
            segments,
            mode=mode,
        ),
    }
