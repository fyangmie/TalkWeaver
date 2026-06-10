"""JSON and Markdown export helpers."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


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
    title: str = "TalkWeaver Mock Transcript",
) -> Path:
    """Write a readable speaker-attributed transcript."""

    lines = [
        f"# {title}",
        "",
        "> This file contains deterministic mock/demo output.",
        "",
    ]
    for segment in segments:
        warning = " [OVERLAP - REVIEW]" if segment["overlap"] else ""
        lines.extend(
            [
                (
                    f"## {segment['start']:.2f}-{segment['end']:.2f} "
                    f"{segment['speaker']}{warning}"
                ),
                "",
                f"**Raw:** {segment['raw_text']}",
                "",
                f"**Corrected:** {segment['corrected_text']}",
                "",
                (
                    "**Retrieved terms:** "
                    + (", ".join(segment["retrieved_terms"]) or "none")
                ),
                "",
            ]
        )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    return output
