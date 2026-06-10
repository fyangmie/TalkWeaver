"""JSON and Markdown export helpers."""

from __future__ import annotations

import json
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
