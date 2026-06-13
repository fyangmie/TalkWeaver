"""Reusable evidence components for correction and audit views."""

from __future__ import annotations

import html
import re
from difflib import SequenceMatcher
from typing import Any

import streamlit as st

from webapp.data_loader import (
    get_anchor_audio_window,
    get_event_audio_window,
    resolve_local_audio_path,
)

TOKEN_PATTERN = re.compile(r"\w+(?:[.-]\w+)*|[^\w\s]", re.UNICODE)


def _tokens(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(str(text or ""))


def _join_tokens(tokens: list[str]) -> str:
    text = " ".join(tokens)
    return re.sub(r"\s+([,.;:!?])", r"\1", text)


def build_text_diff(raw_text: str, corrected_text: str) -> dict[str, Any]:
    """Build an approximate token-level diff with escaped HTML fragments."""

    raw = str(raw_text or "")
    corrected = str(corrected_text or "")
    raw_tokens = _tokens(raw)
    corrected_tokens = _tokens(corrected)
    matcher = SequenceMatcher(a=raw_tokens, b=corrected_tokens, autojunk=False)
    raw_parts: list[str] = []
    corrected_parts: list[str] = []
    changes: list[dict[str, Any]] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        old_tokens = raw_tokens[i1:i2]
        new_tokens = corrected_tokens[j1:j2]
        old_text = html.escape(_join_tokens(old_tokens))
        new_text = html.escape(_join_tokens(new_tokens))
        if tag == "equal":
            raw_parts.append(old_text)
            corrected_parts.append(new_text)
            continue
        if old_text:
            raw_parts.append(f'<span class="tw-diff-removed">{old_text}</span>')
        if new_text:
            corrected_parts.append(f'<span class="tw-diff-added">{new_text}</span>')
        changes.append(
            {
                "operation": tag,
                "removed": _join_tokens(old_tokens),
                "added": _join_tokens(new_tokens),
            }
        )

    return {
        "identical": raw == corrected,
        "raw_text": raw,
        "corrected_text": corrected,
        "raw_html": " ".join(part for part in raw_parts if part),
        "corrected_html": " ".join(part for part in corrected_parts if part),
        "changes": changes,
    }


def render_text_diff(
    raw_text: str,
    corrected_text: str,
    *,
    raw_label: str = "Raw ASR evidence",
    corrected_label: str = "Corrected / retained text",
) -> dict[str, Any]:
    """Render raw and corrected text while returning testable diff metadata."""

    diff = build_text_diff(raw_text, corrected_text)
    if diff["identical"]:
        st.info(
            "No lexical correction was applied. TalkWeaver retained the raw "
            "evidence rather than forcing an unsupported edit."
        )
    raw_column, corrected_column = st.columns(2)
    raw_column.markdown(f"**{raw_label}**")
    raw_column.markdown(
        f'<div class="tw-diff-panel tw-diff-raw">{diff["raw_html"]}</div>',
        unsafe_allow_html=True,
    )
    corrected_column.markdown(f"**{corrected_label}**")
    corrected_column.markdown(
        '<div class="tw-diff-panel tw-diff-corrected">'
        f'{diff["corrected_html"] or "(empty)"}</div>',
        unsafe_allow_html=True,
    )
    return diff


def render_audio_evidence(
    conversation_map: dict[str, Any],
    item: dict[str, Any] | None = None,
    *,
    item_type: str = "event",
    padding_seconds: float = 0.5,
    label: str = "Audio evidence",
) -> dict[str, Any]:
    """Render a whole clip or timestamp-bounded local audio evidence player."""

    path = resolve_local_audio_path(conversation_map)
    if path is None:
        st.info(
            "Local audio is unavailable. Raw public/private audio is ignored "
            "by Git, so regenerate or download it locally to enable playback."
        )
        return {"available": False, "path": None, "window": None}

    duration_value = conversation_map.get("metadata", {}).get("duration_seconds")
    try:
        duration = float(duration_value)
    except (TypeError, ValueError):
        duration = None

    if item is None:
        st.markdown(f"**{label}**")
        st.audio(path)
        return {"available": True, "path": path, "window": None}

    if item_type == "anchor":
        window = get_anchor_audio_window(
            item,
            padding_seconds,
            duration_seconds=duration,
        )
    else:
        window = get_event_audio_window(
            item,
            padding_seconds,
            duration_seconds=duration,
        )
    st.markdown(f"**{label}**")
    st.audio(
        path,
        start_time=window["start"],
        end_time=window["end"],
    )
    st.caption(
        f"Evidence window {window['start']:.2f}s-{window['end']:.2f}s. "
        "If the browser does not honor the seek position, jump to "
        f"approximately {window['start']:.1f}s-{window['end']:.1f}s "
        "in the audio player."
    )
    return {"available": True, "path": path, "window": window}
