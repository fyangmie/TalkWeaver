"""Audio persistence, metadata, and playback helpers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import streamlit as st

from backend.preprocessing import load_audio


def safe_upload_name(name: str) -> str:
    """Return a filesystem-safe upload name without directory traversal."""

    original = Path(name).name
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(original).stem).strip("._")
    suffix = re.sub(r"[^A-Za-z0-9.]+", "", Path(original).suffix.lower())
    return f"{stem or 'meeting_audio'}{suffix}"


def save_uploaded_audio(uploaded_file: Any, directory: str | Path) -> Path:
    """Persist one Streamlit upload and return its absolute path."""

    output_dir = Path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / safe_upload_name(str(uploaded_file.name))
    if hasattr(uploaded_file, "getbuffer"):
        payload = bytes(uploaded_file.getbuffer())
    else:
        payload = uploaded_file.read()
    output_path.write_bytes(payload)
    return output_path.resolve()


def get_audio_metadata(path: str | Path) -> dict[str, Any]:
    """Decode basic audio metadata through the Phase 2 loader."""

    audio_path = Path(path)
    samples, sample_rate, channels, loader = load_audio(audio_path)
    duration = len(samples) / sample_rate if sample_rate else 0.0
    return {
        "name": audio_path.name,
        "path": str(audio_path.resolve()),
        "size_bytes": audio_path.stat().st_size,
        "duration_seconds": round(float(duration), 3),
        "sample_rate": int(sample_rate),
        "channels": int(channels),
        "loader": loader,
        "format": audio_path.suffix.lower().lstrip(".") or "unknown",
    }


def render_audio_metadata(metadata: dict[str, Any]) -> None:
    """Render compact audio metadata columns."""

    columns = st.columns(5)
    columns[0].metric("Duration", f"{metadata['duration_seconds']:.2f} s")
    columns[1].metric("Sample rate", f"{metadata['sample_rate']:,} Hz")
    columns[2].metric("Channels", str(metadata["channels"]))
    columns[3].metric("Size", f"{metadata['size_bytes'] / 1_048_576:.2f} MB")
    columns[4].metric("Decoder", str(metadata["loader"]))


def render_audio_player(audio: str | Path | Any | None) -> None:
    """Render local or uploaded audio with a concise empty state."""

    if audio is None:
        st.info("No audio is loaded. Upload a recording or use mock mode.")
        return

    if isinstance(audio, (str, Path)):
        audio_path = Path(audio)
        if not audio_path.exists():
            st.warning(f"Audio file is no longer available: {audio_path}")
            return
        media_format = f"audio/{audio_path.suffix.lower().lstrip('.') or 'wav'}"
        st.audio(audio_path.read_bytes(), format=media_format)
        st.caption(str(audio_path))
        return

    st.audio(audio)
    name = getattr(audio, "name", "uploaded audio")
    size = getattr(audio, "size", 0)
    st.caption(f"{name} | {size / 1_048_576:.2f} MB")
