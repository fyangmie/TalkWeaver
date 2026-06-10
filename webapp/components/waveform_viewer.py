"""Decoded waveform preview with graceful dependency fallbacks."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from backend.preprocessing import load_audio


def load_waveform_preview(
    path: str | Path,
    *,
    max_points: int = 4_000,
) -> dict[str, Any]:
    """Load, mix to mono, and downsample audio for visualization."""

    if max_points <= 1:
        raise ValueError("max_points must be greater than one.")
    samples, sample_rate, channels, loader = load_audio(path)
    mono = samples.mean(axis=1, dtype=np.float32)
    if mono.size == 0:
        raise ValueError("Audio contains no samples.")

    point_count = min(max_points, mono.size)
    indices = np.linspace(0, mono.size - 1, point_count, dtype=int)
    preview = mono[indices]
    times = indices / float(sample_rate)
    return {
        "time": times,
        "amplitude": preview,
        "sample_rate": sample_rate,
        "channels": channels,
        "loader": loader,
        "duration_seconds": mono.size / float(sample_rate),
    }


def render_waveform(path: str | Path | None) -> None:
    """Render a decoded waveform or explain why it is unavailable."""

    if path is None:
        render_waveform_placeholder()
        return
    try:
        preview = load_waveform_preview(path)
    except Exception as exc:
        st.info(f"Waveform preview unavailable: {exc}")
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        frame = pd.DataFrame(
            {
                "time": preview["time"],
                "amplitude": preview["amplitude"],
            }
        ).set_index("time")
        st.line_chart(frame, height=190)
        return

    figure, axis = plt.subplots(figsize=(11, 2.2))
    axis.plot(
        preview["time"],
        preview["amplitude"],
        color="#176B87",
        linewidth=0.75,
    )
    axis.axhline(0, color="#667085", linewidth=0.6)
    axis.set_xlabel("Time (seconds)")
    axis.set_ylabel("Amplitude")
    axis.set_xlim(0, max(0.01, preview["duration_seconds"]))
    axis.grid(axis="x", alpha=0.18)
    figure.tight_layout()
    st.pyplot(figure, width="stretch")
    plt.close(figure)
    st.caption(
        f"Decoded with {preview['loader']} at "
        f"{preview['sample_rate']:,} Hz; display is downsampled only."
    )


def render_waveform_placeholder() -> None:
    """Render a clearly labeled deterministic mock signal preview."""

    points = 180
    frame = pd.DataFrame(
        {
            "time": [index / 20 for index in range(points)],
            "amplitude": [
                0.6 * math.sin(index / 5) * math.sin(index / 31)
                for index in range(points)
            ],
        }
    ).set_index("time")
    st.line_chart(frame, height=180)
    st.caption("Mock/demo signal preview; no waveform was fabricated on disk.")
