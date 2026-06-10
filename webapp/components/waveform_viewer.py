"""Waveform placeholder component."""

from __future__ import annotations

import math

import pandas as pd
import streamlit as st


def render_waveform_placeholder() -> None:
    """Render a deterministic signal preview until real decoding is added."""

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
    st.caption("Mock signal preview. Real waveform decoding is Phase 2 work.")
