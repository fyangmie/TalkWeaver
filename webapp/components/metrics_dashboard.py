"""Experiment result discovery and metrics dashboard rendering."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import streamlit as st


METRIC_ALIASES = {
    "WER": ("wer",),
    "WDER / speaker error": (
        "speaker_error_or_wder",
        "wder",
        "speaker_attribution_error",
        "speaker_error",
    ),
    "Term Error Rate": ("term_error_rate", "ter"),
    "Overlap analysis": (
        "overlap_error",
        "overlap_wer",
        "overlap_error_rate",
    ),
    "Latency": ("latency_seconds", "latency"),
}


def discover_result_csvs(directories: Iterable[str | Path]) -> list[Path]:
    """Return unique CSV result files from configured directories."""

    files: list[Path] = []
    for directory in directories:
        root = Path(directory)
        if root.exists():
            files.extend(root.glob("*.csv"))
    return sorted({path.resolve() for path in files})


def load_result_frames(paths: Iterable[str | Path]) -> pd.DataFrame:
    """Load result CSVs into one frame with source provenance."""

    frames: list[pd.DataFrame] = []
    for path_value in paths:
        path = Path(path_value)
        frame = pd.read_csv(path)
        frame.insert(0, "source_file", path.name)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def contains_mock_metrics(frame: pd.DataFrame) -> bool:
    """Return whether any row is explicitly labeled mock or demo."""

    if "is_mock" in frame:
        values = frame["is_mock"].fillna(False).astype(str).str.lower()
        if values.isin({"true", "1", "yes"}).any():
            return True
    for column in ("result_type", "mode", "status"):
        if column in frame:
            values = frame[column].fillna("").astype(str).str.lower()
            if values.str.contains("mock|demo", regex=True).any():
                return True
    return False


def metric_snapshot(frame: pd.DataFrame) -> dict[str, float | None]:
    """Extract the latest numeric value for each required metric."""

    snapshot: dict[str, float | None] = {}
    for label, aliases in METRIC_ALIASES.items():
        value: float | None = None
        for alias in aliases:
            if alias not in frame:
                continue
            numeric = pd.to_numeric(frame[alias], errors="coerce").dropna()
            if not numeric.empty:
                value = float(numeric.iloc[-1])
                break
        snapshot[label] = value
    return snapshot


def _format_metric(label: str, value: float | None) -> str:
    if value is None:
        return "Pending"
    if label == "Latency":
        return f"{value:.2f} s"
    return f"{value:.3f}"


def render_metrics_dashboard(
    *,
    result_directories: Iterable[str | Path],
    chart_directory: str | Path,
) -> None:
    """Render measured metrics, result tables, and saved charts."""

    csv_paths = discover_result_csvs(result_directories)
    frame = load_result_frames(csv_paths)
    snapshot = metric_snapshot(frame)
    columns = st.columns(len(METRIC_ALIASES))
    for column, (label, value) in zip(columns, snapshot.items()):
        column.metric(label, _format_metric(label, value))

    if frame.empty:
        st.info(
            "No experiment CSV files were found. Run the evaluation scripts "
            "after adding reference transcripts and speaker labels."
        )
    else:
        if contains_mock_metrics(frame):
            st.warning(
                "Mock/demo rows are scaffolds only. Empty values are not "
                "performance measurements and must not be cited."
            )
        st.subheader("Experiment Results")
        st.dataframe(frame, width="stretch", hide_index=True)
        st.caption(
            "Expected evaluation axes: WER, WDER or speaker attribution "
            "error, Term Error Rate, overlap error analysis, and latency."
        )

    chart_dir = Path(chart_directory)
    chart_paths = sorted(
        path
        for pattern in ("*.png", "*.jpg", "*.jpeg")
        for path in chart_dir.glob(pattern)
    ) if chart_dir.exists() else []
    st.subheader("Result Charts")
    if not chart_paths:
        st.info(
            "No result charts are available yet. "
            "`python experiments/plot_results.py` will populate this section "
            "when numeric reference-backed results exist."
        )
        return
    for chart_path in chart_paths:
        st.image(chart_path, caption=chart_path.name, width="stretch")


def render_metrics(result_path: Path) -> None:
    """Backward-compatible single-file dashboard entry point."""

    render_metrics_dashboard(
        result_directories=[result_path.parent],
        chart_directory=result_path.parents[2] / "assets" / "result_charts",
    )
