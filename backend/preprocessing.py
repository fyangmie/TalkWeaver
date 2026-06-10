"""Audio preprocessing interface.

Real waveform processing is intentionally deferred to Phase 2.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def preprocess_audio(
    audio_path: str | Path | None,
    *,
    mock: bool = False,
    denoise: bool = False,
) -> dict[str, Any]:
    """Return preprocessing metadata or a clear Phase 2 error."""

    if mock:
        return {
            "mode": "mock_demo",
            "input": str(audio_path) if audio_path else None,
            "sample_rate": 16000,
            "channels": 1,
            "normalized": True,
            "denoised": denoise,
            "note": "No waveform was modified in Phase 1 mock mode.",
        }

    if audio_path is None:
        raise ValueError("An audio path is required outside mock mode.")

    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    raise RuntimeError(
        "Real preprocessing is scheduled for Phase 2. Run with --mock for now."
    )
