"""Standard DER/JER helpers backed by pyannote.metrics when available."""

from __future__ import annotations

from typing import Any


def standard_diarization_metrics_available() -> bool:
    """Return whether pyannote's standard diarization metrics can run."""

    try:
        from pyannote.core import Annotation, Segment  # noqa: F401
        from pyannote.metrics.diarization import (  # noqa: F401
            DiarizationErrorRate,
            JaccardErrorRate,
        )
    except Exception:
        return False
    return True


def _value(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _to_annotation(turns: list[Any], *, uri: str):
    """Convert serializable turns to a pyannote Annotation."""

    from pyannote.core import Annotation, Segment

    annotation = Annotation(uri=uri)
    for index, turn in enumerate(turns):
        start = float(_value(turn, "start", 0.0))
        end = float(_value(turn, "end", 0.0))
        speaker = str(_value(turn, "speaker", "UNKNOWN"))
        if end <= start or speaker in {"", "UNKNOWN", "OVERLAP"}:
            continue
        annotation[Segment(start, end), f"track_{index:05d}"] = speaker
    return annotation


def compute_der_jer(
    reference_turns: list[Any],
    predicted_turns: list[Any],
    *,
    uri: str = "clip",
    collar: float = 0.25,
    skip_overlap: bool = False,
) -> dict[str, Any]:
    """Compute standard DER and JER with pyannote.metrics.

    Returned ``status`` is ``ok`` when both metrics were computed. If optional
    dependencies are unavailable, the row is marked ``skipped`` instead of
    returning project-specific approximations under standard metric names.
    """

    try:
        from pyannote.metrics.diarization import (
            DiarizationErrorRate,
            JaccardErrorRate,
        )
    except Exception as exc:
        return {
            "status": "skipped",
            "der": "",
            "jer": "",
            "reason": f"pyannote.metrics unavailable: {exc}",
        }

    try:
        reference = _to_annotation(reference_turns, uri=uri)
        hypothesis = _to_annotation(predicted_turns, uri=uri)
        der_metric = DiarizationErrorRate(
            collar=collar,
            skip_overlap=skip_overlap,
        )
        jer_metric = JaccardErrorRate(
            collar=collar,
            skip_overlap=skip_overlap,
        )
        return {
            "status": "ok",
            "der": float(der_metric(reference, hypothesis)),
            "jer": float(jer_metric(reference, hypothesis)),
            "reason": "",
        }
    except Exception as exc:
        return {
            "status": "error",
            "der": "",
            "jer": "",
            "reason": str(exc),
        }
