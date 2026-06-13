"""Lightweight speaker-time and conversation-event baseline metrics."""

from __future__ import annotations

from itertools import permutations
from typing import Any


SPECIAL_SPEAKERS = {"UNKNOWN", "OVERLAP", ""}


def _value(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _normalize_turns(turns: list[Any]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for turn in turns:
        start = float(_value(turn, "start", 0.0))
        end = float(_value(turn, "end", 0.0))
        if end <= start:
            continue
        normalized.append(
            {
                "start": start,
                "end": end,
                "speaker": str(_value(turn, "speaker", "UNKNOWN")),
            }
        )
    return normalized


def _intersection(
    first: dict[str, Any],
    second: dict[str, Any],
) -> float:
    return max(
        0.0,
        min(float(first["end"]), float(second["end"]))
        - max(float(first["start"]), float(second["start"])),
    )


def _interval_iou(first: Any, second: Any) -> float:
    first_start = float(_value(first, "start", 0.0))
    first_end = float(_value(first, "end", 0.0))
    second_start = float(_value(second, "start", 0.0))
    second_end = float(_value(second, "end", 0.0))
    intersection = max(
        0.0,
        min(first_end, second_end) - max(first_start, second_start),
    )
    union = max(first_end, second_end) - min(first_start, second_start)
    return intersection / union if union > 0 else 0.0


def _merged_intervals(turns: list[dict[str, Any]]) -> list[tuple[float, float]]:
    intervals = sorted(
        (float(turn["start"]), float(turn["end"])) for turn in turns
    )
    merged: list[tuple[float, float]] = []
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def _interval_total(intervals: list[tuple[float, float]]) -> float:
    return sum(end - start for start, end in intervals)


def turn_time_coverage(
    reference_turns: list[Any],
    predicted_turns: list[Any],
) -> float:
    """Return reference speech-union time covered by any predicted turn."""

    reference = _normalize_turns(reference_turns)
    predicted = _normalize_turns(predicted_turns)
    reference_intervals = _merged_intervals(reference)
    total = _interval_total(reference_intervals)
    if total == 0:
        return 1.0
    predicted_intervals = _merged_intervals(predicted)
    covered = 0.0
    for ref_start, ref_end in reference_intervals:
        for pred_start, pred_end in predicted_intervals:
            covered += max(
                0.0,
                min(ref_end, pred_end) - max(ref_start, pred_start),
            )
    return min(1.0, covered / total)


def best_speaker_label_mapping(
    reference_turns: list[Any],
    predicted_turns: list[Any],
) -> dict[str, str]:
    """Find a one-to-one predicted-to-reference label mapping by overlap."""

    reference = _normalize_turns(reference_turns)
    predicted = _normalize_turns(predicted_turns)
    reference_labels = sorted(
        {
            turn["speaker"]
            for turn in reference
            if turn["speaker"] not in SPECIAL_SPEAKERS
        }
    )
    predicted_labels = sorted(
        {
            turn["speaker"]
            for turn in predicted
            if turn["speaker"] not in SPECIAL_SPEAKERS
        }
    )
    if not reference_labels or not predicted_labels:
        return {}

    scores = {
        (predicted_label, reference_label): sum(
            _intersection(predicted_turn, reference_turn)
            for predicted_turn in predicted
            if predicted_turn["speaker"] == predicted_label
            for reference_turn in reference
            if reference_turn["speaker"] == reference_label
        )
        for predicted_label in predicted_labels
        for reference_label in reference_labels
    }
    best_mapping: dict[str, str] = {}
    best_score = -1.0
    if len(predicted_labels) <= len(reference_labels):
        for assignment in permutations(
            reference_labels,
            len(predicted_labels),
        ):
            mapping = dict(zip(predicted_labels, assignment))
            score = sum(scores[(predicted, reference_label)] for predicted, reference_label in mapping.items())
            if score > best_score:
                best_mapping = mapping
                best_score = score
    else:
        for selected_predictions in permutations(
            predicted_labels,
            len(reference_labels),
        ):
            mapping = dict(zip(selected_predictions, reference_labels))
            score = sum(scores[(predicted, reference_label)] for predicted, reference_label in mapping.items())
            if score > best_score:
                best_mapping = mapping
                best_score = score
    return best_mapping


def speaker_label_error_rate(
    reference_turns: list[Any],
    predicted_turns: list[Any],
) -> float:
    """Compute duration-weighted speaker error after best label mapping.

    This is a project-level speaker attribution metric, not full DER. It does
    not separately report missed speech, false alarm, or collar-aware errors.
    """

    reference = _normalize_turns(reference_turns)
    predicted = _normalize_turns(predicted_turns)
    mapping = best_speaker_label_mapping(reference, predicted)
    boundaries = sorted(
        {
            value
            for turn in [*reference, *predicted]
            for value in (float(turn["start"]), float(turn["end"]))
        }
    )
    correct = 0.0
    total = 0.0
    for start, end in zip(boundaries, boundaries[1:]):
        if end <= start:
            continue
        midpoint = (start + end) / 2
        reference_active = {
            turn["speaker"]
            for turn in reference
            if turn["start"] <= midpoint < turn["end"]
            and turn["speaker"] not in SPECIAL_SPEAKERS
        }
        if not reference_active:
            continue
        predicted_active = {
            mapping[turn["speaker"]]
            for turn in predicted
            if turn["start"] <= midpoint < turn["end"]
            and turn["speaker"] in mapping
        }
        duration = end - start
        total += duration * len(reference_active)
        correct += duration * len(reference_active & predicted_active)
    if total == 0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - correct / total))


def boundary_mean_absolute_error(
    reference_turns: list[Any],
    predicted_turns: list[Any],
) -> float | None:
    """Average start/end boundary error over greedily matched mapped turns."""

    reference = _normalize_turns(reference_turns)
    predicted = _normalize_turns(predicted_turns)
    mapping = best_speaker_label_mapping(reference, predicted)
    candidates: list[tuple[float, int, int]] = []
    for reference_index, reference_turn in enumerate(reference):
        for predicted_index, predicted_turn in enumerate(predicted):
            if mapping.get(predicted_turn["speaker"]) != reference_turn["speaker"]:
                continue
            iou = _interval_iou(reference_turn, predicted_turn)
            if iou > 0:
                candidates.append((iou, reference_index, predicted_index))
    matched_reference: set[int] = set()
    matched_predicted: set[int] = set()
    errors: list[float] = []
    for _iou, reference_index, predicted_index in sorted(
        candidates,
        reverse=True,
    ):
        if reference_index in matched_reference or predicted_index in matched_predicted:
            continue
        matched_reference.add(reference_index)
        matched_predicted.add(predicted_index)
        reference_turn = reference[reference_index]
        predicted_turn = predicted[predicted_index]
        errors.extend(
            [
                abs(reference_turn["start"] - predicted_turn["start"]),
                abs(reference_turn["end"] - predicted_turn["end"]),
            ]
        )
    return sum(errors) / len(errors) if errors else None


def _event_payload(event: Any) -> dict[str, Any]:
    return {
        "start": float(_value(event, "start", 0.0)),
        "end": float(_value(event, "end", 0.0)),
        "type": str(_value(event, "type", "")),
        "speakers": sorted(
            str(speaker)
            for speaker in (_value(event, "speakers", []) or [])
        ),
    }


def _event_scores(
    reference_events: list[Any],
    predicted_events: list[Any],
    *,
    event_type: str,
    iou_threshold: float,
    require_speaker_pair: bool,
) -> dict[str, float | int]:
    reference = [
        _event_payload(event)
        for event in reference_events
        if str(_value(event, "type", "")) == event_type
    ]
    predicted = [
        _event_payload(event)
        for event in predicted_events
        if str(_value(event, "type", "")) == event_type
    ]
    candidates: list[tuple[float, int, int]] = []
    for reference_index, reference_event in enumerate(reference):
        for predicted_index, predicted_event in enumerate(predicted):
            if (
                require_speaker_pair
                and reference_event["speakers"]
                and predicted_event["speakers"]
                and reference_event["speakers"]
                != predicted_event["speakers"]
            ):
                continue
            iou = _interval_iou(reference_event, predicted_event)
            if iou >= iou_threshold and iou > 0:
                candidates.append((iou, reference_index, predicted_index))
    matched_reference: set[int] = set()
    matched_predicted: set[int] = set()
    for _iou, reference_index, predicted_index in sorted(
        candidates,
        reverse=True,
    ):
        if reference_index in matched_reference or predicted_index in matched_predicted:
            continue
        matched_reference.add(reference_index)
        matched_predicted.add(predicted_index)

    true_positives = len(matched_reference)
    if not predicted:
        precision = 1.0 if not reference else 0.0
    else:
        precision = true_positives / len(predicted)
    if not reference:
        recall = 1.0
    else:
        recall = true_positives / len(reference)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "reference_count": len(reference),
        "predicted_count": len(predicted),
    }


def overlap_event_precision_recall_f1(
    reference_events: list[Any],
    predicted_events: list[Any],
    *,
    iou_threshold: float = 0.3,
) -> dict[str, float | int]:
    """Match overlap intervals using one-to-one interval IoU."""

    return _event_scores(
        reference_events,
        predicted_events,
        event_type="overlap",
        iou_threshold=iou_threshold,
        require_speaker_pair=False,
    )


def interruption_event_precision_recall_f1(
    reference_events: list[Any],
    predicted_events: list[Any],
) -> dict[str, float | int]:
    """Match interruption intervals and speaker pairs when both are known."""

    return _event_scores(
        reference_events,
        predicted_events,
        event_type="interruption",
        iou_threshold=1e-12,
        require_speaker_pair=True,
    )
