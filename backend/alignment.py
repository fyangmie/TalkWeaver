"""Word-speaker alignment and temporal-anchor transcript construction."""

from __future__ import annotations

from typing import Any

from backend.confidence import confidence_for_assignment, uncertainty_label


def _speakers_at_time(
    timestamp: float,
    speaker_turns: list[dict[str, Any]],
) -> list[str]:
    return sorted(
        {
            str(turn["speaker"])
            for turn in speaker_turns
            if float(turn["start"]) <= timestamp < float(turn["end"])
        }
    )


def _word_entries(
    asr_segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for segment_index, segment in enumerate(asr_segments):
        segment_start = float(segment["start"])
        segment_end = float(segment["end"])
        words = segment.get("words") or []
        timed_words = [
            word
            for word in words
            if word.get("start") is not None and word.get("end") is not None
        ]
        if not timed_words:
            entries.append(
                {
                    "word": str(segment.get("text", "")).strip(),
                    "start": segment_start,
                    "end": segment_end,
                    "segment_index": segment_index,
                    "word_index": 0,
                }
            )
            continue

        for word_index, word in enumerate(timed_words):
            entries.append(
                {
                    "word": str(word.get("word", "")).strip(),
                    "start": float(word["start"]),
                    "end": float(word["end"]),
                    "segment_index": segment_index,
                    "word_index": word_index,
                }
            )
    return sorted(
        entries,
        key=lambda word: (
            word["start"],
            word["end"],
            word["segment_index"],
            word["word_index"],
        ),
    )


def align_words_to_speakers(
    asr_segments: list[dict[str, Any]],
    speaker_turns: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Assign each timed ASR word using its timestamp midpoint."""

    aligned_words: list[dict[str, Any]] = []
    for word in _word_entries(asr_segments):
        midpoint = (float(word["start"]) + float(word["end"])) / 2
        speakers = _speakers_at_time(midpoint, speaker_turns)
        overlap = len(speakers) > 1
        if overlap:
            speaker = "OVERLAP"
        elif speakers:
            speaker = speakers[0]
        else:
            speaker = "UNKNOWN"
        confidence = confidence_for_assignment(speaker, overlap=overlap)
        aligned_words.append(
            {
                "word": word["word"],
                "start": round(float(word["start"]), 3),
                "end": round(float(word["end"]), 3),
                "speaker": speaker,
                "speakers": speakers,
                "overlap": overlap,
                "confidence": confidence,
                "uncertainty": uncertainty_label(confidence),
            }
        )
    return aligned_words


def _new_segment(word: dict[str, Any]) -> dict[str, Any]:
    return {
        "start": word["start"],
        "end": word["end"],
        "speaker": word["speaker"],
        "speakers": list(word["speakers"]),
        "raw_text": word["word"],
        "corrected_text": "",
        "overlap": word["overlap"],
        "confidence": word["confidence"],
        "uncertainty": word["uncertainty"],
        "retrieved_terms": [],
        "words": [word],
    }


def group_aligned_words(
    aligned_words: list[dict[str, Any]],
    *,
    max_gap_seconds: float = 1.0,
) -> list[dict[str, Any]]:
    """Group consecutive words with the same speaker assignment."""

    segments: list[dict[str, Any]] = []
    for word in aligned_words:
        if not segments:
            segments.append(_new_segment(word))
            continue

        current = segments[-1]
        same_assignment = (
            current["speaker"] == word["speaker"]
            and current["speakers"] == word["speakers"]
            and current["overlap"] == word["overlap"]
        )
        gap = float(word["start"]) - float(current["end"])
        if same_assignment and gap <= max_gap_seconds:
            current["end"] = word["end"]
            current["raw_text"] = (
                f"{current['raw_text']} {word['word']}".strip()
            )
            current["confidence"] = min(
                float(current["confidence"]),
                float(word["confidence"]),
            )
            current["words"].append(word)
            continue
        segments.append(_new_segment(word))
    return segments


def align_segments(
    asr_segments: list[dict[str, Any]],
    speaker_turns: list[dict[str, Any]],
    overlap_regions: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build temporal-anchor segments from word-level midpoint assignments."""

    del overlap_regions  # Overlap is derived directly from active turns per word.
    aligned_words = align_words_to_speakers(asr_segments, speaker_turns)
    return group_aligned_words(aligned_words)
