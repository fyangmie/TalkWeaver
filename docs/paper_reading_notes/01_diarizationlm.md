# DiarizationLM

## Source

Venue, year, authors, and stable link must be verified from the original
publication before final submission.

## Problem

Study how language-model post-processing can improve combined ASR and speaker
diarization output.

## Key Idea

The project requirements identify a compact textual representation of ASR and
diarization information as the motivating idea. Confirm the exact method from
the paper.

## Limitation

Record model assumptions, overlap behavior, datasets, and failure cases after
reading the source.

## Our Adaptation

Format each transcript segment with timestamps, speaker, overlap, confidence,
raw text, and retrieved terminology before constrained correction.

## Implementation Mapping

- `backend/prompting.py`
- `backend/alignment.py`
- `backend/llm_correction.py`
