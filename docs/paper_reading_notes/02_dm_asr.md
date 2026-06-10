# DM-ASR

## Source

Preprint or publication status, year, authors, and link must be verified from
the original source.

## Problem

Multi-speaker ASR must preserve speaker and temporal structure, especially
when speech overlaps.

## Key Idea

The project requirements describe speaker- and time-conditioned queries.
Confirm the exact architecture and experiments from the paper.

## Limitation

Document compute requirements, model access, overlap assumptions, and
generalization limits after reading.

## Our Adaptation

Correct one speaker-time segment at a time instead of submitting the entire
meeting as an unstructured block.

## Implementation Mapping

- `backend/prompting.py`
- `backend/llm_correction.py`
- `backend/pipeline.py`
