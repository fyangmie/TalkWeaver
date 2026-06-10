# TagSpeech

## Source

Preprint or publication status, year, authors, and stable link must be verified
from the original source.

## Problem

Ground multi-speaker transcript content in speaker identity and time.

## Key Idea

The project requirements identify temporal anchors as the motivating idea.
Verify the paper's terminology and technical mechanism.

## Limitation

Record training requirements, timestamp precision, speaker assumptions, and
overlap results after source review.

## Our Adaptation

Use a temporal-anchor JSON record that preserves start, end, speaker, raw and
corrected text, overlap, confidence, and retrieved terms.

## Implementation Mapping

- `backend/alignment.py`
- `backend/overlap.py`
- `backend/export.py`
- `webapp/components/speaker_timeline.py`
