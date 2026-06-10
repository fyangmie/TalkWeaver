# TagSpeech

## Source

Mingyue Huo, Yiwen Shao, and Yuheng Zhang. *TagSpeech: End-to-End
Multi-Speaker ASR and Diarization with Fine-Grained Temporal Grounding*.
arXiv preprint, submitted January 11, 2026.

<https://arxiv.org/abs/2601.06896>

## Problem

Multi-speaker systems need fine-grained synchronization among lexical content,
speaker identity, and time, particularly during complex overlap.

## Key Idea

TagSpeech uses Temporal Anchor Grounding, decoupled semantic and speaker
streams, Serialized Output Training, and interleaved time anchors. The design
explicitly models who spoke what and when.

## Limitation

The paper is an end-to-end trained framework evaluated on AMI and AliMeeting.
Its training setup and benchmark results are outside TalkWeaver's lightweight
post-processing scope.

## Our Adaptation

TalkWeaver exports a temporal-anchor JSON record connecting start/end time,
speaker identities, raw and corrected text, overlap, confidence, and retrieved
terms.

## Implementation Mapping

- `backend/alignment.py`
- `backend/overlap.py`
- `backend/export.py`
- `webapp/components/speaker_timeline.py`
- `experiments/evaluate_wder.py`
