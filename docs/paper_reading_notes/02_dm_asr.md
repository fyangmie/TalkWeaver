# DM-ASR

## Source

Li Li, Ming Cheng, Weixin Zhu, Yannan Wang, Juan Liu, and Ming Li.
*DM-ASR: Diarization-aware Multi-speaker ASR with Large Language Models*.
arXiv preprint, submitted April 24, 2026.

<https://arxiv.org/abs/2604.22467>

## Problem

Multi-speaker ASR must jointly model lexical content, speaker identity, and
time. Learning all three inside one system is difficult and data-intensive.

## Key Idea

Use diarization as an explicit structural prior and decompose recognition into
speaker- and time-conditioned queries. Each query targets one speaker in one
time segment. The paper also describes optional interleaved word timestamps.

## Limitation

DM-ASR is a trained speech-LLM framework evaluated on Mandarin and English
benchmarks. TalkWeaver does not reproduce its model, training data, or reported
results. Errors in the diarization prior can propagate into each conditioned
query.

## Our Adaptation

TalkWeaver performs correction one temporal speaker segment at a time. The
speaker and time anchors remain immutable during correction.

## Implementation Mapping

- `backend/prompting.py`
- `backend/llm_correction.py`
- `backend/pipeline.py`
- `webapp/components/transcript_viewer.py`
