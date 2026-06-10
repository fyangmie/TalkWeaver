# DiarizationLM

## Source

Quan Wang, Yiling Huang, Guanlong Zhao, Evan Clark, Wei Xia, and Hank Liao.
*DiarizationLM: Speaker Diarization Post-Processing with Large Language
Models*. Proceedings of Interspeech 2024, pages 3754-3758.

<https://arxiv.org/abs/2401.03506>

## Problem

ASR and diarization systems can produce readable words with incorrect speaker
assignment. Existing component outputs need a post-processing method that can
use linguistic context without retraining the ASR or diarization models.

## Key Idea

Represent ASR and diarization output in a compact textual format and include
it in a prompt to an optionally fine-tuned LLM. The LLM can improve readability
or refine speaker assignment. The paper evaluates WDER on telephone
conversation datasets.

## Limitation

The method depends on upstream ASR and diarization quality and on the
post-processing model. A compact transcript cannot restore acoustic evidence
that was lost, and paper-reported improvements should not be transferred to a
different pipeline without evaluation.

## Our Adaptation

TalkWeaver uses a compact speaker-time segment with timestamps, active
speakers, overlap, confidence, raw text, and retrieved terms. The adaptation
adds explicit uncertainty and preserves a raw-versus-corrected audit trail.

## Implementation Mapping

- `backend/prompting.py`
- `backend/alignment.py`
- `backend/llm_correction.py`
- `experiments/evaluate_wder.py`
