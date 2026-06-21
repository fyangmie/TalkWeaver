# Course Anchor Paper: Xu Tong Thesis Initial Note

## Source

Local file checked on June 20, 2026:

```text
参考文献/xutong_paper.pdf
```

PDF metadata reports 79 pages. The first-page title is:

```text
基于语音识别与大语言模型的多人对话内容管理系统
```

This note is an abstract-level initial read. Full section-by-section claims
must be checked against the thesis text before final submission.

## Problem

The thesis targets practical meeting-content management. The abstract
identifies long audio, alternating multi-speaker speech, multilingual mixing,
speaker distinction errors, repeated or semantically broken ASR output,
insufficient speaker separation, and weak structure in meeting summaries as
core problems.

## Key Idea

The system combines speech recognition and large language models. The abstract
describes:

- multiple ASR models and decoding strategy optimization for long-audio
  stability;
- deep speaker-embedding features plus hierarchical clustering for automatic
  speaker distinction;
- LLM post-processing for text correction, speaker attribution adjustment,
  scene inference, stance organization, and structured summary generation;
- a three-stage approach for heavy cross-speech: signal-level separation,
  feature-level clustering, and semantic-level correction;
- Streamlit integration for upload, transcription, speaker labeling, and
  structured export.

## Limitation

The abstract-level description does not yet establish whether the thesis has
public held-out datasets, standard DER/WER reporting, reproducible ablations,
or hallucination/correction provenance controls. Those claims require reading
the experiment chapters.

## Our Adaptation

TalkWeaver keeps the same broad problem framing but narrows the research claim
to auditable evidence:

- temporal-anchor records instead of free-form meeting summaries;
- explicit overlap and uncertainty fields;
- reference-backed ASR, speaker, overlap, and term-recovery metrics;
- constrained RAG/LLM correction with unsupported-change checks;
- separate mock, diagnostic, held-out, and proxy-mobile claim levels.

## Implementation Mapping

- ASR stability: `backend/asr.py`, `experiments/run_asr_benchmark.py`
- Speaker/time evidence: `backend/diarization.py`, `backend/alignment.py`,
  `experiments/run_speaker_overlap_baseline.py`
- Cross-speech evidence: `backend/overlap.py`,
  `experiments/evaluate_overlap_safety.py`
- LLM post-processing: `backend/prompting.py`, `backend/llm_correction.py`
- Term recovery: `backend/rag.py`, `backend/term_verifier.py`
- Streamlit review: `webapp/`
