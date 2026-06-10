# TalkWeaver Project Report

**Title:** TalkWeaver: An Overlap-Aware Multi-Speaker ASR System with
Diarization-Structured LLM Correction

**Subtitle:** RAG-Enhanced Domain Term Recovery for Noisy Meeting Speech

> Status: report template. Replace all placeholders with verified sources,
> reproducible experiment settings, and clearly labeled real results.

## 1. Abstract

Summarize the noisy multi-speaker meeting problem, the overlap-aware pipeline,
the diarization-structured correction method, the auxiliary glossary
retrieval module, and the measured findings. Do not claim state-of-the-art
performance.

## 2. Introduction

Describe why ASR alone is insufficient for meetings. Motivate speaker
attribution, overlap uncertainty, domain terminology, and auditable
correction.

## 3. Related Work

Review the provided course paper when available, DiarizationLM, diarization-
aware multi-speaker ASR, DM-ASR, temporal-anchor approaches such as TagSpeech,
and retrieval-augmented ASR correction. Cite only verified sources.

## 4. Research Gaps

1. ASR and diarization outputs can be misaligned.
2. Overlapping speech creates ambiguous evidence.
3. Unconstrained LLM correction can hallucinate or erase uncertainty.
4. Rare domain terms are frequently mistranscribed.
5. Research systems can be difficult to reproduce in a final-project setting.

## 5. Method

### 5.1 Audio Preprocessing

Document mono conversion, 16 kHz resampling, normalization, optional denoising,
and VAD.

### 5.2 ASR and Diarization

Document faster-whisper and pyannote configurations, fallbacks, and output
schemas.

### 5.3 Alignment and Overlap Analysis

Explain timestamp-midpoint assignment, overlap-region detection, and
confidence estimation.

### 5.4 Temporal-Anchor Transcript

Define the JSON record containing timestamps, speaker, raw and corrected text,
overlap, confidence, and retrieved terms.

### 5.5 Structured LLM Correction

Explain speaker-time conditioned prompts, audit trails, and conservative
correction rules.

### 5.6 RAG Domain-Term Recovery

Explain local knowledge-base loading, candidate retrieval, and why RAG is an
auxiliary correction mechanism rather than the main project.

## 6. System Architecture

Insert the final architecture figure and map every stage to its repository
module.

## 7. Experiments

Describe datasets, synthetic-overlap generation, references, comparison
groups A-F, metrics, hardware, software versions, and reproducibility steps.

## 8. Results

Add real result tables and confidence intervals where appropriate. Keep mock
or smoke-test outputs in a separate, explicitly labeled subsection.

## 9. Error Analysis

Analyze speaker swaps, missed overlap, domain-term substitutions, unsupported
LLM edits, and errors caused by poor audio quality.

## 10. Limitations

Discuss model access, dataset size, language coverage, diarization assumptions,
manual annotation quality, latency, and the limits of post-hoc correction.

## 11. Future Work

Consider stronger overlap-aware models, calibrated confidence, improved
speaker linking, multilingual evaluation, and larger controlled studies.

## 12. Conclusion

Conclude with evidence tied directly to the four research questions. Emphasize
research understanding, engineering adaptation, and reproducible evaluation.
