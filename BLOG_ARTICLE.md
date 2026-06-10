# When Whisper Meets a Noisy Meeting

## Building TalkWeaver, an Overlap-Aware Multi-Speaker ASR System

A clean paragraph is often the wrong representation of a meeting. It hides
interruptions, speaker changes, uncertain boundaries, and the difference
between what the recognizer heard and what a language model later rewrote.
TalkWeaver was built around a stricter question:

> Can we improve a transcript without losing who spoke, when they spoke, or
> where the evidence became ambiguous?

## Beyond "What Was Said?"

Automatic speech recognition handles lexical content. Speaker diarization
estimates who spoke when. In real meetings those outputs do not always agree,
especially around cross-speech. A fluent correction can make the final text
look better while assigning it to the wrong speaker or hiding uncertainty.

TalkWeaver therefore keeps ASR and diarization separate long enough to audit
them, then combines them through timestamp alignment. Every word is assigned
using its temporal midpoint. If two turns contain that midpoint, the result is
not forced to one speaker: it becomes an explicit overlap segment.

## A Transcript with Temporal Anchors

The central data structure stores start and end times, speaker labels, all
active speakers, raw text, corrected text, overlap state, confidence, and
retrieved terminology. That record connects the backend, exports, experiments,
and Streamlit interface.

This design is inspired by several research directions. DiarizationLM shows
that compact ASR and diarization representations can support LLM
post-processing. DM-ASR uses speaker- and time-conditioned subtasks.
TagSpeech emphasizes fine-grained temporal grounding for who spoke what and
when. TalkWeaver does not reproduce those trained models; it adapts their
structural ideas into a lightweight pipeline.

## Correction Has Constraints

Correction is performed per speaker-time segment. Timestamps and speaker
labels are preserved. Overlap segments receive conservative instructions and
remain uncertain. An output validator rejects empty text, unsupported words,
and rearrangements beyond glossary-backed substitutions.

Without an API key, the same interface runs deterministic rules. This is
important for reproducibility: the project can demonstrate the complete
workflow without presenting mock behavior as a real model result.

## RAG Has One Narrow Job

The local TF-IDF index retrieves terms such as `pyannote`, `diarization`,
`WER`, `DER`, and `RAG`. It helps recover technical terms from errors like
`piano note`, `diary station`, and `the ear`.

RAG does not become the main application. It supplies correction candidates
and supports secondary transcript search. Speaker attribution, overlap, and
LLM + ASR interaction remain the research center.

## Measuring the Pipeline

Phase 7 adds WER, a clearly labeled temporal speaker-error approximation, Term
Error Rate, overlap error, hallucinated correction checks, and per-stage
latency. The ablation runner produces six rows from Whisper-only through
overlap-aware correction, and the plotting script creates five presentation
charts.

The included values are deterministic mock/demo metrics. They prove that the
evaluators, CSV schema, dashboard, and charts work. They do not prove that the
system generalizes. A real study still needs consented or licensed audio,
reference transcripts, speaker labels, overlap annotations, fixed model
versions, and a held-out test set.

## What the Project Demonstrates

TalkWeaver is not a claim of state-of-the-art speech recognition. It is a
research-driven engineering project showing that:

- speaker and time structure can remain visible through correction;
- overlap uncertainty can be represented instead of silently erased;
- retrieval can be restricted to domain-term recovery;
- every correction can retain its raw evidence;
- mock and real evaluation can share one reproducible interface.

The next step is not another UI feature. It is the careful annotation and
execution of the real A-F ablation study.
