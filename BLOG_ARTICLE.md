# When Whisper Meets a Noisy Meeting

## Building an Overlap-Aware Multi-Speaker ASR System

> Draft foundation for the final presentation-friendly article.

A clean transcript hides the hardest parts of a meeting. Several people may
interrupt one another, technical names may be rare, and a recognizer may
produce fluent text while assigning it to the wrong speaker. TalkWeaver starts
from the premise that a meeting system should preserve this uncertainty rather
than smoothing it away.

## Beyond "What Was Said?"

The central questions are who spoke, when they spoke, what evidence overlaps,
and which corrections are justified. TalkWeaver combines ASR output with
speaker turns and temporal anchors before any LLM correction is attempted.

## A Structured Correction Record

Each segment keeps its start and end time, speaker label, raw transcript,
overlap flag, confidence, retrieved terms, and corrected text. This makes the
correction auditable and allows overlap regions to receive more conservative
handling.

## RAG Has a Narrow Job

The local knowledge base retrieves likely domain terms such as
`pyannote.audio`, `speaker diarization`, `WER`, and `RAG`. It supports ASR
correction; it does not turn the project into a generic meeting chatbot.

## How We Will Evaluate It

The final study will compare preprocessing, diarization and alignment,
structured correction, glossary retrieval, and overlap-aware constraints.
Metrics will include WER, speaker-attribution error, Term Error Rate,
hallucinated corrections, overlap errors, and latency.

## Current Status

The Phase 1 repository provides the research structure, deterministic mock
pipeline, and Streamlit review workflow. Real model integration and measured
experiments are the next steps. Mock outputs are demonstrations only.
