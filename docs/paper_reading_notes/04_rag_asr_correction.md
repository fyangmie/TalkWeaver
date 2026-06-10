# Retrieval-Augmented ASR Correction

## Source

Identify the exact paper or papers, publication status, authors, year, and
stable links before final submission.

## Problem

ASR systems often mistranscribe rare entities and domain-specific terms.

## Key Idea

Retrieve plausible rare terms or entities and provide them as constrained
context for correction. Verify retrieval method and evaluation details from
the selected source.

## Limitation

Irrelevant candidates can bias correction, and retrieval cannot recover
content unsupported by the audio.

## Our Adaptation

Load a local Markdown knowledge base, retrieve a short list of project-domain
terms, and measure both retrieval quality and final Term Error Rate.

## Implementation Mapping

- `backend/rag.py`
- `backend/llm_correction.py`
- `experiments/evaluate_terms.py`
- `docs/knowledge_base/domain_terms.md`
