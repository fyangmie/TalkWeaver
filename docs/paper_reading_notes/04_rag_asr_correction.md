# Retrieval-Augmented ASR Correction

## Source

Ernest Pusateri et al. *Retrieval Augmented Correction of Named Entity Speech
Recognition Errors*. arXiv preprint submitted September 9, 2024; the arXiv
record states that it was submitted to ICASSP 2025.

<https://arxiv.org/abs/2409.06062>

## Problem

End-to-end ASR systems continue to misrecognize rare entity names that appear
infrequently in training data.

## Key Idea

Generate retrieval queries from errorful ASR hypotheses, retrieve relevant
entities from a vector database, and provide the candidates plus the
hypothesis to an adapted LLM for correction.

## Limitation

Irrelevant candidates can bias correction. The paper focuses on rare named
entities and synthetic voice-assistant tests, while TalkWeaver focuses on
technical meeting terminology and multi-speaker structure.

## Our Adaptation

TalkWeaver loads local Markdown, retrieves a small TF-IDF-ranked term list,
and permits only glossary-supported substitutions. RAG remains an auxiliary
module.

## Implementation Mapping

- `backend/rag.py`
- `backend/llm_correction.py`
- `experiments/evaluate_terms.py`
- `docs/knowledge_base/domain_terms.md`
