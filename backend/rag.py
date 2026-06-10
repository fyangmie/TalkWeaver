"""Auxiliary domain-term retrieval for ASR correction."""

from __future__ import annotations

from pathlib import Path


DOMAIN_TERMS = [
    "pyannote.audio",
    "speaker diarization",
    "overlapping speech",
    "cross-speech",
    "ASR",
    "WER",
    "DER",
    "WDER",
    "RAG",
    "faster-whisper",
    "VAD",
    "LLM correction",
    "temporal anchor",
    "speaker attribution",
]

TERM_HINTS = {
    "piano note": ["pyannote.audio"],
    "diary station": ["speaker diarization"],
    "rack": ["RAG"],
    "where": ["WER"],
    "the ear": ["DER"],
}


def load_knowledge_base(directory: str | Path) -> list[str]:
    """Load local Markdown documents without requiring a vector database."""

    path = Path(directory)
    return [
        document.read_text(encoding="utf-8")
        for document in sorted(path.glob("*.md"))
    ]


def retrieve_terms(text: str, *, limit: int = 4) -> list[str]:
    """Return deterministic glossary candidates for Phase 1 mock correction."""

    lowered = text.lower()
    candidates: list[str] = []
    for phrase, terms in TERM_HINTS.items():
        if phrase in lowered:
            candidates.extend(terms)
    for term in DOMAIN_TERMS:
        if term.lower() in lowered:
            candidates.append(term)
    return list(dict.fromkeys(candidates))[:limit]
