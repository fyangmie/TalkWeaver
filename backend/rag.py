"""Local TF-IDF retrieval for domain-term ASR recovery."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.config import ROOT_DIR


DEFAULT_KNOWLEDGE_BASE = ROOT_DIR / "docs" / "knowledge_base"
DEFAULT_DOMAIN_TERMS = (
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
)
DEFAULT_CORRECTION_PAIRS = {
    "piano note": "pyannote",
    "diary station": "diarization",
    "where": "WER",
    "the ear": "DER",
    "rack": "RAG",
}
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[.-][A-Za-z0-9]+)*")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "be",
    "can",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "use",
    "we",
    "with",
}


@dataclass(frozen=True)
class KnowledgeDocument:
    """One local Markdown knowledge-base document."""

    source: str
    text: str


@dataclass(frozen=True)
class KnowledgeChunk:
    """A retrievable paragraph or glossary row."""

    source: str
    text: str
    terms: tuple[str, ...] = ()


def _tokenize(text: str) -> list[str]:
    return [
        token
        for match in TOKEN_PATTERN.finditer(text)
        if (token := match.group(0).lower()) not in STOPWORDS
    ]


def load_knowledge_base(
    directory: str | Path = DEFAULT_KNOWLEDGE_BASE,
) -> list[KnowledgeDocument]:
    """Load all local Markdown documents in deterministic filename order."""

    path = Path(directory)
    if not path.exists():
        raise FileNotFoundError(f"Knowledge-base directory not found: {path}")
    documents = [
        KnowledgeDocument(
            source=document.name,
            text=document.read_text(encoding="utf-8"),
        )
        for document in sorted(path.glob("*.md"))
    ]
    if not documents:
        raise ValueError(f"No Markdown knowledge-base files found in: {path}")
    return documents


def _parse_glossary_terms(text: str) -> list[str]:
    terms: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|") or stripped.startswith("| ---"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if not cells or cells[0].lower() == "term":
            continue
        terms.append(cells[0])
    return terms


def _parse_correction_pairs(text: str) -> dict[str, str]:
    pairs: dict[str, str] = {}
    for source, target in re.findall(
        r"^\s*([A-Za-z0-9 ._-]+?)\s*->\s*([A-Za-z0-9 ._-]+?)\s*$",
        text,
        flags=re.MULTILINE,
    ):
        pairs[source.strip().lower()] = target.strip()
    return pairs


def _terms_in_text(text: str, domain_terms: list[str]) -> tuple[str, ...]:
    lowered = text.lower()
    return tuple(term for term in domain_terms if term.lower() in lowered)


def _build_chunks(
    documents: list[KnowledgeDocument],
    domain_terms: list[str],
) -> list[KnowledgeChunk]:
    chunks: list[KnowledgeChunk] = []
    for document in documents:
        for line in document.text.splitlines():
            stripped = line.strip()
            if (
                stripped.startswith("|")
                and not stripped.startswith("| ---")
                and not stripped.lower().startswith("| term ")
            ):
                chunks.append(
                    KnowledgeChunk(
                        source=document.source,
                        text=stripped,
                        terms=_terms_in_text(stripped, domain_terms),
                    )
                )
        paragraphs = re.split(r"\n\s*\n", document.text)
        for paragraph in paragraphs:
            cleaned = " ".join(
                line.strip()
                for line in paragraph.splitlines()
                if line.strip()
                and not line.strip().startswith(("```", "|"))
                and "->" not in line
            )
            if not cleaned:
                continue
            chunks.append(
                KnowledgeChunk(
                    source=document.source,
                    text=cleaned,
                    terms=_terms_in_text(cleaned, domain_terms),
                )
            )
    return chunks


class TfidfKnowledgeBase:
    """Small in-process TF-IDF index with glossary metadata."""

    def __init__(
        self,
        directory: str | Path = DEFAULT_KNOWLEDGE_BASE,
    ) -> None:
        self.directory = Path(directory)
        self.documents = load_knowledge_base(self.directory)
        glossary_terms: list[str] = []
        correction_pairs: dict[str, str] = {}
        for document in self.documents:
            glossary_terms.extend(_parse_glossary_terms(document.text))
            correction_pairs.update(_parse_correction_pairs(document.text))

        self.domain_terms = list(
            dict.fromkeys(glossary_terms or DEFAULT_DOMAIN_TERMS)
        )
        self.correction_pairs = {
            **DEFAULT_CORRECTION_PAIRS,
            **correction_pairs,
        }
        self.chunks = _build_chunks(self.documents, self.domain_terms)
        self._document_frequency: Counter[str] = Counter()
        for chunk in self.chunks:
            self._document_frequency.update(set(_tokenize(chunk.text)))

    def _idf(self, token: str) -> float:
        document_count = max(1, len(self.chunks))
        frequency = self._document_frequency.get(token, 0)
        return math.log((1 + document_count) / (1 + frequency)) + 1.0

    def _vector(self, text: str) -> dict[str, float]:
        counts = Counter(_tokenize(text))
        if not counts:
            return {}
        total = sum(counts.values())
        return {
            token: (count / total) * self._idf(token)
            for token, count in counts.items()
        }

    @staticmethod
    def _cosine(
        first: dict[str, float],
        second: dict[str, float],
    ) -> float:
        if not first or not second:
            return 0.0
        common = first.keys() & second.keys()
        numerator = sum(first[token] * second[token] for token in common)
        first_norm = math.sqrt(sum(value * value for value in first.values()))
        second_norm = math.sqrt(
            sum(value * value for value in second.values())
        )
        if first_norm == 0 or second_norm == 0:
            return 0.0
        return numerator / (first_norm * second_norm)

    def retrieve(
        self,
        text: str,
        *,
        limit: int = 4,
    ) -> dict[str, Any]:
        """Return ranked domain terms and supporting local chunks."""

        if limit <= 0:
            return {"terms": [], "matches": [], "correction_pairs": {}}

        query_vector = self._vector(text)
        ranked: list[tuple[float, int, KnowledgeChunk]] = []
        for index, chunk in enumerate(self.chunks):
            score = self._cosine(query_vector, self._vector(chunk.text))
            if score > 0:
                ranked.append((score, index, chunk))
        ranked.sort(key=lambda item: (-item[0], item[1]))

        lowered = text.lower()
        matched_pairs = {
            source: target
            for source, target in self.correction_pairs.items()
            if re.search(rf"\b{re.escape(source)}\b", lowered)
        }
        terms: list[str] = list(matched_pairs.values())
        for _score, _index, chunk in ranked:
            if matched_pairs and not chunk.text.startswith("|"):
                continue
            terms.extend(chunk.terms)
        for term in self.domain_terms:
            if term.lower() in lowered:
                terms.append(term)

        matches = [
            {
                "source": chunk.source,
                "score": round(score, 4),
                "text": chunk.text,
            }
            for score, _index, chunk in ranked[:limit]
        ]
        return {
            "terms": list(dict.fromkeys(terms))[:limit],
            "matches": matches,
            "correction_pairs": matched_pairs,
        }


def retrieve_terms(
    text: str,
    *,
    limit: int = 4,
    directory: str | Path = DEFAULT_KNOWLEDGE_BASE,
    index: TfidfKnowledgeBase | None = None,
) -> list[str]:
    """Return domain-term candidates for one transcript segment."""

    knowledge_base = index or TfidfKnowledgeBase(directory)
    return knowledge_base.retrieve(text, limit=limit)["terms"]


def enrich_segments_with_terms(
    segments: list[dict[str, Any]],
    *,
    directory: str | Path = DEFAULT_KNOWLEDGE_BASE,
    limit: int = 4,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Attach retrieved terms and compact provenance to each segment."""

    index = TfidfKnowledgeBase(directory)
    enriched: list[dict[str, Any]] = []
    for segment in segments:
        retrieval = index.retrieve(str(segment["raw_text"]), limit=limit)
        updated = dict(segment)
        updated["retrieved_terms"] = retrieval["terms"]
        updated["retrieval_sources"] = sorted(
            {match["source"] for match in retrieval["matches"]}
        )
        updated["retrieval_matches"] = retrieval["matches"]
        enriched.append(updated)
    metadata = {
        "mode": "local_tfidf",
        "knowledge_base_dir": str(index.directory),
        "documents": [document.source for document in index.documents],
        "chunk_count": len(index.chunks),
        "domain_term_count": len(index.domain_terms),
    }
    return enriched, metadata
