"""Glossary, fuzzy, and phonetic-like term candidate retrieval."""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

from backend.rag import DEFAULT_CORRECTION_PAIRS, DEFAULT_DOMAIN_TERMS
from backend.schemas import TemporalAnchor, TermRescueCandidate


WORD_PATTERN = re.compile(r"[A-Za-z0-9\u4e00-\u9fff]+")


@dataclass(frozen=True)
class GlossaryEntry:
    canonical: str
    spoken_forms: tuple[str, ...] = ()
    asr_error_forms: tuple[str, ...] = ()


MULTILINGUAL_ENTRIES = (
    GlossaryEntry(
        "speaker diarization",
        ("diarization", "说话人分离", "diarisation des locuteurs"),
        ("diary station",),
    ),
    GlossaryEntry(
        "overlapping speech",
        ("cross-speech", "重叠语音", "parole chevauchante"),
        ("overlap speech", "cross speech"),
    ),
    GlossaryEntry(
        "temporal anchor",
        ("时间锚点", "ancrage temporel"),
        ("time anchor",),
    ),
)


def _normalize(text: str) -> str:
    return " ".join(WORD_PATTERN.findall(text.lower()))


def _phonetic_like(text: str) -> str:
    normalized = _normalize(text).replace(" ", "")
    normalized = re.sub(r"[aeiouy]", "", normalized)
    return re.sub(r"(.)\1+", r"\1", normalized)


def _entry_catalog(
    glossary_docs: str | Path | Iterable[str | Path] | None = None,
) -> list[GlossaryEntry]:
    entries: dict[str, GlossaryEntry] = {
        entry.canonical.lower(): entry for entry in MULTILINGUAL_ENTRIES
    }
    for source, target in DEFAULT_CORRECTION_PAIRS.items():
        key = target.lower()
        existing = entries.get(key)
        entries[key] = GlossaryEntry(
            canonical=target,
            spoken_forms=existing.spoken_forms if existing else (),
            asr_error_forms=tuple(
                dict.fromkeys(
                    [
                        *(existing.asr_error_forms if existing else ()),
                        source,
                    ]
                )
            ),
        )
    for term in DEFAULT_DOMAIN_TERMS:
        entries.setdefault(term.lower(), GlossaryEntry(canonical=term))

    paths: list[Path] = []
    if glossary_docs is not None:
        if isinstance(glossary_docs, (str, Path)):
            root = Path(glossary_docs)
            paths = sorted(root.glob("*.md")) if root.is_dir() else [root]
        else:
            paths = [Path(path) for path in glossary_docs]
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if "->" in line:
                source, target = [part.strip() for part in line.split("->", 1)]
                if source and target:
                    key = target.lower()
                    existing = entries.get(key, GlossaryEntry(target))
                    entries[key] = GlossaryEntry(
                        canonical=existing.canonical,
                        spoken_forms=existing.spoken_forms,
                        asr_error_forms=tuple(
                            dict.fromkeys([*existing.asr_error_forms, source])
                        ),
                    )
    return list(entries.values())


def _candidate_match(
    text: str,
    entry: GlossaryEntry,
) -> tuple[float, str] | None:
    normalized = _normalize(text)
    forms = [
        entry.canonical,
        *entry.spoken_forms,
        *entry.asr_error_forms,
    ]
    for form in forms:
        normalized_form = _normalize(form)
        if normalized_form and normalized_form in normalized:
            method = (
                "exact_error_form"
                if form in entry.asr_error_forms
                else "exact_glossary"
            )
            return 1.0, method

    text_tokens = normalized.split()
    best = 0.0
    for form in forms:
        form_tokens = _normalize(form).split()
        if not form_tokens:
            continue
        width = len(form_tokens)
        for start in range(max(1, len(text_tokens) - width + 1)):
            phrase = " ".join(text_tokens[start : start + width])
            best = max(
                best,
                SequenceMatcher(None, phrase, " ".join(form_tokens)).ratio(),
            )
    if best >= 0.82:
        return round(best, 3), "fuzzy"

    text_tokens = normalized.split()
    phonetic_forms = [entry.canonical, *entry.spoken_forms]
    for form in phonetic_forms:
        form_phonetic = _phonetic_like(form)
        form_tokens = _normalize(form).split()
        if not form_phonetic or len(form_phonetic) < 3 or not form_tokens:
            continue
        width = len(form_tokens)
        for window_width in {width, width + 1}:
            if window_width > len(text_tokens):
                continue
            for start in range(len(text_tokens) - window_width + 1):
                window = " ".join(
                    text_tokens[start : start + window_width]
                )
                similarity = SequenceMatcher(
                    None,
                    _phonetic_like(window),
                    form_phonetic,
                ).ratio()
                if similarity >= 0.9:
                    return round(similarity, 3), "phonetic_like"
    return None


def retrieve_term_candidates(
    anchors: list[TemporalAnchor],
    *,
    glossary_docs: str | Path | Iterable[str | Path] | None = None,
    limit_per_anchor: int = 5,
) -> list[TermRescueCandidate]:
    """Retrieve candidates without automatically changing transcript text."""

    entries = _entry_catalog(glossary_docs)
    aggregated: dict[tuple[str, str], TermRescueCandidate] = {}
    for anchor in anchors:
        matches: list[tuple[float, str, GlossaryEntry]] = []
        for entry in entries:
            match = _candidate_match(anchor.raw_text, entry)
            if match:
                score, method = match
                matches.append((score, method, entry))
        matches.sort(key=lambda item: (-item[0], item[2].canonical.lower()))
        selected = matches[:limit_per_anchor]
        anchor.retrieved_terms = list(
            dict.fromkeys(entry.canonical for _, _, entry in selected)
        )
        for score, method, entry in selected:
            key = (entry.canonical.lower(), method)
            candidate = aggregated.get(key)
            if candidate is None:
                candidate = TermRescueCandidate(
                    term_id=f"term_{len(aggregated) + 1:03d}",
                    canonical=entry.canonical,
                    spoken_forms=list(entry.spoken_forms),
                    asr_error_forms=list(entry.asr_error_forms),
                    retrieved_score=score,
                    retrieval_method=method,
                    evidence_anchor_ids=[],
                )
                aggregated[key] = candidate
            candidate.retrieved_score = max(
                candidate.retrieved_score, score
            )
            if anchor.anchor_id not in candidate.evidence_anchor_ids:
                candidate.evidence_anchor_ids.append(anchor.anchor_id)
    return list(aggregated.values())


def candidates_for_anchor(
    anchor_id: str,
    candidates: list[TermRescueCandidate],
) -> list[TermRescueCandidate]:
    return [
        candidate
        for candidate in candidates
        if anchor_id in candidate.evidence_anchor_ids
    ]
