"""Glossary, fuzzy, and phonetic-like term candidate retrieval."""

from __future__ import annotations

import json
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
    aliases: tuple[str, ...] = ()
    spoken_forms: tuple[str, ...] = ()
    asr_error_forms: tuple[str, ...] = ()
    language: str = "en"
    category: str = "domain_term"
    allowed_contexts: tuple[str, ...] = ()


@dataclass(frozen=True)
class TermMatch:
    """One retrieval decision with evidence and correction safety state."""

    canonical: str
    matched_form: str
    score: float
    retrieval_method: str
    safe_to_apply: bool
    needs_review: bool
    context_reason: str
    spoken_forms: tuple[str, ...] = ()
    asr_error_forms: tuple[str, ...] = ()


MULTILINGUAL_ENTRIES = (
    GlossaryEntry(
        canonical="speaker diarization",
        spoken_forms=(
            "diarization",
            "说话人分离",
            "diarisation des locuteurs",
        ),
        asr_error_forms=("diary station",),
    ),
    GlossaryEntry(
        canonical="overlapping speech",
        spoken_forms=(
            "cross-speech",
            "重叠语音",
            "parole chevauchante",
        ),
        asr_error_forms=("overlap speech", "cross speech"),
    ),
    GlossaryEntry(
        canonical="temporal anchor",
        spoken_forms=("时间锚点", "ancrage temporel"),
        asr_error_forms=("time anchor",),
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


def _contains_normalized_phrase(text: str, phrase: str) -> bool:
    if not phrase:
        return False
    if any("\u4e00" <= character <= "\u9fff" for character in phrase):
        return phrase in text
    return re.search(
        rf"(?:^|\s){re.escape(phrase)}(?:$|\s)",
        text,
    ) is not None


def load_reference_glossary(path: str | Path) -> list[GlossaryEntry]:
    """Load the controlled JSON glossary without adding project defaults."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_entries = (
        payload.get("terms", payload)
        if isinstance(payload, dict)
        else payload
    )
    if not isinstance(raw_entries, list):
        raise ValueError("Reference term glossary must contain a list.")
    entries: list[GlossaryEntry] = []
    for index, item in enumerate(raw_entries):
        if (
            not isinstance(item, dict)
            or not str(item.get("canonical", "")).strip()
        ):
            raise ValueError(f"Invalid glossary entry at index {index}.")
        entries.append(
            GlossaryEntry(
                canonical=str(item["canonical"]).strip(),
                aliases=tuple(str(value) for value in item.get("aliases", [])),
                spoken_forms=tuple(
                    str(value) for value in item.get("spoken_forms", [])
                ),
                asr_error_forms=tuple(
                    str(value) for value in item.get("asr_error_forms", [])
                ),
                language=str(item.get("language", "en")),
                category=str(item.get("category", "domain_term")),
                allowed_contexts=tuple(
                    str(value) for value in item.get("allowed_contexts", [])
                ),
            )
        )
    return entries


AMBIGUOUS_FORMS = {
    "dear",
    "rack",
    "rag",
    "tag speech",
    "where",
}
NEGATIVE_CONTEXTS = {
    "DER": {"dear friend", "dear team", "letter", "greeting"},
    "RAG": {
        "equipment rack",
        "metal rack",
        "physical rack",
        "server rack",
        "shelf",
    },
    "TagSpeech": {"price tag", "speech tag", "html tag"},
    "WER": {"ask where", "location", "place", "where is", "where we"},
}


def _context_support(
    entry: GlossaryEntry,
    *,
    matched_form: str,
    retrieval_method: str,
    text: str,
    context: str,
) -> tuple[bool, str]:
    combined = _normalize(f"{text} {context}")
    normalized_form = _normalize(matched_form)
    negative_cues = NEGATIVE_CONTEXTS.get(entry.canonical, set())
    if any(_normalize(cue) in combined for cue in negative_cues):
        return False, "Context indicates the common-word meaning."
    if (
        normalized_form not in AMBIGUOUS_FORMS
        and retrieval_method.startswith("exact")
    ):
        return True, "The matched form is sufficiently term-specific."
    allowed = [
        _normalize(value)
        for value in entry.allowed_contexts
        if _normalize(value)
    ]
    if any(value in combined for value in allowed):
        return True, "Domain context supports the ambiguous term form."
    return False, "Ambiguous form lacks supporting domain context."


def _exact_match(
    text: str,
    entry: GlossaryEntry,
) -> tuple[float, str, str] | None:
    normalized = _normalize(text)
    forms = [
        entry.canonical,
        *entry.aliases,
        *entry.spoken_forms,
        *entry.asr_error_forms,
    ]
    for form in forms:
        normalized_form = _normalize(form)
        if _contains_normalized_phrase(normalized, normalized_form):
            method = (
                "exact_error_form"
                if form in entry.asr_error_forms
                else "exact_glossary"
            )
            return 1.0, method, form
    return None


def _window_scores(text: str, form: str) -> Iterable[tuple[str, float]]:
    text_tokens = _normalize(text).split()
    form_tokens = _normalize(form).split()
    if not text_tokens or not form_tokens:
        return []
    scores: list[tuple[str, float]] = []
    widths = {
        max(1, len(form_tokens) - 1),
        len(form_tokens),
        len(form_tokens) + 1,
    }
    for width in widths:
        if width > len(text_tokens):
            continue
        for start in range(len(text_tokens) - width + 1):
            phrase = " ".join(text_tokens[start : start + width])
            scores.append(
                (
                    phrase,
                    SequenceMatcher(
                        None,
                        phrase,
                        " ".join(form_tokens),
                    ).ratio(),
                )
            )
    return scores


def _fuzzy_match(
    text: str,
    entry: GlossaryEntry,
) -> tuple[float, str, str] | None:
    best: tuple[float, str] = (0.0, "")
    forms = [
        entry.canonical,
        *entry.aliases,
        *entry.spoken_forms,
        *entry.asr_error_forms,
    ]
    for form in forms:
        for phrase, score in _window_scores(text, form):
            if score > best[0]:
                best = (score, phrase)
    canonical_compact = _normalize(entry.canonical).replace(" ", "")
    if len(canonical_compact) <= 4 and best[0] < 1.0:
        return None
    if best[0] >= 0.82:
        return round(best[0], 3), "fuzzy", best[1]
    return None


def _phonetic_match(
    text: str,
    entry: GlossaryEntry,
) -> tuple[float, str, str] | None:
    best: tuple[float, str] = (0.0, "")
    forms = [
        entry.canonical,
        *entry.aliases,
        *entry.spoken_forms,
        *entry.asr_error_forms,
    ]
    for form in forms:
        target = _phonetic_like(form)
        if len(target) < 3:
            continue
        for phrase, _score in _window_scores(text, form):
            score = SequenceMatcher(
                None,
                _phonetic_like(phrase),
                target,
            ).ratio()
            if score > best[0]:
                best = (score, phrase)
    if best[0] >= 0.94:
        return round(best[0], 3), "phonetic_like", best[1]
    return None


def retrieve_controlled_matches(
    text: str,
    entries: list[GlossaryEntry],
    *,
    strategy: str,
    context: str = "",
    limit: int = 8,
) -> tuple[list[TermMatch], list[TermMatch]]:
    """Retrieve controlled candidates and separate context-rejected matches."""

    if strategy not in {"exact_glossary", "fuzzy", "phonetic_like", "fused"}:
        raise ValueError(f"Unsupported term retrieval strategy: {strategy}")
    matches: list[TermMatch] = []
    for entry in entries:
        exact = _exact_match(text, entry)
        fuzzy = _fuzzy_match(text, entry)
        phonetic = _phonetic_match(text, entry)
        selected: tuple[float, str, str] | None
        if strategy == "exact_glossary":
            selected = exact
        elif strategy == "fuzzy":
            selected = fuzzy
        elif strategy == "phonetic_like":
            selected = phonetic
        else:
            options = [value for value in (exact, fuzzy, phonetic) if value]
            selected = max(options, key=lambda value: value[0]) if options else None
        if selected is None:
            continue
        score, method, matched_form = selected
        if (
            strategy == "fused"
            and method in {"fuzzy", "phonetic_like"}
            and score < 0.88
        ):
            continue
        supported, reason = _context_support(
            entry,
            matched_form=matched_form,
            retrieval_method=method,
            text=text,
            context=context,
        )
        matches.append(
            TermMatch(
                canonical=entry.canonical,
                matched_form=matched_form,
                score=score,
                retrieval_method=method,
                safe_to_apply=supported,
                needs_review=not supported,
                context_reason=reason,
                spoken_forms=entry.spoken_forms,
                asr_error_forms=entry.asr_error_forms,
            )
        )
    matches.sort(key=lambda item: (-item.score, item.canonical.casefold()))
    if strategy == "fused":
        accepted = [match for match in matches if match.safe_to_apply][:limit]
        rejected = [match for match in matches if not match.safe_to_apply][:limit]
        return accepted, rejected
    return matches[:limit], []


def matches_to_candidates(
    matches: list[TermMatch],
    *,
    anchor_id: str,
) -> list[TermRescueCandidate]:
    """Convert controlled matches into the shared workflow schema."""

    return [
        TermRescueCandidate(
            term_id=f"controlled_term_{index:03d}",
            canonical=match.canonical,
            spoken_forms=list(match.spoken_forms),
            asr_error_forms=list(match.asr_error_forms),
            retrieved_score=match.score,
            retrieval_method=match.retrieval_method,
            evidence_anchor_ids=[anchor_id],
        )
        for index, match in enumerate(matches, start=1)
    ]


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
        if _contains_normalized_phrase(normalized, normalized_form):
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
