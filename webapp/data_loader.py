"""Read-only artifact loading for the AI Meeting Detective frontend."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
CONVERSATION_MAP_DIR = ROOT_DIR / "outputs" / "conversation_maps"
RESULTS_DIR = ROOT_DIR / "experiments" / "results"
CHART_DIR = ROOT_DIR / "assets" / "result_charts"

ASR_SUMMARY_PATH = RESULTS_DIR / "asr_benchmark_summary_real.csv"
WORKFLOW_ABLATION_PATH = RESULTS_DIR / "workflow_ablation_real.csv"
SPEAKER_OVERLAP_PATH = RESULTS_DIR / "speaker_overlap_baseline_real.csv"
TERM_RESCUE_SUMMARY_PATH = RESULTS_DIR / "term_rescue_summary_controlled.csv"
TERM_RESCUE_RESULTS_PATH = RESULTS_DIR / "term_rescue_controlled.csv"
OVERLAP_SAFETY_SUMMARY_PATH = RESULTS_DIR / "overlap_safety_summary_controlled.csv"
OVERLAP_SAFETY_RESULTS_PATH = RESULTS_DIR / "overlap_safety_controlled.csv"

STOPWORDS = {
    "about",
    "after",
    "again",
    "also",
    "and",
    "are",
    "because",
    "been",
    "before",
    "but",
    "can",
    "for",
    "from",
    "have",
    "into",
    "just",
    "know",
    "like",
    "our",
    "that",
    "the",
    "their",
    "then",
    "there",
    "they",
    "this",
    "what",
    "when",
    "where",
    "which",
    "who",
    "will",
    "with",
    "would",
    "you",
    "your",
}


def _empty_frame(path: Path, message: str) -> pd.DataFrame:
    frame = pd.DataFrame()
    frame.attrs["warning"] = message
    frame.attrs["source_path"] = str(path)
    return frame


def _load_csv(path: str | Path, label: str) -> pd.DataFrame:
    source = Path(path)
    if not source.exists():
        return _empty_frame(
            source,
            f"{label} is not available at {source}. Run the corresponding "
            "experiment before using this view.",
        )
    try:
        frame = pd.read_csv(source)
    except (OSError, pd.errors.ParserError, UnicodeError) as exc:
        return _empty_frame(source, f"Could not read {label}: {exc}")
    frame.attrs["source_path"] = str(source)
    if frame.empty:
        frame.attrs["warning"] = f"{label} exists but contains no rows."
    return frame


def list_available_conversation_maps(
    root: str | Path = CONVERSATION_MAP_DIR,
) -> list[Path]:
    """Return valid ConversationMap JSON paths, ordered for UI selection."""

    directory = Path(root)
    if not directory.exists():
        return []
    paths: list[Path] = []
    for path in directory.rglob("*_conversation_map.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and payload.get("clip_id"):
            paths.append(path.resolve())
    return sorted(paths, key=lambda item: str(item).lower())


def load_conversation_map(path: str | Path | None) -> dict[str, Any]:
    """Load one ConversationMap or return an explicit frontend warning."""

    if path is None:
        return {
            "_warning": (
                "No ConversationMap is available. Run the TalkWeaver workflow "
                "or workflow ablation to generate local evidence artifacts."
            )
        }
    source = Path(path)
    if not source.exists():
        return {"_warning": f"ConversationMap not found: {source}"}
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return {"_warning": f"Could not read ConversationMap {source}: {exc}"}
    if not isinstance(payload, dict) or not payload.get("clip_id"):
        return {"_warning": f"File is not a valid ConversationMap: {source}"}
    payload["_source_path"] = str(source.resolve())
    return payload


def resolve_local_audio_path(
    conversation_map: dict[str, Any],
    root: str | Path = ROOT_DIR,
) -> Path | None:
    """Resolve a local audio artifact without allowing paths outside the repo."""

    audio_value = conversation_map.get("metadata", {}).get("audio_path")
    if not audio_value:
        return None
    repository_root = Path(root).resolve()
    candidate = Path(str(audio_value))
    if not candidate.is_absolute():
        candidate = repository_root / candidate
    try:
        resolved = candidate.resolve()
        resolved.relative_to(repository_root)
    except (OSError, ValueError):
        return None
    return resolved if resolved.is_file() else None


def audio_available(
    conversation_map: dict[str, Any],
    root: str | Path = ROOT_DIR,
) -> bool:
    """Return whether the selected map references a safe local audio file."""

    return resolve_local_audio_path(conversation_map, root) is not None


def _audio_window(
    item: dict[str, Any],
    *,
    padding_seconds: float,
    duration_seconds: float | None = None,
) -> dict[str, float]:
    start = max(0.0, float(item.get("start", 0.0)) - max(0.0, padding_seconds))
    end = max(start, float(item.get("end", start)) + max(0.0, padding_seconds))
    if duration_seconds is not None and duration_seconds >= 0:
        end = min(end, duration_seconds)
        start = min(start, end)
    return {
        "start": round(start, 3),
        "end": round(end, 3),
        "duration": round(max(0.0, end - start), 3),
    }


def get_anchor_audio_window(
    anchor: dict[str, Any],
    padding_seconds: float = 0.5,
    *,
    duration_seconds: float | None = None,
) -> dict[str, float]:
    """Return a padded, non-negative anchor playback window."""

    return _audio_window(
        anchor,
        padding_seconds=padding_seconds,
        duration_seconds=duration_seconds,
    )


def get_event_audio_window(
    event: dict[str, Any],
    padding_seconds: float = 0.5,
    *,
    duration_seconds: float | None = None,
) -> dict[str, float]:
    """Return a padded, non-negative event playback window."""

    return _audio_window(
        event,
        padding_seconds=padding_seconds,
        duration_seconds=duration_seconds,
    )


def related_anchors_for_event(
    conversation_map: dict[str, Any],
    event: dict[str, Any],
) -> list[dict[str, Any]]:
    """Link an event to explicit anchor IDs or intersecting timed anchors."""

    anchors = list(conversation_map.get("anchors", []))
    evidence_ids = set(event.get("evidence_anchor_ids", []))
    if evidence_ids:
        linked = [
            anchor
            for anchor in anchors
            if anchor.get("anchor_id") in evidence_ids
        ]
        if linked:
            return linked
    start = float(event.get("start", 0.0))
    end = float(event.get("end", start))
    return [
        anchor
        for anchor in anchors
        if float(anchor.get("start", 0.0)) < end
        and float(anchor.get("end", 0.0)) > start
    ]


def get_event_investigation_rows(
    conversation_map: dict[str, Any],
) -> list[dict[str, Any]]:
    """Enrich ConversationEvents with linked transcript and review evidence."""

    duration = conversation_map.get("metadata", {}).get("duration_seconds")
    try:
        duration_value = float(duration)
    except (TypeError, ValueError):
        duration_value = None
    rows: list[dict[str, Any]] = []
    for event in conversation_map.get("events", []):
        linked = related_anchors_for_event(conversation_map, event)
        evidence_ids = list(event.get("evidence_anchor_ids", []))
        if not evidence_ids:
            evidence_ids = [
                str(anchor.get("anchor_id"))
                for anchor in linked
                if anchor.get("anchor_id")
            ]
        window = get_event_audio_window(
            event,
            duration_seconds=duration_value,
        )
        rows.append(
            {
                **event,
                "evidence_anchor_ids": evidence_ids,
                "needs_review": any(
                    bool(anchor.get("needs_review")) for anchor in linked
                ),
                "related_raw_text": " ".join(
                    str(anchor.get("raw_text", "")).strip()
                    for anchor in linked
                    if str(anchor.get("raw_text", "")).strip()
                ),
                "related_corrected_text": " ".join(
                    str(anchor.get("corrected_text", "")).strip()
                    for anchor in linked
                    if str(anchor.get("corrected_text", "")).strip()
                ),
                "audio_window_start": window["start"],
                "audio_window_end": window["end"],
            }
        )
    return rows


def _speaker_ids_from_anchors(anchors: list[dict[str, Any]]) -> list[str]:
    speakers = {
        str(speaker)
        for anchor in anchors
        for speaker in (
            anchor.get("speakers")
            or ([anchor.get("speaker")] if anchor.get("speaker") else [])
        )
        if speaker not in {None, "", "UNKNOWN", "OVERLAP"}
    }
    return sorted(speakers)


def _top_anchor_terms(
    anchors: list[dict[str, Any]],
    existing_terms: Iterable[str],
) -> list[str]:
    retrieved = [
        str(term)
        for anchor in anchors
        for term in anchor.get("retrieved_terms", [])
        if str(term).strip()
    ]
    words = re.findall(
        r"[A-Za-z][A-Za-z0-9.-]{2,}",
        " ".join(
            str(anchor.get("corrected_text") or anchor.get("raw_text") or "")
            for anchor in anchors
        ).lower(),
    )
    counts = Counter(word for word in words if word not in STOPWORDS)
    ordered = list(dict.fromkeys([str(term) for term in existing_terms] + retrieved))
    ordered.extend(word for word, _count in counts.most_common() if word not in ordered)
    return ordered[:6]


def build_speaker_evidence_cards(
    conversation_map: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build evidence-linked speaker cards with extractive fallbacks."""

    anchors = list(conversation_map.get("anchors", []))
    existing = {
        str(card.get("speaker")): card
        for card in conversation_map.get("speaker_cards", [])
        if card.get("speaker")
    }
    speakers = sorted(set(existing) | set(_speaker_ids_from_anchors(anchors)))
    cards: list[dict[str, Any]] = []
    for speaker in speakers:
        speaker_anchors = [
            anchor
            for anchor in anchors
            if speaker
            in (
                anchor.get("speakers")
                or ([anchor.get("speaker")] if anchor.get("speaker") else [])
            )
        ]
        source = existing.get(speaker, {})
        evidence_ids = list(
            dict.fromkeys(
                [
                    str(item)
                    for item in source.get("evidence_anchor_ids", [])
                    if item
                ]
                + [
                    str(anchor.get("anchor_id"))
                    for anchor in speaker_anchors
                    if anchor.get("anchor_id")
                ]
            )
        )
        speaking_time = source.get("speaking_time_seconds")
        if speaking_time is None:
            speaking_time = sum(
                max(
                    0.0,
                    float(anchor.get("end", 0.0))
                    - float(anchor.get("start", 0.0)),
                )
                for anchor in speaker_anchors
            )
        raw_quotes = [
            {
                "anchor_id": anchor.get("anchor_id", ""),
                "start": float(anchor.get("start", 0.0)),
                "end": float(anchor.get("end", 0.0)),
                "text": str(anchor.get("raw_text", "")),
            }
            for anchor in speaker_anchors
            if str(anchor.get("raw_text", "")).strip()
        ][:4]
        corrected_quotes = [
            {
                "anchor_id": anchor.get("anchor_id", ""),
                "start": float(anchor.get("start", 0.0)),
                "end": float(anchor.get("end", 0.0)),
                "text": str(
                    anchor.get("corrected_text")
                    or anchor.get("raw_text")
                    or ""
                ),
            }
            for anchor in speaker_anchors
            if str(
                anchor.get("corrected_text") or anchor.get("raw_text") or ""
            ).strip()
        ][:4]
        claims = list(source.get("main_claims", []))
        claim_modes = {str(claim.get("mode", "")) for claim in claims}
        summary_mode = (
            "llm_assisted"
            if any(mode and mode != "extractive_fallback" for mode in claim_modes)
            else "extractive_fallback"
        )
        stance_summary = str(source.get("stance_summary", "")).strip()
        if not stance_summary:
            stance_summary = (
                "Extractive fallback only. No opinion, intent, or personality "
                "is inferred beyond the linked quotes."
            )
        cards.append(
            {
                "speaker_id": speaker,
                "speaking_time_seconds": round(float(speaking_time), 3),
                "num_anchors": len(speaker_anchors),
                "num_overlap_anchors": sum(
                    bool(anchor.get("overlap")) for anchor in speaker_anchors
                ),
                "needs_review_anchors": sum(
                    bool(anchor.get("needs_review")) for anchor in speaker_anchors
                ),
                "top_terms": _top_anchor_terms(
                    speaker_anchors,
                    source.get("top_terms", []),
                ),
                "representative_raw_quotes": raw_quotes,
                "representative_corrected_quotes": corrected_quotes,
                "main_claims": claims,
                "action_items": list(source.get("action_items", [])),
                "stance_summary": stance_summary,
                "summary_mode": summary_mode,
                "evidence_anchor_ids": evidence_ids,
            }
        )
    return cards


def build_clip_detective_summary(
    conversation_map: dict[str, Any],
) -> dict[str, str]:
    """Create a deterministic clip-level summary from existing evidence."""

    metadata = conversation_map.get("metadata", {})
    anchors = list(conversation_map.get("anchors", []))
    events = get_event_investigation_rows(conversation_map)
    speaker_cards = build_speaker_evidence_cards(conversation_map)
    summary_text = str(
        conversation_map.get("summary", {}).get("summary", "")
    ).strip()
    if not summary_text:
        summary_text = " ".join(
            str(anchor.get("corrected_text") or anchor.get("raw_text") or "")
            for anchor in anchors[:3]
        ).strip() or "No transcript summary is available."
    review_count = sum(bool(anchor.get("needs_review")) for anchor in anchors)
    overlap_events = [event for event in events if event.get("type") == "overlap"]
    interruption_events = [
        event for event in events if event.get("type") == "interruption"
    ]
    rejected = sum(
        bool(audit.get("correction_rejected"))
        or bool(audit.get("unsupported_changes"))
        for audit in conversation_map.get("correction_audits", [])
    )
    source_type = metadata.get("source_type", "unknown")
    if source_type == "public_dataset":
        source_label = "Real public-data clip"
    elif metadata.get("is_mock"):
        source_label = "Deterministic mock/demo clip"
    else:
        source_label = str(source_type).replace("_", " ").title()
    speaker_names = ", ".join(
        card["speaker_id"] for card in speaker_cards
    ) or "No named speaker evidence"
    cross_talk = (
        f"{len(overlap_events)} overlap event(s) and "
        f"{len(interruption_events)} interruption candidate(s)."
    )
    return {
        "what_happened": summary_text,
        "who_spoke": speaker_names,
        "where_cross_talk": cross_talk,
        "what_needs_review": (
            f"{review_count} anchor(s) need review."
            if review_count
            else "No anchors are currently flagged for review."
        ),
        "corrections_rejected": (
            f"{rejected} correction audit(s) were rejected or unsupported."
            if rejected
            else "No rejected correction is recorded in this ConversationMap."
        ),
        "evidence_scope": source_label,
    }


def load_asr_summary(path: str | Path = ASR_SUMMARY_PATH) -> pd.DataFrame:
    return _load_csv(path, "ASR benchmark summary")


def load_workflow_ablation(
    path: str | Path = WORKFLOW_ABLATION_PATH,
) -> pd.DataFrame:
    return _load_csv(path, "workflow ablation results")


def load_speaker_overlap_baseline(
    path: str | Path = SPEAKER_OVERLAP_PATH,
) -> pd.DataFrame:
    return _load_csv(path, "speaker/overlap baseline")


def load_term_rescue_summary(
    path: str | Path = TERM_RESCUE_SUMMARY_PATH,
) -> pd.DataFrame:
    return _load_csv(path, "controlled term rescue summary")


def load_term_rescue_results(
    path: str | Path = TERM_RESCUE_RESULTS_PATH,
) -> pd.DataFrame:
    return _load_csv(path, "controlled term rescue results")


def load_term_rescue_cases(
    path: str | Path = TERM_RESCUE_RESULTS_PATH,
) -> pd.DataFrame:
    """Load row-level controlled term rescue case results."""

    return _load_csv(path, "controlled term rescue cases")


def load_overlap_safety_summary(
    path: str | Path = OVERLAP_SAFETY_SUMMARY_PATH,
) -> pd.DataFrame:
    return _load_csv(path, "controlled overlap safety summary")


def load_overlap_safety_results(
    path: str | Path = OVERLAP_SAFETY_RESULTS_PATH,
) -> pd.DataFrame:
    return _load_csv(path, "controlled overlap safety results")


def load_overlap_safety_cases(
    path: str | Path = OVERLAP_SAFETY_RESULTS_PATH,
) -> pd.DataFrame:
    """Load row-level controlled overlap correction case results."""

    return _load_csv(path, "controlled overlap safety cases")


def _nonempty_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().ne("")


def _boolean_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})


def _json_list_is_empty(value: Any) -> bool:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return True
    if isinstance(value, list):
        return not value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return not value.strip()
        return isinstance(parsed, list) and not parsed
    return False


def _pick_first_matching(
    frame: pd.DataFrame,
    phrases: Iterable[str],
) -> list[dict[str, Any]]:
    rows = frame.to_dict("records")
    selected: list[dict[str, Any]] = []
    used_case_ids: set[str] = set()
    for phrase in phrases:
        for row in rows:
            case_id = str(row.get("case_id", ""))
            if case_id in used_case_ids:
                continue
            if phrase in str(row.get("raw_asr_text", "")).lower():
                selected.append(row)
                used_case_ids.add(case_id)
                break
    return selected


def get_best_term_rescue_examples(
    path: str | Path = TERM_RESCUE_RESULTS_PATH,
) -> pd.DataFrame:
    """Return deterministic, context-supported rescue demonstrations."""

    frame = load_term_rescue_cases(path)
    if frame.empty:
        return frame
    preferred = frame[
        frame["variant"].eq("fused_plus_rule_correction")
        & frame["raw_asr_text"].fillna("").ne(frame["corrected_text"].fillna(""))
        & frame["term_f1"].fillna(0).ge(0.99)
    ].copy()
    phrases = (
        "piano note",
        "diary station",
        "rack glossary",
        "report where",
        "temporal anger",
    )
    selected = _pick_first_matching(preferred, phrases)
    result = pd.DataFrame(selected)
    if result.empty:
        result = preferred.sort_values(
            ["term_f1", "text_error_after"],
            ascending=[False, True],
        ).head(5)
    result.attrs.update(frame.attrs)
    return result.reset_index(drop=True)


def get_best_llm_rejection_examples(
    term_path: str | Path = TERM_RESCUE_RESULTS_PATH,
    overlap_path: str | Path = OVERLAP_SAFETY_RESULTS_PATH,
) -> pd.DataFrame:
    """Return strict-LLM outputs rejected or retained for human review."""

    term = load_term_rescue_cases(term_path)
    overlap = load_overlap_safety_cases(overlap_path)
    records: list[dict[str, Any]] = []
    if not term.empty:
        term_rejected = term[
            term["variant"].eq("fused_plus_llm_correction")
            & _boolean_series(term["api_used"])
            & (
                _nonempty_text(term["correction_error"])
                | _boolean_series(term["needs_review"])
            )
        ].copy()
        for row in term_rejected.head(4).to_dict("records"):
            row["source_experiment"] = "controlled_term_rescue"
            row["rejection_reason"] = (
                row.get("correction_error")
                if isinstance(row.get("correction_error"), str)
                and row.get("correction_error", "").strip()
                else row.get("notes", "Strict LLM output requires review.")
            )
            row.setdefault("correction_rejected", bool(row.get("correction_error")))
            records.append(row)
    if not overlap.empty:
        overlap_rejected = overlap[
            overlap["variant"].eq("overlap_aware_llm")
            & _boolean_series(overlap["correction_rejected"])
        ].copy()
        for row in overlap_rejected.head(4).to_dict("records"):
            row["source_experiment"] = "controlled_overlap_safety"
            row["rejection_reason"] = row.get(
                "notes",
                "Overlap-aware policy rejected the correction.",
            )
            records.append(row)
    return pd.DataFrame(records).reset_index(drop=True)


def get_negative_control_examples(
    term_path: str | Path = TERM_RESCUE_RESULTS_PATH,
    overlap_path: str | Path = OVERLAP_SAFETY_RESULTS_PATH,
) -> pd.DataFrame:
    """Return common-word cases that were deliberately not domain-corrected."""

    term = load_term_rescue_cases(term_path)
    overlap = load_overlap_safety_cases(overlap_path)
    records: list[dict[str, Any]] = []
    if not term.empty:
        preferred = term[term["variant"].eq("fused_plus_rule_correction")].copy()
        mask = preferred["expected_terms"].map(_json_list_is_empty)
        mask &= preferred["raw_asr_text"].fillna("").eq(
            preferred["corrected_text"].fillna("")
        )
        for row in preferred[mask].head(6).to_dict("records"):
            row["source_experiment"] = "controlled_term_rescue"
            row["preservation_reason"] = (
                "No supported domain term in context; raw lexical evidence retained."
            )
            records.append(row)
    if not overlap.empty:
        preferred = overlap[
            overlap["variant"].eq("overlap_aware_rule")
            & ~_boolean_series(overlap["correction_allowed"])
            & ~_boolean_series(overlap["correction_rejected"])
            & overlap["raw_asr_text"].fillna("").eq(
                overlap["corrected_text"].fillna("")
            )
        ]
        for row in preferred.head(4).to_dict("records"):
            row["source_experiment"] = "controlled_overlap_safety"
            row["preservation_reason"] = row.get(
                "notes",
                "No evidence-supported correction was applied.",
            )
            records.append(row)
    return pd.DataFrame(records).reset_index(drop=True)


def get_correction_diff_examples(
    path: str | Path = TERM_RESCUE_RESULTS_PATH,
) -> pd.DataFrame:
    """Return accepted controlled corrections with visible before/after text."""

    frame = load_term_rescue_cases(path)
    if frame.empty:
        return frame
    accepted = frame[
        frame["variant"].isin(
            ["fused_plus_rule_correction", "fused_plus_llm_correction"]
        )
        & frame["raw_asr_text"].fillna("").ne(frame["corrected_text"].fillna(""))
        & ~_boolean_series(frame["needs_review"])
        & frame["unsupported_changes"].map(_json_list_is_empty)
    ].copy()
    accepted["error_delta"] = (
        accepted["text_error_before"].fillna(0)
        - accepted["text_error_after"].fillna(0)
    )
    accepted = accepted.sort_values(
        ["error_delta", "term_f1"],
        ascending=[False, False],
    ).drop_duplicates("case_id")
    accepted.attrs.update(frame.attrs)
    return accepted.head(10).reset_index(drop=True)


def load_chart(
    path: str | Path,
    chart_root: str | Path = CHART_DIR,
) -> Path | None:
    """Resolve a chart name or repository-relative chart path."""

    candidate = Path(path)
    if candidate.is_absolute():
        resolved = candidate
    elif len(candidate.parts) > 1:
        resolved = ROOT_DIR / candidate
    else:
        resolved = Path(chart_root) / candidate
    return resolved.resolve() if resolved.is_file() else None


def discover_charts(
    names: Iterable[str] | None = None,
    chart_root: str | Path = CHART_DIR,
) -> dict[str, Path]:
    """Return available chart paths keyed by filename."""

    root = Path(chart_root)
    requested = list(names) if names is not None else [
        path.name
        for pattern in ("*.png", "*.jpg", "*.jpeg")
        for path in root.glob(pattern)
    ] if root.exists() else []
    charts: dict[str, Path] = {}
    for name in requested:
        path = load_chart(name, root)
        if path is not None:
            charts[Path(name).name] = path
    return charts


def frame_warning(frame: pd.DataFrame) -> str:
    """Return a loader warning attached to a DataFrame, if any."""

    return str(frame.attrs.get("warning", ""))


def _demo_score(path: Path, payload: dict[str, Any]) -> tuple[int, int, int]:
    metadata = payload.get("metadata", {})
    dataset = str(metadata.get("dataset_name", "")).lower()
    variant = str(metadata.get("variant", "")).lower()
    relative = str(path).lower()
    score = 0
    if "ami" in dataset:
        score += 80
    if variant == "full_talkweaver":
        score += 60
    if "reference_assisted_real" in relative:
        score += 30
    if metadata.get("is_mock") is False:
        score += 20
    if metadata.get("uses_real_asr_prediction"):
        score += 20
    if metadata.get("diarization_mode") == "reference":
        score += 15
    score += min(len(payload.get("events", [])) * 5, 25)
    score += min(
        sum(bool(anchor.get("overlap")) for anchor in payload.get("anchors", []))
        * 3,
        18,
    )
    return score, len(payload.get("anchors", [])), len(payload.get("events", []))


def get_best_available_demo_clip(
    root: str | Path = CONVERSATION_MAP_DIR,
) -> Path | None:
    """Choose the strongest local investigation artifact without fabricating one."""

    ranked: list[tuple[tuple[int, int, int], Path]] = []
    for path in list_available_conversation_maps(root):
        payload = load_conversation_map(path)
        if "_warning" not in payload:
            ranked.append((_demo_score(path, payload), path))
    return max(ranked, default=(None, None), key=lambda item: item[0])[1]


def conversation_map_label(path: Path, payload: dict[str, Any] | None = None) -> str:
    """Build a concise selector label with clip, dataset, and evidence mode."""

    data = payload or load_conversation_map(path)
    metadata = data.get("metadata", {})
    clip_id = data.get("clip_id", path.stem.replace("_conversation_map", ""))
    dataset = metadata.get("dataset_name", "unknown source")
    variant = metadata.get("variant") or metadata.get("workflow") or "workflow"
    return f"{clip_id} | {dataset} | {variant}"
