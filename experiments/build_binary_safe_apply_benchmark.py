#!/usr/bin/env python3
"""Build the reference-derived binary safe-to-apply benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.metrics.text_metrics import evaluate_text


COLUMNS = [
    "proposal_id",
    "source",
    "category",
    "language",
    "raw_asr_text",
    "proposed_corrected_text",
    "reference_text",
    "context",
    "retrieved_terms",
    "overlap_flag",
    "heavy_overlap_flag",
    "speaker_ambiguity_flag",
    "partial_utterance_flag",
    "error_before",
    "error_after",
    "error_delta",
    "binary_label",
    "label_source",
    "label_rule",
    "notes",
]
UNSAFE_FIELDS = (
    "unsupported_changes",
    "invented_content",
    "speaker_attribution_changed",
    "forbidden_change_count",
)
INCOMPLETE_MARKERS = ("...", "[inaudible]", "[cut off]", "--")


def _clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return _clean(value).casefold() in {"1", "true", "yes", "y"}


def _as_float(value: Any) -> float | None:
    text = _clean(value)
    if not text:
        return None
    try:
        result = float(text)
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def _json_list(value: Any) -> list[str]:
    text = _clean(value)
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return [part.strip() for part in text.split(",") if part.strip()]
    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if str(item).strip()]
    return [str(parsed).strip()] if str(parsed).strip() else []


def derive_binary_label(
    error_before: float,
    error_after: float,
    *,
    margin: float,
    unsafe_override: bool = False,
) -> tuple[str, str]:
    """Apply the explicit reference-improvement labeling policy."""

    if margin < 0:
        raise ValueError("Improvement margin must be non-negative.")
    if unsafe_override:
        return (
            "do_not_apply",
            "Safety override: unsupported, invented, forbidden, or "
            "speaker-attribution-changing content was detected.",
        )
    if error_after + margin < error_before:
        return (
            "safe_to_apply",
            f"error_after + {margin:.3f} < error_before",
        )
    return (
        "do_not_apply",
        f"error_after + {margin:.3f} >= error_before",
    )


def _load_jsonl(path: str | Path) -> dict[str, dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        return {}
    rows = {}
    for line in source.read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            rows[str(payload["case_id"])] = payload
    return rows


def _unsafe_override(row: dict[str, Any]) -> bool:
    if _json_list(row.get("unsupported_changes")):
        return True
    if _as_bool(row.get("invented_content")):
        return True
    if _as_bool(row.get("speaker_attribution_changed")):
        return True
    return (_as_float(row.get("forbidden_change_count")) or 0.0) > 0


def _score_if_needed(
    row: dict[str, Any],
    language: str,
) -> tuple[float, float]:
    before = _as_float(row.get("text_error_before"))
    after = _as_float(row.get("text_error_after"))
    if before is not None and after is not None:
        return before, after
    reference = _clean(row.get("reference_text"))
    raw = _clean(row.get("raw_asr_text"))
    corrected = _clean(
        row.get("corrected_text")
        or row.get("proposed_corrected_text")
    )
    if not reference:
        raise ValueError("Reference-derived row is missing reference text.")
    return (
        float(evaluate_text(reference, raw, language)["error_rate"]),
        float(evaluate_text(reference, corrected, language)["error_rate"]),
    )


def _term_rows(
    path: str | Path,
    fixtures: dict[str, dict[str, Any]],
    margin: float,
) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    rows = []
    for source_row in source_rows:
        case_id = _clean(source_row.get("case_id"))
        fixture = fixtures.get(case_id, {})
        language = _clean(source_row.get("language")) or "en"
        before, after = _score_if_needed(source_row, language)
        label, rule = derive_binary_label(
            before,
            after,
            margin=margin,
            unsafe_override=_unsafe_override(source_row),
        )
        expected_terms = _json_list(source_row.get("expected_terms"))
        category = (
            "ordinary_word_negative_control"
            if not expected_terms
            else "technical_term_recovery"
        )
        variant = _clean(source_row.get("variant"))
        rows.append(
            {
                "proposal_id": f"term::{case_id}::{variant}",
                "source": "term_rescue_controlled",
                "category": category,
                "language": language,
                "raw_asr_text": _clean(source_row.get("raw_asr_text")),
                "proposed_corrected_text": _clean(
                    source_row.get("corrected_text")
                ),
                "reference_text": _clean(source_row.get("reference_text")),
                "context": _clean(fixture.get("context")),
                "retrieved_terms": json.dumps(
                    _json_list(source_row.get("retrieved_candidates")),
                    ensure_ascii=False,
                ),
                "overlap_flag": False,
                "heavy_overlap_flag": False,
                "speaker_ambiguity_flag": False,
                "partial_utterance_flag": False,
                "error_before": round(before, 6),
                "error_after": round(after, 6),
                "error_delta": round(before - after, 6),
                "binary_label": label,
                "label_source": "controlled_reference",
                "label_rule": rule,
                "notes": (
                    f"Controlled term fixture; variant={variant}; "
                    "not measured real-audio ASR."
                ),
            }
        )
    return rows


def _glossary_terms(path: str | Path) -> list[str]:
    source = Path(path)
    if not source.is_file():
        return []
    payload = json.loads(source.read_text(encoding="utf-8"))
    entries = payload.get("terms", payload) if isinstance(payload, dict) else payload
    if not isinstance(entries, list):
        return []
    return [
        _clean(entry.get("canonical"))
        for entry in entries
        if isinstance(entry, dict) and _clean(entry.get("canonical"))
    ]


def _terms_in_proposal(
    corrected_text: str,
    raw_text: str,
    glossary: Iterable[str],
) -> list[str]:
    corrected = corrected_text.casefold()
    raw = raw_text.casefold()
    return [
        term
        for term in glossary
        if term.casefold() in corrected and term.casefold() not in raw
    ]


def _overlap_rows(
    path: str | Path,
    fixtures: dict[str, dict[str, Any]],
    glossary: list[str],
    margin: float,
) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    rows = []
    for source_row in source_rows:
        case_id = _clean(source_row.get("case_id"))
        fixture = fixtures.get(case_id, {})
        language = _clean(source_row.get("language")) or "en"
        raw = _clean(source_row.get("raw_asr_text"))
        corrected = _clean(source_row.get("corrected_text"))
        before, after = _score_if_needed(source_row, language)
        label, rule = derive_binary_label(
            before,
            after,
            margin=margin,
            unsafe_override=_unsafe_override(source_row),
        )
        overlap = _as_bool(fixture.get("overlap", source_row.get("overlap")))
        uncertainty = _clean(
            fixture.get(
                "uncertainty_level",
                source_row.get("uncertainty_level"),
            )
        ).casefold()
        speakers = fixture.get("speakers", [])
        context = _clean(fixture.get("context"))
        partial = any(marker in raw for marker in INCOMPLETE_MARKERS)
        speaker_ambiguity = (
            "ambiguous" in context.casefold()
            or "unknown" in context.casefold()
            or (
                overlap
                and isinstance(speakers, list)
                and len(speakers) > 1
                and uncertainty == "high"
            )
        )
        variant = _clean(source_row.get("variant"))
        rows.append(
            {
                "proposal_id": f"overlap::{case_id}::{variant}",
                "source": "overlap_safety_controlled",
                "category": (
                    "heavy_overlap"
                    if overlap and uncertainty == "high"
                    else "overlap_correction"
                    if overlap
                    else "single_speaker_safety"
                ),
                "language": language,
                "raw_asr_text": raw,
                "proposed_corrected_text": corrected,
                "reference_text": _clean(source_row.get("reference_text")),
                "context": context,
                "retrieved_terms": json.dumps(
                    _terms_in_proposal(corrected, raw, glossary),
                    ensure_ascii=False,
                ),
                "overlap_flag": overlap,
                "heavy_overlap_flag": overlap and uncertainty == "high",
                "speaker_ambiguity_flag": speaker_ambiguity,
                "partial_utterance_flag": partial,
                "error_before": round(before, 6),
                "error_after": round(after, 6),
                "error_delta": round(before - after, 6),
                "binary_label": label,
                "label_source": "controlled_reference",
                "label_rule": rule,
                "notes": (
                    f"Controlled overlap fixture; variant={variant}; "
                    "not measured real-audio ASR."
                ),
            }
        )
    return rows


def _pilot_rows(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    rows = []
    for source_row in source_rows:
        suggested = _clean(
            source_row.get("suggested_gold_label")
        ).casefold()
        label = (
            "safe_to_apply" if suggested == "accept" else "do_not_apply"
        )
        rows.append(
            {
                "proposal_id": f"pilot::{source_row['proposal_id']}",
                "source": "r0_selective_pilot",
                "category": _clean(source_row.get("category")),
                "language": _clean(source_row.get("language")) or "en",
                "raw_asr_text": _clean(source_row.get("raw_asr_text")),
                "proposed_corrected_text": _clean(
                    source_row.get("proposed_corrected_text")
                ),
                "reference_text": "",
                "context": _clean(source_row.get("context")),
                "retrieved_terms": _clean(source_row.get("retrieved_terms"))
                or "[]",
                "overlap_flag": _as_bool(
                    source_row.get("overlap_flag")
                ),
                "heavy_overlap_flag": _as_bool(
                    source_row.get("heavy_overlap_flag")
                ),
                "speaker_ambiguity_flag": _as_bool(
                    source_row.get("speaker_ambiguity_flag")
                ),
                "partial_utterance_flag": _as_bool(
                    source_row.get("partial_utterance_flag")
                ),
                "error_before": "",
                "error_after": "",
                "error_delta": "",
                "binary_label": label,
                "label_source": "pilot_suggested_if_no_reference",
                "label_rule": (
                    f"R0 suggested label {suggested!r} mapped to binary; "
                    "accept -> safe_to_apply, all others -> do_not_apply."
                ),
                "notes": (
                    "No reference transcript; retained only as a secondary "
                    "pilot-suggested slice."
                ),
            }
        )
    return rows


def _prediction_rows(
    directory: str | Path | None,
    margin: float,
) -> list[dict[str, Any]]:
    if not directory or not Path(directory).is_dir():
        return []
    rows = []
    for path in sorted(Path(directory).glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw = _clean(
            payload.get("raw_asr_text")
            or payload.get("hypothesis_text")
            or payload.get("hypothesis")
        )
        corrected = _clean(
            payload.get("proposed_corrected_text")
            or payload.get("corrected_text")
        )
        reference = _clean(
            payload.get("reference_text") or payload.get("reference")
        )
        if not (raw and corrected and reference):
            continue
        language = _clean(payload.get("language")) or "en"
        before = float(evaluate_text(reference, raw, language)["error_rate"])
        after = float(
            evaluate_text(reference, corrected, language)["error_rate"]
        )
        label, rule = derive_binary_label(
            before,
            after,
            margin=margin,
        )
        rows.append(
            {
                "proposal_id": f"prediction::{path.stem}",
                "source": "asr_prediction_with_correction",
                "category": "real_prediction_correction",
                "language": language,
                "raw_asr_text": raw,
                "proposed_corrected_text": corrected,
                "reference_text": reference,
                "context": _clean(payload.get("context")),
                "retrieved_terms": json.dumps(
                    payload.get("retrieved_terms", []),
                    ensure_ascii=False,
                ),
                "overlap_flag": _as_bool(payload.get("overlap")),
                "heavy_overlap_flag": _as_bool(
                    payload.get("heavy_overlap")
                ),
                "speaker_ambiguity_flag": _as_bool(
                    payload.get("speaker_ambiguity")
                ),
                "partial_utterance_flag": _as_bool(
                    payload.get("partial_utterance")
                ),
                "error_before": round(before, 6),
                "error_after": round(after, 6),
                "error_delta": round(before - after, 6),
                "binary_label": label,
                "label_source": "reference_derived",
                "label_rule": rule,
                "notes": f"Loaded from correction-bearing prediction {path}.",
            }
        )
    return rows


def build_binary_benchmark(
    *,
    term_input: str | Path,
    overlap_input: str | Path,
    pilot_input: str | Path,
    margin: float,
    predictions_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Build all eligible binary correction proposals."""

    rows = []
    rows.extend(
        _term_rows(
            term_input,
            _load_jsonl("data/controlled_terms/term_rescue_cases.jsonl"),
            margin,
        )
    )
    rows.extend(
        _overlap_rows(
            overlap_input,
            _load_jsonl(
                "data/controlled_overlap/overlap_correction_cases.jsonl"
            ),
            _glossary_terms("data/controlled_terms/reference_terms.json"),
            margin,
        )
    )
    rows.extend(_pilot_rows(pilot_input))
    rows.extend(_prediction_rows(predictions_dir, margin))
    identifiers = [str(row["proposal_id"]) for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Binary benchmark proposal IDs are not unique.")
    return [{column: row.get(column, "") for column in COLUMNS} for row in rows]


def write_binary_benchmark(
    rows: list[dict[str, Any]],
    output_path: str | Path,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=COLUMNS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a binary safe-to-apply correction benchmark."
    )
    parser.add_argument(
        "--term-input",
        default="experiments/results/term_rescue_controlled.csv",
    )
    parser.add_argument(
        "--overlap-input",
        default="experiments/results/overlap_safety_controlled.csv",
    )
    parser.add_argument(
        "--pilot-input",
        default="data/pilot/selective_correction_pilot.csv",
    )
    parser.add_argument(
        "--predictions-dir",
        default="experiments/results/asr_predictions_real",
    )
    parser.add_argument(
        "--output",
        default="data/pilot/binary_safe_apply_benchmark.csv",
    )
    parser.add_argument("--margin", type=float, default=0.01)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = build_binary_benchmark(
        term_input=args.term_input,
        overlap_input=args.overlap_input,
        pilot_input=args.pilot_input,
        margin=args.margin,
        predictions_dir=args.predictions_dir,
    )
    write_binary_benchmark(rows, args.output)
    labels = {
        label: sum(row["binary_label"] == label for row in rows)
        for label in ("safe_to_apply", "do_not_apply")
    }
    sources = {
        source: sum(row["label_source"] == source for row in rows)
        for source in sorted({str(row["label_source"]) for row in rows})
    }
    print(f"Binary proposals: {len(rows)}")
    print(f"Labels: {labels}")
    print(f"Label sources: {sources}")
    print(f"Margin: {args.margin}")
    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
