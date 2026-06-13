#!/usr/bin/env python3
"""Build small-subset ConversationMaps with reference speaker-time evidence."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.asr_prediction import load_asr_prediction_json  # noqa: E402
from backend.config import get_settings  # noqa: E402
from backend.conversation_map import (  # noqa: E402
    build_conversation_map,
    save_conversation_map,
)
from backend.reference_evidence import (  # noqa: E402
    load_reference_evidence,
    resolve_project_path,
)


PREDICTION_DIRECTORIES = (
    ROOT / "experiments" / "results" / "asr_predictions_real",
    ROOT / "experiments" / "results" / "asr_predictions_ami_no_vad_real",
)
MODEL_PREFERENCE = ("base", "tiny")


def load_manifest_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def find_prediction_json(clip_id: str) -> Path | None:
    """Find a standard Phase 2C prediction before diagnostic variants."""

    for directory in PREDICTION_DIRECTORIES:
        for model in MODEL_PREFERENCE:
            candidate = directory / f"{model}__{clip_id}.json"
            if candidate.is_file():
                return candidate
    return None


def _reference_asr(
    row: dict[str, str],
    reference: dict[str, Any],
) -> dict[str, Any]:
    return {
        "mode": "reference",
        "segments": reference["asr_segments"],
        "language": row["language"],
    }


def run_reference_maps(
    *,
    manifest: str | Path,
    dataset: str,
    output_dir: str | Path,
) -> list[Path]:
    """Build reference-diarization maps for matching manifest rows."""

    manifest_path = resolve_project_path(manifest)
    destination = resolve_project_path(output_dir)
    settings = get_settings()
    selected = [
        row
        for row in load_manifest_rows(manifest_path)
        if row.get("dataset_name", "").strip().casefold()
        == dataset.strip().casefold()
    ]
    if not selected:
        raise ValueError(f"No manifest rows matched dataset {dataset!r}.")

    outputs: list[Path] = []
    for row in selected:
        reference = load_reference_evidence(row)
        if not reference["speaker_turns"]:
            print(
                f"Skipping {row['clip_id']}: no reference speaker turns.",
                file=sys.stderr,
            )
            continue
        prediction_path = find_prediction_json(row["clip_id"])
        asr_output = (
            load_asr_prediction_json(prediction_path)
            if prediction_path is not None
            else _reference_asr(row, reference)
        )
        diarization_output = {
            "mode": "reference",
            "turns": reference["speaker_turns"],
            "is_mock": False,
        }
        conversation_map = build_conversation_map(
            {
                **row,
                "clip_id": row["clip_id"],
                "evaluation_scope": "small_subset",
                "asr_prediction_json": (
                    str(prediction_path) if prediction_path else ""
                ),
                "speaker_time_evidence": "reference",
            },
            asr_output,
            diarization_output,
            reference["events"],
            settings.knowledge_base_dir,
            {"use_api": False},
        )
        output = save_conversation_map(conversation_map, destination)
        outputs.append(output)
        print(
            f"{row['clip_id']}: "
            f"asr_mode={conversation_map.metadata['asr_mode']} "
            "diarization_mode=reference "
            f"output={output}"
        )
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        outputs = run_reference_maps(
            manifest=args.manifest,
            dataset=args.dataset,
            output_dir=args.output_dir,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Reference workflow map generation failed: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote {len(outputs)} reference-assisted ConversationMaps.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
