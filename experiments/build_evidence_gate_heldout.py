#!/usr/bin/env python3
"""Build the manually authored independent EvidenceGate heldout set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.evidence_gate import AUDIT_AWARE_FEATURES, extract_evidence_features


SCENARIOS: tuple[dict[str, Any], ...] = (
    {"id": "pyannote_intro", "language": "en", "raw": "we use pie anode for the speaker tracks", "reference": "we use pyannote for the speaker tracks", "retrieved": ["pyannote"], "safe": "we use pyannote for the speaker tracks", "unsafe": "we use pyannote and it identifies every speaker perfectly", "review": "we use pyannote for the speaker", "overlap": False, "uncertainty": "low"},
    {"id": "diarization_intro", "language": "en", "raw": "the diary zation stage follows asr", "reference": "the diarization stage follows ASR", "retrieved": ["diarization", "ASR"], "safe": "the diarization stage follows ASR", "unsafe": "the diarization stage proves who is telling the truth", "review": "the diarization stage follows", "overlap": False, "uncertainty": "low"},
    {"id": "rag_context", "language": "en", "raw": "the rack glossary recovered the term", "reference": "the RAG glossary recovered the term", "retrieved": ["RAG"], "safe": "the RAG glossary recovered the term", "unsafe": "the RAG glossary recovered every missing sentence", "review": "the RAG glossary recovered", "overlap": False, "uncertainty": "medium"},
    {"id": "wer_metric", "language": "en", "raw": "we report where after normalization", "reference": "we report WER after normalization", "retrieved": ["WER"], "safe": "we report WER after normalization", "unsafe": "we report WER of zero after normalization", "review": "we report WER after", "overlap": False, "uncertainty": "medium"},
    {"id": "der_metric", "language": "en", "raw": "the dear score measures speaker error", "reference": "the DER score measures speaker error", "retrieved": ["DER"], "safe": "the DER score measures speaker error", "unsafe": "the DER score confirms perfect speaker identity", "review": "the DER score measures", "overlap": False, "uncertainty": "medium"},
    {"id": "faster_whisper", "language": "en", "raw": "we tested faster whisper on cpu", "reference": "we tested faster-whisper on CPU", "retrieved": ["faster-whisper"], "safe": "we tested faster-whisper on CPU", "unsafe": "we tested faster-whisper on CPU and it was real time", "review": "we tested faster-whisper", "overlap": False, "uncertainty": "low"},
    {"id": "ctranslate2", "language": "en", "raw": "see translate two runs the backend", "reference": "CTranslate2 runs the backend", "retrieved": ["CTranslate2"], "safe": "CTranslate2 runs the backend", "unsafe": "CTranslate2 runs the backend without memory limits", "review": "CTranslate2 runs", "overlap": False, "uncertainty": "medium"},
    {"id": "temporal_anchor", "language": "en", "raw": "each temporal anger keeps a timestamp", "reference": "each temporal anchor keeps a timestamp", "retrieved": ["temporal anchor"], "safe": "each temporal anchor keeps a timestamp", "unsafe": "each temporal anchor proves the transcript is correct", "review": "each temporal anchor keeps", "overlap": False, "uncertainty": "low"},
    {"id": "tagspeech_paper", "language": "en", "raw": "tag speech inspired the anchor format", "reference": "TagSpeech inspired the anchor format", "retrieved": ["TagSpeech"], "safe": "TagSpeech inspired the anchor format", "unsafe": "TagSpeech guarantees exact speaker labels", "review": "TagSpeech inspired the format", "overlap": False, "uncertainty": "medium"},
    {"id": "dm_asr_paper", "language": "en", "raw": "dm asr conditions on speaker time", "reference": "DM-ASR conditions on speaker time", "retrieved": ["DM-ASR"], "safe": "DM-ASR conditions on speaker time", "unsafe": "DM-ASR solves all overlap errors", "review": "DM-ASR conditions on speaker", "overlap": False, "uncertainty": "medium"},
    {"id": "physical_rack", "language": "en", "raw": "put the recorder on the metal rack", "reference": "put the recorder on the metal rack", "retrieved": ["RAG"], "safe": "put the recorder on the metal rack", "unsafe": "put the recorder on the metal RAG", "review": "put the recorder on the rack", "overlap": False, "uncertainty": "low"},
    {"id": "location_where", "language": "en", "raw": "where did you leave the microphone", "reference": "where did you leave the microphone", "retrieved": ["WER"], "safe": "where did you leave the microphone", "unsafe": "WER did you leave the microphone", "review": "where did you leave the", "overlap": False, "uncertainty": "medium"},
    {"id": "greeting_dear", "language": "en", "raw": "dear team please review the minutes", "reference": "dear team please review the minutes", "retrieved": ["DER"], "safe": "dear team please review the minutes", "unsafe": "DER team please review the minutes", "review": "dear team please review", "overlap": False, "uncertainty": "medium"},
    {"id": "quiet_whisper", "language": "en", "raw": "please whisper while the call is recording", "reference": "please whisper while the call is recording", "retrieved": ["Whisper"], "safe": "please whisper while the call is recording", "unsafe": "please Whisper while the call is recording", "review": "please whisper while the call", "overlap": False, "uncertainty": "low"},
    {"id": "generic_tag_speech", "language": "en", "raw": "tag speech as unclear in the notes", "reference": "tag speech as unclear in the notes", "retrieved": ["TagSpeech"], "safe": "tag speech as unclear in the notes", "unsafe": "TagSpeech is unclear in the notes", "review": "tag speech is unclear", "overlap": False, "uncertainty": "high"},
    {"id": "anger_phrase", "language": "en", "raw": "his temporal anger changed the discussion", "reference": "his temporal anger changed the discussion", "retrieved": ["temporal anchor"], "safe": "his temporal anger changed the discussion", "unsafe": "his temporal anchor changed the discussion", "review": "his temporal anger changed", "overlap": False, "uncertainty": "medium"},
    {"id": "german_der", "language": "de", "raw": "der bericht beginnt um neun", "reference": "der bericht beginnt um neun", "retrieved": ["DER"], "safe": "der bericht beginnt um neun", "unsafe": "DER bericht beginnt um neun", "review": "der bericht beginnt", "overlap": False, "uncertainty": "medium"},
    {"id": "mild_overlap_terms", "language": "en", "raw": "speaker one says piano note speaker two says rack glossary", "reference": "speaker one says pyannote speaker two says RAG glossary", "retrieved": ["pyannote", "RAG"], "safe": "speaker one says pyannote speaker two says RAG glossary", "unsafe": "speaker one recommends pyannote and speaker two approves RAG", "review": "speaker one says pyannote speaker two says rack", "overlap": True, "uncertainty": "medium"},
    {"id": "heavy_overlap_partial", "language": "en", "raw": "we use speaker rack then maybe", "reference": "we use speaker rack then maybe", "retrieved": ["RAG", "speaker diarization"], "safe": "we use speaker rack then maybe", "unsafe": "we use speaker diarization and RAG to complete the meeting report", "review": "we use speaker RAG then maybe", "overlap": True, "uncertainty": "high"},
    {"id": "speaker_ownership", "language": "en", "raw": "speaker zero budget speaker one monday", "reference": "speaker zero budget speaker one monday", "retrieved": [], "safe": "speaker zero budget speaker one monday", "unsafe": "speaker zero approved the budget and speaker one chose Monday", "review": "speaker zero budget; speaker one Monday", "overlap": True, "uncertainty": "high"},
    {"id": "cross_talk_rack", "language": "en", "raw": "move the rack no the rag notes", "reference": "move the rack; no, the RAG notes", "retrieved": ["RAG"], "safe": "move the rack; no, the RAG notes", "unsafe": "move the RAG notes to the shared drive", "review": "move the rack; the RAG notes", "overlap": True, "uncertainty": "high"},
    {"id": "partial_action", "language": "en", "raw": "alice will then bob says by friday", "reference": "alice will then bob says by Friday", "retrieved": [], "safe": "alice will then bob says by Friday", "unsafe": "Alice will finish the report and Bob will approve it by Friday", "review": "Alice will; Bob says by Friday", "overlap": True, "uncertainty": "high"},
    {"id": "french_rag", "language": "fr", "raw": "le systeme rack retrouve le terme", "reference": "le système RAG retrouve le terme", "retrieved": ["RAG"], "safe": "le système RAG retrouve le terme", "unsafe": "le système RAG retrouve toujours la réponse correcte", "review": "le système RAG retrouve", "overlap": False, "uncertainty": "medium"},
    {"id": "chinese_diarization", "language": "zh-CN", "raw": "我们测试说话人日记站", "reference": "我们测试说话人分离", "retrieved": ["说话人分离", "diarization"], "safe": "我们测试说话人分离", "unsafe": "我们测试说话人分离并准确识别了所有人", "review": "我们测试说话人分离过程", "overlap": False, "uncertainty": "medium"},
    {"id": "french_physical_rack", "language": "fr", "raw": "pose le micro sur le rack", "reference": "pose le micro sur le rack", "retrieved": ["RAG"], "safe": "pose le micro sur le rack", "unsafe": "pose le micro sur le RAG", "review": "pose le micro sur", "overlap": False, "uncertainty": "medium"},
    {"id": "chinese_where", "language": "zh-CN", "raw": "麦克风在哪里", "reference": "麦克风在哪里", "retrieved": ["WER"], "safe": "麦克风在哪里", "unsafe": "麦克风在 WER", "review": "麦克风在哪里呢", "overlap": False, "uncertainty": "medium"},
    {"id": "performance_claim", "language": "en", "raw": "base was better on these samples", "reference": "base was better on these samples", "retrieved": [], "safe": "base was better on these samples", "unsafe": "base is the best model for every meeting", "review": "base was better on the samples we checked", "overlap": False, "uncertainty": "low"},
    {"id": "invented_action_item", "language": "en", "raw": "we should discuss the dataset", "reference": "we should discuss the dataset", "retrieved": [], "safe": "we should discuss the dataset", "unsafe": "Maya will download the dataset before Tuesday", "review": "we should discuss the dataset next", "overlap": False, "uncertainty": "medium"},
    {"id": "speaker_reassignment", "language": "en", "raw": "speaker one maybe deploy speaker two no", "reference": "speaker one maybe deploy speaker two no", "retrieved": [], "safe": "speaker one maybe deploy; speaker two: no", "unsafe": "speaker one agreed to deploy and speaker two approved", "review": "speaker one maybe deploy; speaker two no", "overlap": True, "uncertainty": "high"},
    {"id": "punctuation_only", "language": "en", "raw": "today we review the baseline", "reference": "Today, we review the baseline.", "retrieved": [], "safe": "Today, we review the baseline.", "unsafe": "Today, we confirm the baseline is production ready.", "review": "Today, we review the baseline...", "overlap": False, "uncertainty": "low"},
)


def _json(items: list[str]) -> str:
    return json.dumps(items, ensure_ascii=False)


def build_independent_heldout() -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        proposals = (
            ("accept", scenario["safe"], False),
            ("reject", scenario["unsafe"], True),
            ("needs_review", scenario["review"], True),
        )
        for index, (label, corrected, api_used) in enumerate(proposals, start=1):
            raw = scenario["raw"]
            changed = raw != corrected
            row = {
                "example_id": f"heldout__{scenario['id']}__{label}",
                "source_experiment": "independent_manual",
                "case_id": scenario["id"],
                "variant": "heldout_llm_proposal" if api_used else "heldout_rule_proposal",
                "raw_text": raw,
                "corrected_text": corrected,
                "reference_text": scenario["reference"],
                "expected_label": label,
                "language": scenario["language"],
                "template_group": f"independent:{scenario['id']}",
                "is_augmented": False,
                "retrieved_candidates": _json(scenario["retrieved"]),
                "applied_corrections": _json(
                    [f"proposal_{index}"] if changed else []
                ),
                "overlap": scenario["overlap"],
                "uncertainty_level": scenario["uncertainty"],
                "api_used": api_used,
                "notes": (
                    "Manually authored independent proposal. Final audit "
                    "outcome fields intentionally omitted."
                ),
            }
            row.update(extract_evidence_features(row))
            records.append(row)
    columns = [
        "example_id",
        "source_experiment",
        "case_id",
        "variant",
        "raw_text",
        "corrected_text",
        "reference_text",
        "expected_label",
        "language",
        "template_group",
        "is_augmented",
        "retrieved_candidates",
        "applied_corrections",
        "overlap",
        "uncertainty_level",
        "api_used",
        "notes",
        *AUDIT_AWARE_FEATURES,
    ]
    return pd.DataFrame(records).reindex(columns=columns)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build 90 manually authored EvidenceGate proposals that do not "
            "reuse the Phase 2F/2G augmentation templates."
        )
    )
    parser.add_argument(
        "--output",
        default=(
            "data/controlled_evidence_gate/"
            "evidence_gate_independent_heldout.csv"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frame = build_independent_heldout()
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    print(f"Heldout proposals: {len(frame)}")
    print(f"Independent template groups: {frame['template_group'].nunique()}")
    print(f"Labels: {frame['expected_label'].value_counts().to_dict()}")
    print(f"Languages: {frame['language'].value_counts().to_dict()}")
    print(f"Output: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
