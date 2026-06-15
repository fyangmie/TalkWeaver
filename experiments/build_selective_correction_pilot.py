#!/usr/bin/env python3
"""Build the manually authored selective-correction feasibility pilot."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


COLUMNS = [
    "proposal_id",
    "category",
    "language",
    "raw_asr_text",
    "proposed_corrected_text",
    "context",
    "retrieved_terms",
    "overlap_flag",
    "heavy_overlap_flag",
    "speaker_ambiguity_flag",
    "partial_utterance_flag",
    "suggested_gold_label",
    "suggested_rationale",
    "human_checked_label",
    "human_checked_rationale",
    "notes",
]


def _proposal(
    raw: str,
    corrected: str,
    context: str,
    terms: list[str],
    rationale: str,
    *,
    language: str = "en",
    overlap: bool = False,
    heavy_overlap: bool = False,
    speaker_ambiguity: bool = False,
    partial: bool = False,
) -> dict[str, Any]:
    return {
        "language": language,
        "raw_asr_text": raw,
        "proposed_corrected_text": corrected,
        "context": context,
        "retrieved_terms": terms,
        "overlap_flag": overlap,
        "heavy_overlap_flag": heavy_overlap,
        "speaker_ambiguity_flag": speaker_ambiguity,
        "partial_utterance_flag": partial,
        "suggested_rationale": rationale,
    }


SCENARIO_GROUPS = [
    {
        "category": "technical_term_recovery",
        "cases": {
            "accept": _proposal(
                "we use piano note for speaker segmentation",
                "we use pyannote for speaker segmentation",
                "The team is discussing pyannote speaker diarization.",
                ["pyannote", "speaker diarization"],
                "The retrieved term and diarization context directly support pyannote.",
            ),
            "reject": _proposal(
                "the piano note was too sharp",
                "the pyannote was too sharp",
                "A musician is discussing the pitch of a physical piano.",
                ["pyannote"],
                "The ordinary music phrase is not a pyannote reference.",
            ),
            "needs_review": _proposal(
                "pie anode handled the segments",
                "pyannote handled the segments",
                "The surrounding sentence mentions segments but not speakers.",
                ["pyannote"],
                "The phonetic match is plausible, but the domain context is weak.",
            ),
        },
    },
    {
        "category": "technical_term_recovery",
        "cases": {
            "accept": _proposal(
                "diary station assigns each word to a speaker",
                "diarization assigns each word to a speaker",
                "Speaker-attributed ASR and diarization are being reviewed.",
                ["diarization"],
                "Speaker assignment strongly supports the diarization term.",
            ),
            "reject": _proposal(
                "the diary station is beside the bookstore",
                "the diarization is beside the bookstore",
                "A traveler is describing a shop named Diary Station.",
                ["diarization"],
                "The phrase is a place name, not a speech-processing term.",
            ),
            "needs_review": _proposal(
                "the diary zation stage failed",
                "the diarization stage failed",
                "A generic pipeline stage is mentioned without speaker evidence.",
                ["diarization"],
                "The sound match is strong, but the local context is underspecified.",
            ),
        },
    },
    {
        "category": "technical_term_recovery",
        "cases": {
            "accept": _proposal(
                "the temporal anger keeps the words aligned",
                "the temporal anchor keeps the words aligned",
                "The discussion is about timestamp alignment and temporal anchors.",
                ["temporal anchor"],
                "Alignment context and retrieval directly support temporal anchor.",
            ),
            "reject": _proposal(
                "his temporal anger faded after the meeting",
                "his temporal anchor faded after the meeting",
                "The sentence describes a temporary emotional reaction.",
                ["temporal anchor"],
                "The correction changes an ordinary emotional phrase into jargon.",
            ),
            "needs_review": _proposal(
                "add a temporal anger near the end",
                "add a temporal anchor near the end",
                "The object being added is not identified.",
                ["temporal anchor"],
                "The correction is plausible but lacks timestamp or alignment evidence.",
            ),
        },
    },
    {
        "category": "ordinary_word_negative_control",
        "cases": {
            "accept": _proposal(
                "the rack glossary improves term recovery",
                "the RAG glossary improves term recovery",
                "The ASR correction module retrieves terms from a knowledge base.",
                ["RAG"],
                "Retrieval and glossary context support RAG.",
            ),
            "reject": _proposal(
                "put the microphones on the rack",
                "put the microphones on the RAG",
                "The team is arranging physical equipment in a room.",
                ["RAG"],
                "Rack is a physical object and must not be replaced.",
            ),
            "needs_review": _proposal(
                "the rack step returns more evidence",
                "the RAG step returns more evidence",
                "Evidence is returned, but retrieval is not explicitly stated.",
                ["RAG"],
                "The context hints at retrieval but remains ambiguous.",
            ),
        },
    },
    {
        "category": "ordinary_word_negative_control",
        "cases": {
            "accept": _proposal(
                "where is our main transcription metric",
                "WER is our main transcription metric",
                "The benchmark compares ASR word error rate.",
                ["WER"],
                "ASR metric context unambiguously supports WER.",
            ),
            "reject": _proposal(
                "where did speaker two go",
                "WER did speaker two go",
                "A participant asks about another person's location.",
                ["WER"],
                "Where is a normal question word.",
            ),
            "needs_review": _proposal(
                "where improved after preprocessing",
                "WER improved after preprocessing",
                "A metric may be intended, but ASR is not named nearby.",
                ["WER"],
                "Preprocessing suggests a metric, but the evidence is incomplete.",
            ),
        },
    },
    {
        "category": "ordinary_word_negative_control",
        "cases": {
            "accept": _proposal(
                "the dear score measures speaker mistakes",
                "the DER score measures speaker mistakes",
                "The diarization benchmark reports speaker error metrics.",
                ["DER"],
                "Diarization metric context directly supports DER.",
            ),
            "reject": _proposal(
                "dear Maria thank you for the update",
                "DER Maria thank you for the update",
                "This is the greeting line of an email.",
                ["DER"],
                "Dear is an ordinary greeting, not a metric.",
            ),
            "needs_review": _proposal(
                "the dear number is lower now",
                "the DER number is lower now",
                "A numeric result is discussed without naming diarization.",
                ["DER"],
                "The metric reading is plausible but not sufficiently grounded.",
            ),
        },
    },
    {
        "category": "weak_retrieval_evidence",
        "cases": {
            "accept": _proposal(
                "我们用 q when 生成纠错结果",
                "我们用 Qwen 生成纠错结果",
                "The team compares LLM providers for ASR correction.",
                ["Qwen"],
                "Provider context and the retrieved model name support Qwen.",
                language="zh-CN",
            ),
            "reject": _proposal(
                "请问队列什么时候开始任务",
                "请 Qwen 队列什么时候开始任务",
                "The speaker is discussing a compute job queue.",
                ["Qwen"],
                "The retrieved model is unrelated to the queue question.",
                language="zh-CN",
            ),
            "needs_review": _proposal(
                "q when 返回了一个答案",
                "Qwen 返回了一个答案",
                "No model or API is identified in the neighboring text.",
                ["Qwen"],
                "An answer could come from a model, but retrieval evidence is weak.",
                language="zh-CN",
            ),
        },
    },
    {
        "category": "weak_retrieval_evidence",
        "cases": {
            "accept": _proposal(
                "tag speech uses time grounded labels",
                "TagSpeech uses time grounded labels",
                "The literature review discusses the TagSpeech paper.",
                ["TagSpeech"],
                "Paper context supports the canonical title.",
            ),
            "reject": _proposal(
                "tag speech segments before publishing",
                "TagSpeech segments before publishing",
                "An editor is adding labels to ordinary speech segments.",
                ["TagSpeech"],
                "The verb phrase does not refer to the research method.",
            ),
            "needs_review": _proposal(
                "tag speech was mentioned in the discussion",
                "TagSpeech was mentioned in the discussion",
                "The neighboring passage does not identify a paper or system.",
                ["TagSpeech"],
                "Capitalization may be right, but the evidence is not decisive.",
            ),
        },
    },
    {
        "category": "weak_retrieval_evidence",
        "cases": {
            "accept": _proposal(
                "dm asr conditions on speaker and time",
                "DM-ASR conditions on speaker and time",
                "The paper review covers speaker-aware ASR methods.",
                ["DM-ASR"],
                "Research and speaker-time context support DM-ASR.",
            ),
            "reject": _proposal(
                "dm asr in the private chat",
                "DM-ASR in the private chat",
                "DM means direct message; ASR appears in a separate clause.",
                ["DM-ASR"],
                "The proposal incorrectly merges unrelated abbreviations.",
            ),
            "needs_review": _proposal(
                "dm asr improved the output",
                "DM-ASR improved the output",
                "The method name is retrieved, but no speaker-time detail is present.",
                ["DM-ASR"],
                "The correction is plausible but weakly contextualized.",
            ),
        },
    },
    {
        "category": "heavy_overlap",
        "cases": {
            "accept": _proposal(
                "speaker a says piano note",
                "speaker a says pyannote",
                "Two speakers overlap briefly while discussing diarization.",
                ["pyannote"],
                "A local term correction remains supported despite mild overlap.",
                overlap=True,
            ),
            "reject": _proposal(
                "speaker a says we use... speaker b says no...",
                "speaker a says we use pyannote and speaker b agrees with the plan",
                "Heavy cross-talk obscures both incomplete utterances.",
                ["pyannote"],
                "The proposal invents agreement and completes missing content.",
                overlap=True,
                heavy_overlap=True,
                partial=True,
            ),
            "needs_review": _proposal(
                "speaker a says rack while speaker b says retrieval",
                "speaker a says RAG while speaker b says retrieval",
                "The speakers overlap during a retrieval discussion.",
                ["RAG"],
                "The term is supported, but cross-talk makes the source uncertain.",
                overlap=True,
                heavy_overlap=True,
                speaker_ambiguity=True,
            ),
        },
    },
    {
        "category": "heavy_overlap",
        "cases": {
            "accept": _proposal(
                "during overlap the metric is where",
                "during overlap the metric is WER",
                "Mild overlap occurs in an ASR benchmark discussion.",
                ["WER"],
                "The explicit metric context supports a small local correction.",
                overlap=True,
            ),
            "reject": _proposal(
                "we measured... [inaudible] ...lower",
                "we measured WER and proved the new model is significantly better",
                "Heavy overlap hides the name and conclusion of the measurement.",
                ["WER"],
                "The proposal invents a result and significance claim.",
                overlap=True,
                heavy_overlap=True,
                partial=True,
            ),
            "needs_review": _proposal(
                "speaker one says where and speaker two says error",
                "speaker one says WER and speaker two says error",
                "The ASR metric is discussed during strong cross-talk.",
                ["WER"],
                "The term is plausible, but word ownership is uncertain.",
                overlap=True,
                heavy_overlap=True,
                speaker_ambiguity=True,
            ),
        },
    },
    {
        "category": "heavy_overlap",
        "cases": {
            "accept": _proposal(
                "diary station is the next stage",
                "diarization is the next stage",
                "Brief overlap occurs in a speaker-analysis pipeline review.",
                ["diarization"],
                "The pipeline context supports a local diarization correction.",
                overlap=True,
            ),
            "reject": _proposal(
                "speaker a: diary... speaker b: station...",
                "speaker a: diarization is accurate and speaker b: the test passed",
                "Heavy overlap splits fragments across speakers.",
                ["diarization"],
                "The proposal invents claims for both speakers.",
                overlap=True,
                heavy_overlap=True,
                speaker_ambiguity=True,
                partial=True,
            ),
            "needs_review": _proposal(
                "diary station... next... [overlap]",
                "diarization is next",
                "The intended pipeline stage is likely but the utterance is incomplete.",
                ["diarization"],
                "The term is supported, but sentence completion should be reviewed.",
                overlap=True,
                heavy_overlap=True,
                partial=True,
            ),
        },
    },
    {
        "category": "partial_utterance",
        "cases": {
            "accept": _proposal(
                "the tool is faster whisper...",
                "the tool is faster-whisper...",
                "The ASR implementation is named before the speaker trails off.",
                ["faster-whisper"],
                "Only the supported tool name is normalized.",
                partial=True,
            ),
            "reject": _proposal(
                "the tool is faster whisper...",
                "the tool is faster-whisper and it reduces latency by fifty percent",
                "The utterance ends before any latency result is stated.",
                ["faster-whisper"],
                "The numerical performance claim is invented.",
                partial=True,
            ),
            "needs_review": _proposal(
                "the tool... faster...",
                "the tool is faster-whisper",
                "The ASR implementation may be intended, but most words are missing.",
                ["faster-whisper"],
                "The candidate is plausible, but completing the phrase is risky.",
                partial=True,
            ),
        },
    },
    {
        "category": "partial_utterance",
        "cases": {
            "accept": _proposal(
                "use c translate two for inference...",
                "use CTranslate2 for inference...",
                "The runtime backend for faster-whisper is being named.",
                ["CTranslate2"],
                "The correction only canonicalizes a supported library name.",
                partial=True,
            ),
            "reject": _proposal(
                "use c translate two...",
                "use CTranslate2 because it is always faster than every GPU runtime",
                "No comparison or universal claim appears in the source.",
                ["CTranslate2"],
                "The proposal adds an unsupported universal performance claim.",
                partial=True,
            ),
            "needs_review": _proposal(
                "c translate... inference...",
                "CTranslate2 inference is recommended",
                "The library is likely, but the recommendation is not audible.",
                ["CTranslate2"],
                "The term is supported while the recommendation is not.",
                partial=True,
            ),
        },
    },
    {
        "category": "partial_utterance",
        "cases": {
            "accept": _proposal(
                "the score is the ear...",
                "the score is DER...",
                "The preceding turn defines the diarization error metric.",
                ["DER"],
                "The local metric name is supported by preceding context.",
                partial=True,
            ),
            "reject": _proposal(
                "the score is...",
                "the score is DER and it proves perfect speaker attribution",
                "The utterance contains neither the metric name nor a conclusion.",
                ["DER"],
                "The proposal invents both a term and a perfect result.",
                partial=True,
            ),
            "needs_review": _proposal(
                "the ear... lower...",
                "DER is lower",
                "A diarization result is discussed, but the phrase is incomplete.",
                ["DER"],
                "The metric is plausible, but the asserted direction needs review.",
                partial=True,
            ),
        },
    },
    {
        "category": "speaker_attribution_risk",
        "cases": {
            "accept": _proposal(
                "Speaker A: we use piano note",
                "Speaker A: we use pyannote",
                "Speaker A is discussing the diarization toolkit.",
                ["pyannote"],
                "The term changes while speaker ownership is preserved.",
            ),
            "reject": _proposal(
                "Speaker A: we use piano note",
                "Speaker B: we use pyannote",
                "The correction has no evidence for changing speakers.",
                ["pyannote"],
                "Speaker attribution must not move without evidence.",
                speaker_ambiguity=True,
            ),
            "needs_review": _proposal(
                "Speaker A or B: we use piano note",
                "Speaker A: we use pyannote",
                "The words are audible but the active speaker is ambiguous.",
                ["pyannote"],
                "The term is supported, but assigning Speaker A requires review.",
                speaker_ambiguity=True,
            ),
        },
    },
    {
        "category": "speaker_attribution_risk",
        "cases": {
            "accept": _proposal(
                "Participant 1: our where is high",
                "Participant 1: our WER is high",
                "Participant 1 presents ASR benchmark metrics.",
                ["WER"],
                "The metric correction preserves participant attribution.",
            ),
            "reject": _proposal(
                "Participant 1: our where is high",
                "Participant 2: our WER is high",
                "No timing evidence supports reassignment to Participant 2.",
                ["WER"],
                "The proposal changes who made the claim.",
                speaker_ambiguity=True,
            ),
            "needs_review": _proposal(
                "Participant 1 and 2 overlap: where is high",
                "Participant 1: WER is high",
                "The metric is audible during overlapping speakers.",
                ["WER"],
                "The metric is supported but its speaker attribution is uncertain.",
                overlap=True,
                speaker_ambiguity=True,
            ),
        },
    },
    {
        "category": "speaker_attribution_risk",
        "cases": {
            "accept": _proposal(
                "Speaker B: diary station failed",
                "Speaker B: diarization failed",
                "Speaker B reports a speaker-analysis pipeline failure.",
                ["diarization"],
                "The supported term changes without moving the claim.",
            ),
            "reject": _proposal(
                "Speaker B: diary station failed",
                "Speaker A: diarization failed because Speaker B configured it badly",
                "No source evidence assigns blame or changes the speaker.",
                ["diarization"],
                "The proposal changes attribution and invents blame.",
                speaker_ambiguity=True,
            ),
            "needs_review": _proposal(
                "Speaker unknown: diary station failed",
                "Speaker B: diarization failed",
                "The term is audible, but the speaker label is unresolved.",
                ["diarization"],
                "The term can be corrected, while speaker assignment needs review.",
                speaker_ambiguity=True,
            ),
        },
    },
    {
        "category": "fluent_hallucination",
        "cases": {
            "accept": _proposal(
                "we evaluate where",
                "we evaluate WER",
                "The section is explicitly about ASR evaluation metrics.",
                ["WER"],
                "The proposal makes one evidence-supported term substitution.",
            ),
            "reject": _proposal(
                "we evaluate where",
                "we evaluate WER and demonstrate a statistically significant improvement",
                "No result, statistical test, or direction is present.",
                ["WER"],
                "The fluent result claim is unsupported.",
            ),
            "needs_review": _proposal(
                "we evaluate where on the meeting set",
                "we evaluate WER on the meeting set and observe an improvement",
                "The dataset is named, but no result direction is audible.",
                ["WER"],
                "WER is supported while the claimed improvement is not.",
            ),
        },
    },
    {
        "category": "fluent_hallucination",
        "cases": {
            "accept": _proposal(
                "rack retrieves the glossary terms",
                "RAG retrieves the glossary terms",
                "The correction pipeline uses retrieval-augmented generation.",
                ["RAG"],
                "The glossary retrieval context supports RAG.",
            ),
            "reject": _proposal(
                "rack retrieves the glossary terms",
                "RAG retrieves the glossary terms and guarantees zero hallucinations",
                "No guarantee or safety result is stated.",
                ["RAG"],
                "The guarantee is an unsupported new claim.",
            ),
            "needs_review": _proposal(
                "rack retrieves terms",
                "RAG retrieves relevant terms and improves every correction",
                "Retrieval is supported, but universal improvement is not.",
                ["RAG"],
                "The term is supported while the broad effectiveness claim is not.",
            ),
        },
    },
    {
        "category": "fluent_hallucination",
        "cases": {
            "accept": _proposal(
                "tag speech provides temporal labels",
                "TagSpeech provides temporal labels",
                "The literature review summarizes temporal speech labeling.",
                ["TagSpeech"],
                "The proposal only canonicalizes the paper name.",
            ),
            "reject": _proposal(
                "tag speech provides temporal labels",
                "TagSpeech solves overlapping speech better than all prior systems",
                "No comparative result is contained in the source.",
                ["TagSpeech"],
                "The superiority claim is invented.",
            ),
            "needs_review": _proposal(
                "tag speech provides labels",
                "TagSpeech provides precise temporal labels for every speaker",
                "The paper is relevant, but precision and universal scope are unstated.",
                ["TagSpeech"],
                "The canonical name is supported but the expanded claim is not.",
            ),
        },
    },
    {
        "category": "no_change_case",
        "cases": {
            "accept": _proposal(
                "put the laptop on the rack",
                "put the laptop on the rack",
                "A physical equipment instruction is already correct.",
                ["RAG"],
                "Abstaining from a tempting jargon substitution is safe.",
            ),
            "reject": _proposal(
                "put the laptop on the rack",
                "put the laptop on the RAG",
                "A physical equipment instruction is already correct.",
                ["RAG"],
                "The proposal corrupts a correct ordinary word.",
            ),
            "needs_review": _proposal(
                "check the rack result",
                "check the RAG result",
                "It is unclear whether rack names hardware or retrieval output.",
                ["RAG"],
                "The ambiguous noun should be deferred.",
            ),
        },
    },
    {
        "category": "no_change_case",
        "cases": {
            "accept": _proposal(
                "where should we meet",
                "where should we meet",
                "A normal location question is already correct.",
                ["WER"],
                "Keeping the ordinary word avoids overcorrection.",
            ),
            "reject": _proposal(
                "where should we meet",
                "WER should we meet",
                "A normal location question is already correct.",
                ["WER"],
                "The metric replacement is nonsensical.",
            ),
            "needs_review": _proposal(
                "where changed after the test",
                "WER changed after the test",
                "A test result is referenced without specifying ASR.",
                ["WER"],
                "A metric reading is possible but not sufficiently supported.",
            ),
        },
    },
    {
        "category": "no_change_case",
        "cases": {
            "accept": _proposal(
                "bonjour dear colleague",
                "bonjour dear colleague",
                "A bilingual greeting is already semantically correct.",
                ["DER"],
                "No correction is needed.",
                language="fr",
            ),
            "reject": _proposal(
                "bonjour dear colleague",
                "bonjour DER colleague",
                "A bilingual greeting is not a diarization metric.",
                ["DER"],
                "The proposal corrupts an ordinary greeting.",
                language="fr",
            ),
            "needs_review": _proposal(
                "le score dear baisse",
                "le score DER baisse",
                "A French sentence mentions a score but not diarization.",
                ["DER"],
                "DER is plausible, but the metric domain is not explicit.",
                language="fr",
            ),
        },
    },
]


def build_pilot_rows() -> list[dict[str, Any]]:
    """Return 72 balanced, explicitly authored feasibility proposals."""

    rows: list[dict[str, Any]] = []
    counter = 1
    for group in SCENARIO_GROUPS:
        for label in ("accept", "reject", "needs_review"):
            payload = dict(group["cases"][label])
            payload.update(
                {
                    "proposal_id": f"pilot_{counter:03d}",
                    "category": group["category"],
                    "retrieved_terms": json.dumps(
                        payload["retrieved_terms"],
                        ensure_ascii=False,
                    ),
                    "suggested_gold_label": label,
                    "human_checked_label": "",
                    "human_checked_rationale": "",
                    "notes": (
                        "pilot_auto_labeled; feasibility fixture; "
                        "not a real-audio annotation"
                    ),
                }
            )
            rows.append({column: payload.get(column, "") for column in COLUMNS})
            counter += 1
    return rows


def validate_pilot_rows(rows: list[dict[str, Any]]) -> None:
    if not 60 <= len(rows) <= 80:
        raise ValueError("Pilot dataset must contain 60 to 80 proposals.")
    labels = {
        label: sum(row["suggested_gold_label"] == label for row in rows)
        for label in ("accept", "reject", "needs_review")
    }
    if max(labels.values()) - min(labels.values()) > 1:
        raise ValueError(f"Pilot labels are not balanced: {labels}")
    if any(row["human_checked_label"] for row in rows):
        raise ValueError("Builder must not invent human-checked labels.")


def write_pilot_dataset(path: str | Path) -> list[dict[str, Any]]:
    rows = build_pilot_rows()
    validate_pilot_rows(rows)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=COLUMNS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the balanced selective-correction feasibility pilot."
    )
    parser.add_argument(
        "--output",
        default="data/pilot/selective_correction_pilot.csv",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = write_pilot_dataset(args.output)
    labels = {
        label: sum(row["suggested_gold_label"] == label for row in rows)
        for label in ("accept", "reject", "needs_review")
    }
    categories = sorted({str(row["category"]) for row in rows})
    print(f"Pilot proposals: {len(rows)}")
    print(f"Labels: {labels}")
    print(f"Categories: {len(categories)}")
    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
