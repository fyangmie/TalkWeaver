# Literature Review

## Scope

TalkWeaver reviews work on speaker diarization, multi-speaker ASR,
overlapping speech, LLM-assisted correction, temporal grounding, and
retrieval-assisted recovery of rare terms. Meeting summarization and QA are
secondary outputs.

## Source Status

The required local course paper, `project/xutong_paper.pdf`, was not available
on June 10, 2026. It is not summarized or cited. The remaining entries below
were checked against their primary arXiv records.

## Research Map

| Paper | Status | Key idea | Limitation relevant to TalkWeaver | TalkWeaver adaptation |
| --- | --- | --- | --- | --- |
| Course `xutong_paper.pdf` | Local source unavailable | Unknown | No claims can be made without the PDF | Placeholder note only |
| DiarizationLM | Interspeech 2024 | Compact text representation of ASR and diarization for LLM post-processing | Post-processing depends on upstream ASR/diarization and an appropriate LLM; overlap remains an upstream evidence problem | Structured speaker-time prompts with overlap, confidence, and audit fields |
| DM-ASR | arXiv preprint, submitted April 24, 2026 | Speaker- and time-conditioned queries use diarization as an explicit prior | Full method is a trained multi-speaker speech-LLM system and is not reproduced here | Correct one diarized temporal segment at a time |
| TagSpeech | arXiv preprint, submitted January 11, 2026 | Temporal Anchor Grounding synchronizes semantic and speaker streams | End-to-end training and benchmark-scale evaluation exceed this project scope | Temporal-anchor JSON for who spoke what and when |
| Retrieval Augmented Correction of Named Entity Speech Recognition Errors | arXiv preprint, submitted September 9, 2024; submitted to ICASSP 2025 | Retrieve relevant entities and provide them to an adapted LLM for ASR correction | Candidate retrieval can add distractors and the reported setting targets named entities | Local TF-IDF glossary retrieval for constrained technical-term correction |

## Paper Notes

### DiarizationLM

Wang et al. represent ASR and diarization outputs in a compact textual form
that can be post-processed by an optionally fine-tuned LLM. The paper reports
WDER reductions on Fisher and Callhome, but those paper results are not
TalkWeaver results. The transferable idea is the interface: independent
speech components can be converted into a structured textual correction task.

Source: <https://arxiv.org/abs/2401.03506>

### DM-ASR

Li et al. frame multi-speaker ASR as a sequence of speaker- and
time-conditioned queries. Diarization supplies reliable speaker identity and
segment boundaries, while an LLM handles linguistic content and longer-range
context. TalkWeaver adopts the decomposition but uses post-ASR correction
instead of a trained speech-LLM recognizer.

Source: <https://arxiv.org/abs/2604.22467>

### TagSpeech

Huo et al. introduce Temporal Anchor Grounding, decoupled semantic and speaker
streams, and interleaved time anchors. The relevant insight is that time can
act as a synchronization signal between what was said and who said it.
TalkWeaver implements a transparent temporal-anchor export rather than the
paper's end-to-end model.

Source: <https://arxiv.org/abs/2601.06896>

### Retrieval-Augmented ASR Correction

Pusateri et al. retrieve entities using queries derived from errorful ASR
hypotheses and supply those entities to an adapted LLM. TalkWeaver narrows the
idea to a local project glossary. Retrieved terms are correction candidates,
not new meeting facts.

Source: <https://arxiv.org/abs/2409.06062>

## Cross-Paper Synthesis

The papers suggest complementary roles:

- diarization provides speaker and time structure;
- ASR provides lexical hypotheses and word timestamps;
- temporal anchors keep speaker and content synchronized;
- LLMs can revise text when constrained by structure;
- retrieval can improve rare-term context but introduces candidate noise.

TalkWeaver's research gap is not a missing foundation model. It is the lack of
a lightweight, auditable workflow that combines these roles while exposing
overlap uncertainty and supporting controlled ablation.

## Evaluation Implications

The literature leads directly to four experiments:

- structured versus unstructured correction for RQ1;
- overlap-aware versus overlap-agnostic correction for RQ2;
- correction with and without glossary retrieval for RQ3;
- raw versus locally preprocessed audio for RQ4.

WER alone is insufficient. Speaker attribution, overlap behavior, term
recovery, hallucination, and latency must also be recorded.
