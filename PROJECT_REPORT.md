# TalkWeaver Project Report

**Title:** TalkWeaver: An Overlap-Aware Multi-Speaker ASR System with
Diarization-Structured LLM Correction

**Subtitle:** RAG-Enhanced Domain Term Recovery for Noisy Meeting Speech

> The repository currently contains deterministic demonstration results.
> Every row with `is_mock=true` validates the evaluation workflow only and is
> not evidence of real model performance.

## 1. Abstract

Noisy meetings combine lexical recognition, speaker attribution, temporal
grounding, and overlapping speech. A standard ASR transcript answers what was
said but may not reliably preserve who spoke, when a turn occurred, or whether
two voices were active simultaneously. TalkWeaver is a modular research
prototype that combines local audio preprocessing, faster-whisper ASR,
pyannote-based diarization, timestamp alignment, overlap detection, a
temporal-anchor transcript, glossary retrieval, and constrained segment-level
LLM correction. Its main contribution is an auditable interface between ASR,
diarization, and correction: timestamps and speaker labels remain fixed,
overlap is represented explicitly, and every correction retains the raw text
and retrieved terminology. RAG is limited to domain-term recovery and
secondary transcript understanding. The repository includes reproducible
mock execution, WER, a simplified temporal speaker-error metric, Term Error
Rate, overlap error, hallucination checks, latency measurement, ablation CSVs,
charts, and a Streamlit review dashboard. Real conclusions require annotated
audio and are deliberately left for a reference-backed study.

## 2. Introduction

Meeting speech is difficult because speakers interrupt each other, turn
boundaries are uncertain, background noise changes recognition quality, and
rare technical terms are easily replaced by common words. Independent ASR and
diarization systems also optimize different objectives. Their outputs can
disagree at boundaries even when each component appears plausible.

TalkWeaver studies a practical question: can paper-inspired structure make a
multi-speaker ASR pipeline more auditable and safer to correct? The system
does not train a new foundation model. It adapts recent ideas into an
engineering pipeline that can run with real models when available and remains
demonstrable without a GPU or external credentials.

The project is organized around four research questions:

1. Does diarization-structured prompting improve speaker consistency and
   transcript readability?
2. Does overlap-aware uncertainty reduce unsupported corrections?
3. Does local domain-term retrieval improve technical-term recovery?
4. Does mono 16 kHz normalization and optional denoising improve noisy ASR?

## 3. Related Work

### 3.1 Course Anchor Paper

The required `project/xutong_paper.pdf` is not present in the repository as of
June 10, 2026. No technical claims are attributed to it. The paper must be
added locally and reviewed before final submission.

### 3.2 DiarizationLM

Wang et al., *DiarizationLM: Speaker Diarization Post-Processing with Large
Language Models*, appeared at Interspeech 2024. It represents ASR and
diarization output in a compact textual form and uses an LLM for diarization
post-processing and readability improvement. TalkWeaver adopts the structured
text interface, but adds explicit overlap flags, confidence, timestamps,
retrieved terms, and a correction audit trail.

Source: <https://arxiv.org/abs/2401.03506>

### 3.3 DM-ASR

Li et al., *DM-ASR: Diarization-aware Multi-speaker ASR with Large Language
Models*, is an arXiv preprint submitted in April 2026. It decomposes
multi-speaker recognition into speaker- and time-conditioned queries using
diarization as a structural prior. TalkWeaver adapts this idea as independent
correction of temporal speaker segments rather than implementing the paper's
trained speech-LLM architecture.

Source: <https://arxiv.org/abs/2604.22467>

### 3.4 TagSpeech

Huo et al., *TagSpeech: End-to-End Multi-Speaker ASR and Diarization with
Fine-Grained Temporal Grounding*, is an arXiv preprint submitted in January
2026. It uses temporal anchor grounding to synchronize semantic and speaker
streams and explicitly models who spoke what and when. TalkWeaver uses a
lightweight JSON temporal-anchor record for the same grounding objective; it
does not reproduce TagSpeech training or claim its reported accuracy.

Source: <https://arxiv.org/abs/2601.06896>

### 3.5 Retrieval-Augmented ASR Correction

Pusateri et al., *Retrieval Augmented Correction of Named Entity Speech
Recognition Errors*, is a 2024 arXiv preprint submitted to ICASSP 2025. It
retrieves relevant entities and supplies them to an adapted LLM for ASR
correction. TalkWeaver replaces the vector database with a small local TF-IDF
index and restricts retrieval to project terminology. This is an auxiliary
module, not the main research contribution.

Source: <https://arxiv.org/abs/2409.06062>

## 4. Research Gaps

The literature motivates four gaps addressed by the project:

- **Modular misalignment:** off-the-shelf ASR words and diarization turns need
  a transparent alignment rule.
- **Overlap ambiguity:** post-processing can erase uncertainty when more than
  one speaker is active.
- **Unsafe correction:** fluent LLM output can introduce unsupported facts or
  silently alter speaker structure.
- **Rare terminology:** domain terms may be acoustically confused with common
  phrases, while unrestricted retrieval can itself bias correction.

TalkWeaver responds with midpoint alignment, explicit overlap intervals,
confidence rules, immutable temporal anchors, lexical correction validation,
and narrow glossary retrieval.

## 5. Method

### 5.1 Audio Preprocessing

`backend/preprocessing.py` loads PCM WAV with the standard library and uses
soundfile or pydub/FFmpeg for additional formats. Audio is mixed to mono,
resampled to 16 kHz, peak-normalized, and optionally denoised when
`noisereduce` is installed. Mock mode records metadata without fabricating an
audio waveform.

### 5.2 ASR and Speaker Diarization

`backend/asr.py` uses faster-whisper with segment and word timestamps when the
dependency is available. Otherwise it exports a deterministic, labeled mock
transcript. `backend/diarization.py` uses pyannote only when the package,
accepted model access, and `HF_TOKEN` are available. Its fallback contains two
speakers and one deliberate overlap interval.

### 5.3 Alignment, Overlap, and Confidence

Words are assigned to turns by timestamp midpoint. One active speaker produces
a normal speaker label; multiple active speakers produce `OVERLAP` and retain
all speaker identities; no active turn produces `UNKNOWN`. Pairwise turn
intersection produces overlap intervals. Deterministic confidence rules assign
high confidence to single-speaker anchors and lower confidence to overlap or
unknown anchors.

### 5.4 Temporal-Anchor Transcript

Every aligned segment stores:

```json
{
  "start": 3.0,
  "end": 3.2,
  "speaker": "OVERLAP",
  "speakers": ["SPEAKER_00", "SPEAKER_01"],
  "raw_text": "The",
  "corrected_text": "The",
  "overlap": true,
  "confidence": 0.55,
  "retrieved_terms": []
}
```

This record is the contract between diarization, correction, export,
evaluation, and the Streamlit review interface.

### 5.5 Diarization-Structured Correction

`backend/prompting.py` formats one speaker-time segment at a time. The prompt
contains timestamps, speaker identities, raw text, overlap state, confidence,
and retrieved terms. `backend/llm_correction.py` preserves temporal anchors
and validates that corrected text does not add unsupported vocabulary or
content. Overlap segments are always marked uncertain. Without an API key, a
deterministic glossary rule system provides a reproducible fallback.

### 5.6 Auxiliary RAG Term Recovery

`backend/rag.py` loads local Markdown knowledge, creates TF-IDF vectors, and
retrieves a short candidate list for each segment. It supports corrections
such as `piano note -> pyannote`, `diary station -> diarization`, and
`rack -> RAG`. Retrieved terms are candidates, not permission to invent facts.

## 6. System Architecture

The pipeline is:

```text
audio -> preprocessing -> ASR -> diarization -> alignment
      -> overlap/confidence -> temporal anchors -> RAG terms
      -> constrained correction -> summary -> metrics -> review UI
```

The architecture figure is `assets/architecture.png`. Core modules live under
`backend/`, experiment scripts under `experiments/`, and review pages under
`webapp/`.

## 7. Experiments

### 7.1 Comparison Groups

| Group | Pipeline | Research role |
| --- | --- | --- |
| A | Whisper only | ASR baseline |
| B | + preprocessing | RQ4 |
| C | + diarization + alignment | speaker attribution baseline |
| D | + structured LLM correction | RQ1 |
| E | + RAG glossary | RQ3 |
| F | + overlap-aware correction | RQ2 |

### 7.2 Metrics

- **WER:** jiwer when installed, otherwise token-level Levenshtein distance.
- **Speaker error/WDER approximation:** duration-weighted active-speaker-set
  mismatch after temporal-overlap alignment. It is not full DER or WDER.
- **Term Error Rate:** required glossary terms absent from the hypothesis.
- **Overlap error:** temporal anchors with an incorrect overlap flag.
- **Hallucinated corrections:** corrected segments rejected by lexical
  grounding validation.
- **Latency:** measured elapsed time for eight pipeline stages.

### 7.3 Mock Demonstration Protocol

The built-in reference contains the intended terms `pyannote`,
`diarization`, `RAG`, `WER`, and `DER`, plus four explicit speaker-time
anchors. The mock ASR intentionally contains acoustic-like confusions. Metrics
are calculated from those fixed inputs and written with `is_mock=true`.

### 7.4 Real Experiment Protocol

Real evaluation requires consented or licensed audio, a frozen test manifest,
verbatim reference text, speaker labels, overlap regions, domain-term
annotations, fixed model versions, and recorded hardware. Each variant must
run on the same clips. The evaluator scripts accept plain text and JSON
artifacts; real rows should replace or be stored separately from mock rows.

## 8. Results

### 8.1 Deterministic Demonstration Results

The following table validates metric direction and artifact plumbing only:

| Pipeline | WER | Speaker error | TER | Overlap error | Hallucinations |
| --- | ---: | ---: | ---: | ---: | ---: |
| Whisper only | 0.4444 | 1.0000 | 1.0000 | 0.2500 | 0 |
| + preprocessing | 0.4444 | 1.0000 | 1.0000 | 0.2500 | 0 |
| + diarization + alignment | 0.4444 | 0.0000 | 1.0000 | 0.0000 | 0 |
| + structured correction | 0.4444 | 0.0000 | 1.0000 | 0.0000 | 0 |
| + RAG glossary | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 |
| + overlap-aware correction | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 |

These values are expected from a deliberately constructed example: the mock
reference exactly contains the glossary-supported corrections and the mock
speaker labels exactly match the deterministic diarization turns. Therefore,
the zeros are not evidence of generalization or model quality.

Mock elapsed times are machine- and run-dependent and measure Python demo
execution, not real-time ASR or GPU inference. The current CSV is the source
of truth for those timings.

### 8.2 Real Results Status

No reference-backed real-audio result table is claimed in this repository.
The next experimental milestone is to annotate a held-out meeting set and
rerun groups A-F.

## 9. Error Analysis

The system exposes the following error categories:

- **Speaker boundary ties:** a word midpoint may fall near two turns. Overlap
  and exact-boundary tie handling must be reviewed.
- **Missed or false overlap:** interval detection depends entirely on
  diarization quality.
- **Speaker permutation:** the simplified project metric assumes stable
  speaker labels and does not solve global label mapping.
- **Term retrieval misses:** TF-IDF may fail when the ASR error has little
  lexical similarity to the glossary entry.
- **Distractor terms:** irrelevant retrieved candidates can encourage a wrong
  correction.
- **Unsupported LLM edits:** lexical validation catches new vocabulary and
  reordering but cannot prove semantic faithfulness to audio.
- **Quiet or noisy speech:** normalization and denoising can help recognition
  but may also alter low-energy speech used by diarization.

## 10. Limitations

TalkWeaver is a modular post-processing prototype, not an end-to-end trained
multi-speaker model. Real ASR and diarization require optional dependencies,
model downloads, and credentials. The simplified speaker metric is not an
industrial DER/WDER implementation. The mock study is intentionally small and
deterministic. TF-IDF retrieval has limited semantic and phonetic recall.
Correction validation operates on text rather than acoustic evidence. English
is the primary tested language, and no privacy or production deployment claim
is made.

## 11. Future Work

Priority extensions are:

1. annotate a consented multi-speaker test set with word, speaker, and overlap
   references;
2. add standard DER/JER scoring with label mapping and collars;
3. evaluate overlap-conditioned correction against a blinded no-overlap
   prompt;
4. retrieve candidates using phonetic and embedding similarity;
5. calibrate confidence from ASR, diarization, and overlap evidence;
6. evaluate multilingual and code-switched meetings;
7. compare local and API correction under the same hallucination policy.

## 12. Conclusion

TalkWeaver demonstrates how recent ideas in diarization-aware ASR can be
translated into a reproducible final-project system. DiarizationLM motivates
structured post-processing, DM-ASR motivates speaker-time conditioned
subtasks, TagSpeech motivates explicit temporal grounding, and
retrieval-augmented ASR correction motivates narrow domain-term candidates.
The resulting pipeline keeps speaker attribution, overlap uncertainty, raw
evidence, and corrections auditable. The implementation and mock experiment
workflow are complete; defensible answers to the research questions now
depend on a real annotated evaluation set.
