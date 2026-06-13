# PRD2.md - TalkWeaver Final Direction

> Status: source of truth for the final project direction
> Version: v1 planning document
> Date: June 12, 2026
> Supersedes: product positioning and future roadmap in `PRD.md`
> Preserves: all working v0 modules, CLI behavior, tests, and mock mode

---

## 0. Executive Summary

TalkWeaver v0 proved that the repository can run an overlap-aware pipeline:

```text
audio
-> preprocessing
-> ASR
-> diarization
-> word-speaker alignment
-> overlap detection
-> temporal-anchor transcript
-> RAG term retrieval
-> constrained correction
-> summary
-> metrics
```

That implementation remains useful and must not be deleted. It becomes the
technical foundation for a more distinctive final product:

```text
TalkWeaver: AI Meeting Detective for Chaotic Multi-Speaker Conversations
```

The final website should feel like an investigation workspace, not a cold
benchmark dashboard. Users should be able to inspect a chaotic conversation
as a set of evidence: speakers, timestamps, interruptions, overlap regions,
misheard terms, retrieved glossary hints, LLM edits, unsupported corrections,
claims, stances, and action items.

The experience can be playful, but every conclusion must remain traceable to
audio, timestamps, model artifacts, or human reference annotations. The
research contribution is not the visual theme by itself. It is the connection
between an engaging conversation map and a rigorous experiment framework.

---

## 1. New Project Positioning

### 1.1 Final Title

```text
TalkWeaver: AI Meeting Detective for Chaotic Multi-Speaker Conversations
```

### 1.2 Final Subtitle

```text
An evidence-grounded conversation map for overlap, interruptions,
misheard terms, and speaker stances.
```

### 1.3 One-Sentence Product Description

TalkWeaver turns noisy multi-speaker audio into an interactive evidence board
that shows who said what, when they spoke, who interrupted whom, where the
audio became ambiguous, what ASR misheard, what RAG recovered, and what the
LLM changed.

### 1.4 Product and Research Identity

TalkWeaver is simultaneously:

- an interactive website for investigating chaotic conversations;
- an overlap- and interruption-aware multi-speaker ASR system;
- an auditable LLM correction pipeline;
- a paper-driven research adaptation;
- a controlled experiment platform;
- a final-video-ready machine learning project.

TalkWeaver is not:

- a plain meeting transcription tool;
- a generic meeting summarizer;
- a generic RAG chatbot;
- a benchmark table with a thin UI;
- a claim that recent foundation models were reproduced;
- a full mobile application project.

### 1.5 Evidence-Grounded Design Rule

Every user-facing insight must have one of these evidence links:

- audio time range;
- ASR word or segment;
- diarization turn;
- overlap or interruption event;
- retrieved glossary candidate;
- raw and corrected text pair;
- correction validation result;
- human reference annotation;
- experiment run identifier.

If evidence is missing, the UI must say `unknown`, `unsupported`, or
`needs review`. It must not silently convert uncertainty into fact.

---

## 2. Why the Old Direction Is Insufficient

### 2.1 The v0 Direction

The original prototype focused on:

- audio upload;
- meeting transcription;
- speaker diarization;
- overlap warnings;
- LLM correction;
- RAG term recovery;
- meeting summary and action items;
- a metrics dashboard.

These modules are technically relevant and should be reused.

### 2.2 Overlap with the Course Anchor Paper

The local course anchor paper, `参考文献/xutong_paper.pdf`, is an
undergraduate thesis titled:

```text
基于语音识别与大语言模型的多人对话内容管理系统
```

Its scope already includes:

- comparison of multiple ASR models;
- long-audio transcription;
- speaker feature extraction and clustering;
- speaker separation for overlapping speech;
- LLM text correction;
- LLM-assisted speaker reassignment;
- structured meeting summaries;
- speaker stance analysis;
- Streamlit interaction;
- overlap-focused experiments.

A final project presented mainly as "upload a meeting, transcribe it, label
speakers, correct it, and produce a summary" would therefore be too close to
the anchor paper in both product story and system workflow.

### 2.3 Product Weakness of the Old Direction

The v0 dashboard is useful for engineering review, but it is not distinctive
enough for a final video:

- the main interaction is reading tables and transcripts;
- pipeline stages are more visible than the conversation itself;
- overlap is shown as a warning instead of an explorable event;
- correction evidence is present but not presented as an investigation;
- speaker viewpoints are secondary;
- benchmark charts dominate the impression of the project;
- the user has limited opportunity to discover surprising conversation
  dynamics.

### 2.4 Reframing Strategy

The new direction does not discard the old pipeline. It changes the question
from:

```text
Can we generate a better meeting transcript?
```

to:

```text
Can we help a person investigate what happened in a chaotic conversation,
while showing which conclusions are supported, uncertain, or model-generated?
```

The differentiating contributions are:

1. conversation forensics rather than passive transcription;
2. interruption and overlap event modeling;
3. explicit ASR, RAG, and LLM change provenance;
4. unsupported-correction detection;
5. speaker-centered stance and claim evidence;
6. multilingual and mobile trade-off studies;
7. a fun interface grounded in real annotations and controlled experiments.

---

## 3. Final Product Vision

### 3.1 Experience Principles

The website should be:

- **Investigative:** users follow evidence through time.
- **Speaker-centered:** speaker identity and interaction are first-class.
- **Visual:** timelines, event markers, links, differences, and evidence
  states should be easier to scan than raw JSON.
- **Auditable:** every correction can be compared with its source.
- **Playful but serious:** the detective framing creates energy without
  trivializing research uncertainty.
- **Video-friendly:** each page should support a clear 30-60 second demo
  moment.
- **Progressively disclosed:** casual viewers see the story first; reviewers
  can open technical evidence and metrics.

### 3.2 Shared Investigation Model

All pages should use the same selected evidence context:

```text
clip_id
selected time range
selected speaker or speakers
selected overlap/interruption event
raw ASR
corrected text
reference annotation, if available
retrieval evidence
correction support status
experiment run metadata
```

Selecting an event on one page should eventually deep-link or filter related
evidence on other pages.

### 3.3 Required Pages

#### Page 1: Conversation Crime Scene

**Purpose:** Provide the memorable opening view for the demo.

**Experience:**

- play the audio;
- show the full conversation as a visual evidence strip;
- mark speaker turns, overlap, interruptions, uncertain regions, and rescued
  terms;
- select any event to open its evidence panel;
- show a concise case summary without hiding uncertainty.

**Primary question:** What happened in this conversation?

**Evidence shown:** timestamps, active speakers, audio playback range, raw
words, corrected words, confidence, and review status.

**Planned files:**

```text
webapp/pages/1_Conversation_Crime_Scene.py
webapp/components/conversation_map.py
webapp/components/evidence_drawer.py
```

#### Page 2: Speaker Timeline Detective

**Purpose:** Investigate who spoke when.

**Experience:**

- one lane per speaker;
- zoomable or filterable turns;
- speaker activity totals;
- uncertain and unknown assignments;
- click a turn to inspect words and audio;
- compare predicted turns with human reference anchors when available.

**Primary question:** Who said what, and when?

**Planned files:**

```text
webapp/pages/2_Speaker_Timeline_Detective.py
webapp/components/speaker_timeline.py
webapp/components/anchor_inspector.py
```

#### Page 3: Interruption Map

**Purpose:** Make overlap and conversational dynamics visible.

**Experience:**

- show interruption edges from one speaker to another;
- distinguish cooperative overlap, backchannels, turn competition, and
  uncertain overlap when labels exist;
- list interruption count, duration, and affected transcript spans;
- replay a short window around an event;
- expose the operational rule used to infer the event.

**Primary question:** Who interrupted whom, and what speech was affected?

**Planned files:**

```text
webapp/pages/3_Interruption_Map.py
webapp/components/interruption_graph.py
backend/interruption.py
```

#### Page 4: Misheard Word Rescue

**Purpose:** Show the clearest RAG + ASR synergy story.

**Experience:**

- present each suspected ASR error as a rescue case;
- compare raw ASR, candidate terms, corrected term, and reference term;
- show lexical, fuzzy, and phonetic-like evidence scores;
- mark successful rescue, missed rescue, false rescue, or needs review;
- allow filters by speaker, language, and domain.

**Primary question:** Which technical terms were misheard, and why was a
replacement justified?

**Planned files:**

```text
webapp/pages/4_Misheard_Word_Rescue.py
webapp/components/term_rescue_viewer.py
backend/term_recovery.py
```

RAG remains an auxiliary correction module. This page must not become a
general-purpose chatbot.

#### Page 5: Hallucination Watchdog

**Purpose:** Audit every LLM change.

**Experience:**

- raw-versus-corrected diff;
- retrieved evidence;
- validation rules triggered;
- support status: supported, weakly supported, unsupported, or review;
- special warnings for overlap regions;
- correction acceptance/rejection audit trail;
- optional human review decision.

**Primary question:** What did the LLM change, and was it allowed to change it?

**Planned files:**

```text
webapp/pages/5_Hallucination_Watchdog.py
webapp/components/correction_audit.py
backend/correction_audit.py
```

#### Page 6: Speaker Stance Cards

**Purpose:** Organize claims and positions by speaker without inventing them.

**Experience:**

- one evidence-backed card per speaker;
- claims, agreements, disagreements, questions, and action items;
- source timestamp for every item;
- confidence and support state;
- side-by-side stance comparison;
- explicit `insufficient evidence` state.

**Primary question:** What position did each speaker actually express?

**Planned files:**

```text
webapp/pages/6_Speaker_Stance_Cards.py
webapp/components/stance_cards.py
backend/stance.py
```

Stance extraction is not personality inference. It may summarize expressed
positions only.

#### Page 7: Evidence Dashboard

**Purpose:** Connect the product story to research rigor.

**Experience:**

- reference coverage and annotation status;
- WER/CER, speaker error, DER/WDER approximation, overlap metrics, TER,
  correction support, and latency;
- ASR comparison and ablation charts;
- clip-level drill-down;
- visible labels for mock, development, and held-out results.

**Primary question:** How well does the evidence pipeline perform?

**Planned files:**

```text
webapp/pages/7_Evidence_Dashboard.py
webapp/components/evidence_metrics.py
```

The dashboard supports the story; it is not the main story.

#### Page 8: Multilingual Demo

**Purpose:** Demonstrate language-dependent behavior and mixed-language
limitations.

**Experience:**

- compare English, Mandarin, and code-switched samples;
- show WER for space-delimited languages and CER where appropriate;
- inspect term rescue across languages;
- show language-specific failure examples;
- avoid combining incomparable metrics without explanation.

**Primary question:** Does the pipeline preserve speakers, terms, and evidence
across languages?

**Planned files:**

```text
webapp/pages/8_Multilingual_Demo.py
experiments/run_multilingual.py
experiments/evaluate_cer.py
```

#### Page 9: Mobile ASR Trade-off Study

**Purpose:** Present deployment trade-offs without building a full mobile app.

**Experience:**

- compare quantized model size, WER/CER, real-time factor, latency, and memory;
- display a Pareto-style accuracy-speed chart;
- distinguish desktop emulation from real-device measurements;
- link each point to model and hardware metadata.

**Primary question:** What accuracy is lost or retained when ASR must run on a
mobile-class device?

**Planned files:**

```text
webapp/pages/9_Mobile_ASR_Tradeoff.py
experiments/benchmark_mobile_asr.py
experiments/plot_mobile_tradeoff.py
```

---

## 4. Research Backbone

### 4.1 Reference Inventory

The following local papers were inspected for this plan:

| Paper | Local source |
| --- | --- |
| Course anchor thesis (`xutong_paper.pdf`) | `参考文献/xutong_paper.pdf` |
| DiarizationLM | `参考文献/2401.03506v11.pdf` |
| Retrieval Augmented Correction of Named Entity Speech Recognition Errors | `参考文献/2409.06062v1.pdf` |
| Diarization-Aware Multi-Speaker ASR via LLMs | `参考文献/2506.05796v1.pdf` |
| TagSpeech | `参考文献/2601.06896v1.pdf` |
| DM-ASR | `参考文献/2604.22467v1.pdf` |

The reference PDFs are research inputs. Do not commit redistributed copies
unless their licenses and repository policy permit it.

### 4.2 Paper-to-Implementation Mapping

| Paper | Key idea | Limitation relevant to TalkWeaver | TalkWeaver adaptation | Implemented or planned files | Experiment that validates it |
| --- | --- | --- | --- | --- | --- |
| `xutong_paper.pdf`: multi-speaker conversation management system using ASR and LLMs | Integrates ASR comparison, speaker features/clustering, overlap-oriented separation, segment correction, speaker/stance analysis, summaries, and Streamlit | A similar upload-transcribe-summarize product would not sufficiently differentiate TalkWeaver; some semantic correction can also fill or reorganize content without a strict evidence audit | Reframe the product as a conversation investigation board; retain model comparison and overlap study, but add interruption events, correction provenance, unsupported-change detection, reference manifests, multilingual analysis, and mobile trade-offs | Existing `backend/`, `experiments/`, and `webapp/`; planned `backend/interruption.py`, `backend/correction_audit.py`, `backend/stance.py`, and detective pages | B pipeline ablation, C correction ablation, E overlap-aware correction, plus human evidence-review analysis |
| DiarizationLM | Converts ASR and diarization output to compact text for LLM post-processing and speaker consistency improvement | Depends on upstream ASR/diarization quality; post-processing can hide uncertainty if the interface lacks explicit evidence state | Preserve a compact speaker-time representation, but include overlap, confidence, raw text, retrieved evidence, validation status, and immutable anchors | Implemented `backend/prompting.py`, `backend/alignment.py`, `backend/llm_correction.py`; planned `backend/correction_audit.py` | C LLM correction ablation and B pipeline ablation using speaker consistency, unsupported edits, WER, and human review |
| Diarization-Aware Multi-Speaker Automatic Speech Recognition via Large Language Models | Conditions an LLM backend on speaker embeddings and time-bound triplets to transcribe highly overlapped multi-speaker audio | Requires trained multimodal encoders, LLM integration, and large-scale data; modular errors and resource requirements remain substantial | Do not reproduce the model. Adapt the speaker-plus-time triplet as an evidence query and evaluation unit in the existing modular pipeline | Implemented `backend/diarization.py`, `backend/alignment.py`, `backend/pipeline.py`; planned `backend/evidence.py` and anchor comparison tools | A ASR comparison, B pipeline ablation, and E overlap experiment on anchor-level attribution |
| DM-ASR | Reformulates multi-speaker ASR as speaker- and time-conditioned dialogue queries using diarization as an explicit prior; optionally predicts word timestamps | The full approach is a trained Speech-LLM and cannot be claimed from post-processing alone; it also relies on diarization quality | Correct and investigate one speaker-time segment at a time while preserving the mixed-audio context and explicit diarization prior | Implemented `backend/prompting.py`, `backend/llm_correction.py`, `backend/alignment.py`; planned `backend/stance.py` | C LLM correction ablation: whole transcript vs segment structured vs overlap-aware segment structured |
| TagSpeech | Uses decoupled semantic/speaker streams and interleaved temporal anchors to model who spoke what and when, including overlap | End-to-end training, projector tuning, and benchmark-scale evaluation are outside project scope | Use temporal anchors as the stable cross-module contract and as the clickable unit in the investigation UI; do not claim TagSpeech reproduction | Implemented `backend/alignment.py`, `backend/export.py`, `webapp/components/speaker_timeline.py`; planned `webapp/components/anchor_inspector.py` | B pipeline ablation, E overlap experiment, and reference-anchor evaluation |
| Retrieval Augmented Correction of Named Entity Speech Recognition Errors | Generates queries from errorful ASR, retrieves relevant entities, and gives candidates to an adapted LLM; acoustic-aware representations improve rare-entity recovery | Retrieval can introduce distractors; the paper uses a large entity database and adapted models that are beyond this project | Extend local TF-IDF retrieval with controlled fuzzy and phonetic-like candidate scoring; show every candidate and require evidence-backed correction | Implemented `backend/rag.py`, `experiments/evaluate_terms.py`; planned `backend/term_recovery.py`, `backend/correction_audit.py` | D RAG term recovery experiment: no retrieval vs TF-IDF vs fuzzy vs phonetic-like vs combined |

### 4.3 Research Questions for the Final Direction

**RQ1 - Conversation structure:** How accurately can a modular pipeline recover
speaker-time anchors, overlap, and interruption events from chaotic meetings?

**RQ2 - Structured correction:** Does speaker-time structured correction
improve lexical quality without increasing unsupported edits or speaker
confusion?

**RQ3 - Overlap safety:** Does explicit overlap uncertainty reduce
hallucinated or overconfident corrections in cross-speech regions?

**RQ4 - Term recovery:** Do glossary, fuzzy, and phonetic-like retrieval
improve domain-term recall without causing false substitutions?

**RQ5 - Speaker evidence:** Can claim, stance, and action-item extraction remain
fully traceable to speaker-time evidence?

**RQ6 - Multilingual behavior:** How do ASR, speaker attribution, term recovery,
and correction differ across English, Mandarin, and code-switched clips?

**RQ7 - Deployment trade-off:** What accuracy, speed, memory, and model-size
trade-offs appear when ASR is moved toward mobile-class inference?

### 4.4 Paper Baselines and Reproduction Policy

Recent papers may be used as baselines when their official code, package, or
pretrained model is feasible in the available environment. Baseline selection
must follow `docs/baseline_feasibility.md`.

#### Level A - Official Runnable Baseline

A baseline qualifies for Level A only when:

- an official repository or paper-provided implementation exists;
- a pretrained model or installable package is available;
- small-sample inference can run in the available environment;
- its outputs can be converted into a documented TalkWeaver format;
- the run can be clearly labeled `small-scale baseline run`.

Level A does not imply full reproduction. A short inference run on
TalkWeaver clips is a feasibility baseline unless the original paper protocol
is matched.

#### Level B - Proxy Baseline

Use Level B when the full method is too heavy, unavailable, or incompatible,
but its core idea can be approximated using existing TalkWeaver modules.

Examples:

- DiarizationLM-style structured post-processing;
- TagSpeech-inspired temporal anchors;
- DM-ASR-inspired speaker-time conditioned subtasks;
- RAG-ASR-inspired term retrieval and constrained correction.

Proxy outputs must name the adapted idea and must not use the original model
name as if the paper system had been executed.

#### Level C - Literature Baseline

Use Level C when no feasible official runtime exists or the required data,
weights, hardware, or protocol are unavailable.

Paper results may be discussed for qualitative comparison or motivation, but
must not be inserted into TalkWeaver result tables as directly comparable
measurements.

#### Strict Reproduction Rules

- Do not claim full reproduction unless the same dataset, model, metric, and
  evaluation protocol are used.
- Do not clone or copy third-party code into this repository unless the
  license is compatible and attribution obligations are documented.
- Prefer isolated external checkouts, packages, containers, or adapters over
  vendoring another project.
- Do not commit downloaded models, private audio, restricted datasets, API
  keys, access tokens, or other credentials.
- Before any baseline run, record repository URL, commit or release version,
  license, dependencies, hardware, expected download size, and expected
  runtime.
- If a baseline cannot run, document the exact blocker.
- If only part of the original method can run, label it
  `small-scale reproduction` or `proxy baseline`, as appropriate.
- Do not run heavy training, multi-gigabyte downloads, or long benchmark jobs
  without explicit approval.
- Keep paper-reported results separate from locally measured results.

#### Required Baseline Run Record

Every attempted baseline should write or document:

```text
baseline_name
baseline_level
official_url
commit_or_version
license
model_id
dataset_or_clip_ids
hardware
dependencies
download_size
runtime_seconds
output_path
status
failure_reason
claim_label
```

---

## 5. Core Technical Modules

### 5.1 Module Status and Requirements

| Module | v1 responsibility | Existing foundation | Planned work | Main outputs |
| --- | --- | --- | --- | --- |
| ASR front end | Run comparable ASR variants with word timestamps and explicit model metadata | `backend/asr.py`, `backend/preprocessing.py` | Config-driven model adapters, batch manifest runner, CER support | raw transcript JSON, model metadata, latency |
| Diarization | Produce speaker turns and overlap-capable speaker activity | `backend/diarization.py` | Reference comparison, stable speaker mapping, optional stronger backends | speaker turns JSON, speaker metadata |
| Temporal-anchor transcript | Preserve who/what/when as the stable evidence contract | `backend/alignment.py`, `backend/export.py` | Add evidence IDs, reference links, correction status, language | anchor JSON and Markdown |
| Overlap and interruption detection | Separate simultaneous activity from conversational interruption events | `backend/overlap.py`, `backend/confidence.py` | Add operational interruption rules and event classification | overlap JSON, interruption JSON |
| RAG glossary / fuzzy / phonetic-like term recovery | Recover likely domain terms from errorful ASR while exposing candidate evidence | `backend/rag.py` | Add fuzzy score, pronunciation-oriented normalization or phonetic keys, candidate fusion, false-rescue controls | term candidates JSON, rescue decisions |
| Constrained LLM correction | Correct one speaker-time segment while preserving anchors and uncertainty | `backend/prompting.py`, `backend/llm_correction.py` | Prompt variants, structured response schema, run metadata | corrected anchors, prompt audit |
| Hallucination / unsupported-correction detector | Classify whether each edit is supported by raw text, retrieved terms, neighboring context, or reference | lexical validator in `backend/llm_correction.py` | Dedicated edit extraction, support rules, review state, human decision field | correction audit JSON |
| Speaker stance extraction | Extract expressed claims, agreements, disagreements, and actions with source anchors | `backend/summarizer.py` | Evidence-only stance schema and contradiction-safe extraction | speaker stance JSON |
| Report export | Create a reviewable case report and experiment package | `backend/export.py` | HTML/PDF-ready evidence report, provenance appendix, redaction options | case report, CSV/JSON bundle |

### 5.2 Preserve Existing Working Modules

The following rules are mandatory:

- do not delete existing backend modules;
- do not break current mock mode;
- do not remove existing CLI commands;
- do not replace deterministic tests with network-dependent tests;
- add adapters around existing modules before rewriting them;
- retain v0 output readers or provide a migration layer;
- keep old generated artifacts clearly labeled as v0/mock.

### 5.3 Temporal-Anchor v1 Schema

The v0 fields remain valid. v1 may extend them as follows:

```json
{
  "anchor_id": "clip01_a0007",
  "clip_id": "clip01",
  "start": 12.4,
  "end": 15.8,
  "speaker": "SPEAKER_01",
  "speakers": ["SPEAKER_01"],
  "language": "en",
  "raw_text": "The rack system improves where.",
  "corrected_text": "The RAG system improves WER.",
  "overlap": false,
  "interruption_event_ids": [],
  "confidence": 0.82,
  "retrieved_terms": ["RAG", "WER"],
  "correction_status": "supported",
  "evidence_ids": ["word_0041", "term_rag", "term_wer"],
  "needs_review": false,
  "is_reference": false
}
```

### 5.4 Interruption Event Schema

Overlap and interruption are related but not identical. An interruption event
requires an operational rule, such as a second speaker entering before the
first speaker's turn ends and then taking or contesting the floor.

```json
{
  "event_id": "clip01_i0002",
  "clip_id": "clip01",
  "start": 21.7,
  "end": 23.1,
  "interrupter": "SPEAKER_02",
  "interrupted": "SPEAKER_00",
  "speakers": ["SPEAKER_00", "SPEAKER_02"],
  "overlap_duration": 1.4,
  "outcome": "floor_takeover",
  "label_source": "human_reference",
  "confidence": 1.0
}
```

Planned outcomes:

- `floor_takeover`;
- `failed_interruption`;
- `cooperative_overlap`;
- `backchannel`;
- `simultaneous_start`;
- `uncertain`.

Automatic inference must remain separate from human reference labels.

### 5.5 Correction Audit Schema

```json
{
  "edit_id": "clip01_e0011",
  "anchor_id": "clip01_a0007",
  "raw_span": "rack",
  "corrected_span": "RAG",
  "edit_type": "domain_term_replacement",
  "support_status": "supported",
  "supporting_term_ids": ["term_rag"],
  "overlap": false,
  "validator_reasons": ["retrieved_glossary_match"],
  "human_decision": "unreviewed"
}
```

Support states:

- `supported`;
- `weakly_supported`;
- `unsupported`;
- `uncertain_overlap`;
- `human_approved`;
- `human_rejected`.

### 5.6 Speaker Stance Schema

```json
{
  "speaker": "SPEAKER_01",
  "claims": [
    {
      "text": "The team should evaluate WER and DER separately.",
      "anchor_ids": ["clip01_a0012"],
      "confidence": 0.91
    }
  ],
  "agreements": [],
  "disagreements": [],
  "questions": [],
  "action_items": [],
  "insufficient_evidence": false
}
```

No stance item may exist without at least one source anchor.

---

## 6. Experiment Plan

### 6.1 Shared Experiment Rules

Every experiment must:

- run from a frozen manifest;
- store model, prompt, dependency, hardware, and commit metadata;
- distinguish mock, development, and held-out test results;
- preserve per-clip outputs before aggregation;
- record failed and excluded runs;
- use the same clip split for compared variants;
- never convert missing measurements into estimated values;
- never present mock outputs as real evidence.

Recommended common output root:

```text
experiments/results/v1/
```

Each run should receive a stable identifier:

```text
YYYYMMDD_HHMM_<experiment>_<variant>_<git-sha>
```

### 6.2 Experiment A - ASR Model Comparison

**Purpose**

Compare lexical accuracy, timestamp availability, language behavior, and
runtime before downstream diarization and correction.

**Inputs**

- frozen `manifest.csv`;
- reference transcripts;
- English, Mandarin, code-switched, noisy, overlap, and mobile-recorded clips;
- identical preprocessing policy per comparison.

**Model variants**

- faster-whisper `small`;
- faster-whisper `medium`;
- faster-whisper `large-v3` when hardware permits;
- whisper.cpp quantized variants for the mobile-style comparison;
- optional additional ASR adapters only when licensing, dependencies, and
  reproducibility are documented.

**Metrics**

- WER for English;
- CER for Mandarin and suitable code-switched analysis;
- domain-term recall;
- timestamp coverage;
- latency;
- real-time factor;
- failure rate;
- model size and peak memory where available.

**Expected output files**

```text
experiments/results/v1/asr_model_comparison.csv
experiments/results/v1/asr_per_clip.csv
outputs/experiments/asr/<run_id>/
assets/result_charts/v1/asr_accuracy_speed.png
```

**Website appearance**

- Evidence Dashboard: model comparison and per-condition filters;
- Multilingual Demo: language-specific examples;
- Mobile ASR Trade-off Study: quantized variants.

### 6.3 Experiment B - Pipeline Ablation

**Purpose**

Measure the contribution and error propagation of the modular pipeline.

**Inputs**

- one fixed ASR output set or fixed audio/model pair;
- reference transcripts;
- reference anchors;
- reference overlap/interruption annotations;
- reference terms.

**Model variants**

```text
B0 ASR only
B1 ASR + preprocessing
B2 ASR + diarization + alignment
B3 B2 + temporal-anchor structured correction
B4 B3 + term retrieval
B5 B4 + overlap-aware constraints
B6 B5 + correction watchdog
```

**Metrics**

- WER/CER;
- speaker attribution error;
- DER or documented approximation;
- overlap precision/recall/F1;
- interruption precision/recall/F1;
- Term Error Rate;
- unsupported correction rate;
- latency.

**Expected output files**

```text
experiments/results/v1/pipeline_ablation.csv
experiments/results/v1/pipeline_ablation_per_clip.csv
assets/result_charts/v1/pipeline_ablation.png
```

**Website appearance**

- Evidence Dashboard: ablation chart;
- Conversation Crime Scene: toggle between selected pipeline variants for one
  clip.

### 6.4 Experiment C - LLM Correction Ablation

**Purpose**

Determine whether structured prompts improve corrections without increasing
unsupported changes.

**Inputs**

- frozen raw ASR and aligned anchors;
- reference transcripts;
- overlap labels;
- reference terms;
- a fixed LLM provider/model or deterministic local rule variant.

**Model variants**

```text
C0 no correction
C1 whole-transcript plain correction
C2 segment correction without speaker/time structure
C3 speaker-time structured correction
C4 speaker-time structured + overlap uncertainty
C5 C4 + correction watchdog rejection
```

**Metrics**

- WER/CER after correction;
- corrected error count;
- introduced error count;
- unsupported correction rate;
- speaker-label mutation count;
- overlap-region correction precision;
- human review acceptance rate;
- latency and token usage when available.

**Expected output files**

```text
experiments/results/v1/llm_correction_ablation.csv
outputs/experiments/correction/<run_id>/prompts.jsonl
outputs/experiments/correction/<run_id>/audit.jsonl
assets/result_charts/v1/correction_support.png
```

**Website appearance**

- Hallucination Watchdog: edit-level evidence;
- Conversation Crime Scene: correction status markers.

### 6.5 Experiment D - RAG Term Recovery

**Purpose**

Test whether retrieval methods recover domain terms without causing false
replacements.

**Inputs**

- clips containing known technical terms;
- reference transcript;
- `reference_terms.json`;
- domain glossary;
- raw ASR hypotheses.

**Model variants**

```text
D0 no retrieval
D1 TF-IDF glossary
D2 fuzzy string candidate retrieval
D3 phonetic-like candidate retrieval
D4 fused TF-IDF + fuzzy + phonetic-like retrieval
D5 D4 + constrained LLM selection
```

Phonetic-like retrieval may use documented pronunciation normalization,
phoneme conversion, Soundex-style keys for English, or pinyin-aware matching
for Mandarin. It must not be described as acoustic retrieval unless it uses
actual acoustic representations.

**Metrics**

- Term Error Rate;
- term precision, recall, and F1;
- candidate recall at K;
- false rescue rate;
- missed rescue count;
- overall WER/CER impact;
- retrieval and correction latency.

**Expected output files**

```text
experiments/results/v1/term_recovery.csv
experiments/results/v1/term_candidates.jsonl
assets/result_charts/v1/term_recovery.png
```

**Website appearance**

- Misheard Word Rescue: candidate and decision cases;
- Evidence Dashboard: TER and false-rescue chart.

### 6.6 Experiment E - Overlap-Aware Correction

**Purpose**

Evaluate whether overlap evidence improves safety and interpretation in
cross-speech regions.

**Inputs**

- real and synthetic overlap clips;
- reference transcript;
- reference active speakers;
- reference overlap and interruption events;
- aligned raw ASR anchors.

**Model variants**

```text
E0 correction without overlap information
E1 correction with binary overlap flag
E2 correction with active speakers + overlap duration
E3 E2 + conservative uncertainty instruction
E4 E3 + watchdog rejection or human-review routing
```

**Metrics**

- overlap-region WER/CER;
- non-overlap-region WER/CER;
- overlap precision/recall/F1;
- interruption precision/recall/F1;
- unsupported correction rate in overlap;
- preserved uncertain-span rate;
- human review acceptance rate.

**Expected output files**

```text
experiments/results/v1/overlap_correction.csv
experiments/results/v1/interruption_detection.csv
outputs/experiments/overlap/<run_id>/events.json
assets/result_charts/v1/overlap_safety.png
```

**Website appearance**

- Interruption Map: event-level playback and errors;
- Hallucination Watchdog: overlap-specific warnings;
- Evidence Dashboard: overlap safety comparison.

### 6.7 Experiment F - Multilingual Experiment

**Requirement level:** Must-have. This experiment and its website page are
required for the final project.

**Purpose**

Measure how language and code-switching affect ASR, speaker attribution, term
recovery, and correction.

**Inputs**

- English clips;
- Mandarin clips;
- English-Mandarin code-switched clips;
- matched or carefully documented recording conditions;
- language-specific references and term annotations.

**Model variants**

- selected multilingual ASR baseline;
- selected language-specialized adapter if available;
- correction off/on;
- term retrieval off/on;
- language-aware term matching off/on.

**Metrics**

- WER;
- CER;
- mixed-language token or span accuracy with a documented method;
- speaker attribution error;
- overlap F1;
- term precision/recall;
- language-switch error analysis;
- latency.

**Expected output files**

```text
experiments/results/v1/multilingual.csv
experiments/results/v1/multilingual_errors.jsonl
assets/result_charts/v1/multilingual_comparison.png
```

**Website appearance**

- Multilingual Demo: side-by-side clips and error cases;
- Evidence Dashboard: language filter.

### 6.8 Experiment G - Mobile ASR Trade-off

**Requirement level:** Must-have at Level 1. A full native mobile application
is not required, but the whisper.cpp quantized-model trade-off experiment and
its website page are required.

**Purpose**

Evaluate practical ASR deployment trade-offs without making a mobile app the
main project.

**Inputs**

- a small frozen mobile benchmark subset;
- clean and noisy phone recordings;
- English and Mandarin samples where supported;
- reference transcripts;
- device and runtime metadata.

**Model variants**

- whisper.cpp quantized models on desktop or mobile-class hardware;
- optional WhisperKit/Core ML models on Apple hardware;
- optional ONNX Runtime or Android-compatible model;
- server baseline for comparison.

**Metrics**

- WER/CER;
- real-time factor;
- end-to-end latency;
- model size;
- peak memory;
- load time;
- energy or battery proxy only if measured reliably;
- accuracy-speed Pareto position.

**Expected output files**

```text
experiments/results/v1/mobile_asr.csv
experiments/results/v1/mobile_device_metadata.json
assets/result_charts/v1/mobile_accuracy_speed.png
assets/result_charts/v1/mobile_memory_size.png
```

**Website appearance**

- Mobile ASR Trade-off Study: filterable comparison and Pareto chart;
- Evidence Dashboard: deployment summary.

---

## 7. Data Plan

### 7.1 Public-Dataset-First Strategy

Formal evaluation must prioritize public datasets with documented references,
licenses, versions, and evaluation splits. Self-recorded audio is not the main
dataset and is not required for every experiment.

The preferred public sources are:

- AISHELL-4 or AliMeeting for Mandarin multi-speaker meeting evaluation;
- AMI or LibriCSS for English meeting and overlap evaluation;
- Common Voice for multilingual ASR evaluation, including English, French,
  and Chinese;
- VoxConverse as optional in-the-wild diarization data.

Public data must be registered in the same manifest and reference schema as
local data. When dataset licenses prohibit redistribution, commit only
metadata, download/preparation instructions, checksums where permitted, and
derived results that comply with the dataset terms.

### 7.2 Dataset Decision Table

| Dataset | Purpose | Languages | Speakers | Overlap | Has transcript | Has speaker/time labels | License/access risk | Use in project |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AISHELL-4 | Mandarin meeting ASR, diarization, and overlap evaluation | Mandarin Chinese | Multi-speaker meetings | Yes | Yes | Yes | Access and redistribution terms must be verified for the selected release | Primary Mandarin meeting candidate |
| AliMeeting | Mandarin meeting ASR, speaker attribution, and overlap evaluation | Mandarin Chinese | Multi-speaker meetings | Yes | Yes | Yes | Version-specific access and redistribution restrictions may apply | Primary or fallback Mandarin meeting candidate |
| AMI Meeting Corpus | English meeting ASR, diarization, interruption, and overlap evaluation | English | Primarily four-person meetings | Yes | Yes | Yes | Follow AMI access and redistribution terms; avoid committing restricted audio | Primary English meeting candidate |
| LibriCSS | Controlled English overlap and multi-speaker ASR evaluation | English | Multi-speaker sessions derived from LibriSpeech | Controlled overlap conditions | Yes | Yes, depending on the selected reference package | Derived-corpus terms and source-data conditions must be recorded | Primary English overlap candidate or AMI complement |
| Mozilla Common Voice | Multilingual ASR comparison and WER/CER evaluation | English, French, Chinese, and other languages | Usually one speaker per clip; many speakers across the corpus | Not a meeting-overlap corpus | Yes | No meeting-level speaker/time annotations | Release-specific terms, required attribution, and download access must be checked | Required multilingual sample set; not used as diarization evidence |
| VoxConverse | Optional in-the-wild diarization robustness evaluation | Primarily English audiovisual speech | Variable | Possible natural overlap | Limited for ASR evaluation | Diarization references are available for supported partitions | Dataset and source-media redistribution rules require review | Optional diarization-only or qualitative robustness track |
| Self-recorded or self-created clip | Consent-safe demo, mobile capture, and controlled technical-term failures | Project-selected, preferably English/Mandarin or code-switching | 2-6 for conversation demos; one or more for mobile ASR | Designed as needed | Human-authored reference required | Human annotations created by the team | Redistribution requires explicit participant consent | Optional but recommended for the video, mobile study, and terms such as pyannote, diarization, RAG, WER, and DER |
| Synthetic overlap sample | Controlled overlap ratio, timing, and SNR study | Based on licensed source speech | Two or more mixed sources | Deliberately controlled | Source transcripts required | Generated timing labels | Every source clip must permit the intended derivative use | Supporting controlled experiment, never the sole formal evidence |

Dataset capabilities and licensing must be verified against the exact release
before download or experiment execution. This table is a planning decision,
not a redistribution authorization.

Recommended experiment mapping:

| Experiment | Primary dataset recommendation | Supporting or optional data |
| --- | --- | --- |
| ASR model comparison | AMI for English meetings; AISHELL-4 or AliMeeting for Mandarin meetings | Common Voice for language-specific ASR behavior |
| Pipeline ablation | A frozen subset of AMI plus AISHELL-4 or AliMeeting with compatible references | LibriCSS controlled-overlap subset |
| LLM correction ablation | Public meeting segments with human references | Consent-safe technical-term clips for controlled correction failures |
| RAG term recovery | Consent-safe or synthetic clips containing the fixed domain glossary | Public segments only when the target terms genuinely occur |
| Overlap-aware correction | LibriCSS controlled overlaps and AMI natural overlaps | AISHELL-4 or AliMeeting Mandarin overlap segments |
| Multilingual evaluation | Common Voice English, French, and Chinese sample sets | Public meeting samples for English/Mandarin context, reported separately |
| Mobile ASR trade-off | One phone-recorded consent-safe sample or documented public mobile/device sample | Common Voice clips replayed or processed under a documented device protocol |
| Optional in-the-wild diarization | VoxConverse | Qualitative case review unless suitable ASR references are also available |

### 7.3 Self-Recorded Data Role

Self-recorded or self-created clips are:

- not mandatory for all experiments;
- recommended for the final video demo because consent and scene design can be
  controlled;
- recommended for the mobile ASR recording experiment;
- recommended for controlled technical-term failures such as `pyannote`,
  `diarization`, `RAG`, `WER`, and `DER`;
- safe for redistribution only when every participant has provided explicit
  consent for the intended use.

When created, controlled clips should deliberately include:

- clean turn-taking;
- short backchannels;
- one clear interruption with floor takeover;
- one failed interruption;
- one cooperative overlap;
- one high-overlap disagreement;
- technical terms likely to be misheard;
- explicit action items;
- claims that can support stance cards;
- uncertain or inaudible content that should remain unresolved.

### 7.4 Minimum Data Plan

The minimum viable study requires:

- **Public data:** at least one English meeting/overlap sample from AMI or
  LibriCSS, one Mandarin meeting sample from AISHELL-4 or AliMeeting, and one
  multilingual ASR sample set covering English, French, and Chinese from
  Common Voice or an equivalently documented public source;
- **Demo data:** one consent-safe chaotic conversation clip, either
  self-recorded or selected from a permissive public source;
- **Mobile data:** one phone-recorded sample or one clearly documented
  mobile/public-device sample;
- a held-out evaluation partition that is not used to tune prompts, correction
  rules, or the RAG glossary.

Additional synthetic overlap samples and VoxConverse evaluation are useful but
not part of the minimum dataset.

### 7.5 Data Governance

For every clip record:

- consent and redistribution status;
- original source;
- language;
- speaker count;
- recording device;
- duration;
- noise condition;
- overlap condition;
- split;
- annotation status.

No private or restricted audio may be committed to GitHub. Only small
synthetic examples or audio whose license and participant consent explicitly
permit repository redistribution may be committed. The repository should
otherwise contain manifests, dataset preparation scripts, versioned
instructions, and permitted derived metadata or results.

---

## 8. Annotation Plan

### 8.1 Directory Layout

```text
data/
├── manifests/
│   └── manifest.csv
└── reference/
    └── <clip_id>/
        ├── reference_transcript.txt
        ├── reference_anchors.json
        ├── reference_terms.json
        └── reference_events.json
```

### 8.2 `manifest.csv`

Required columns:

```text
clip_id
audio_path
source_type
dataset_name
dataset_version
source_recording_id
source_url
license_or_access_notes
split
language
duration_seconds
speaker_count
has_overlap
has_interruptions
has_domain_terms
recording_device
noise_condition
consent_status
redistribution_status
transcript_status
anchor_status
term_status
event_status
notes
```

Example:

```csv
clip_id,audio_path,source_type,dataset_name,dataset_version,source_recording_id,source_url,license_or_access_notes,split,language,duration_seconds,speaker_count,has_overlap,has_interruptions,has_domain_terms,recording_device,noise_condition,consent_status,redistribution_status,transcript_status,anchor_status,term_status,event_status,notes
ami_eval_001,external/ami/ami_eval_001.wav,public_dataset,AMI,recorded_release_version,original_meeting_and_segment_id,official_dataset_url,verify_and_record_exact_terms,test,en,42.8,4,true,true,false,corpus_audio,meeting_room,not_applicable,restricted,complete,complete,not_required,complete,AMI excerpt; audio is not committed
```

For self-recorded entries, `dataset_name` may be `talkweaver_consent_safe`,
`dataset_version` may identify the recording batch, and `source_url` may be
empty. Consent and redistribution fields remain mandatory. For public
datasets, consent may be `not_applicable`, but dataset version, official
source, access terms, and redistribution status must be recorded.

### 8.3 Reference Transcript

`reference_transcript.txt` is a verbatim human transcript:

- preserve disfluencies when audible;
- mark unintelligible spans with a documented token;
- do not silently repair grammar;
- use a documented punctuation policy;
- preserve code-switching;
- include no LLM-generated content unless a human verifies every span.

Recommended uncertainty token:

```text
[INAUDIBLE 12.40-13.10]
```

### 8.4 Reference Anchors JSON

```json
[
  {
    "anchor_id": "demo_001_ref_0001",
    "start": 0.0,
    "end": 2.4,
    "speaker": "SPK_A",
    "speakers": ["SPK_A"],
    "text": "We should compare WER and DER separately.",
    "language": "en",
    "overlap": false,
    "annotation_status": "adjudicated"
  }
]
```

Annotation rules:

- use stable reference speaker IDs;
- preserve simultaneous speakers in `speakers`;
- allow overlapping anchor intervals;
- record annotator and adjudication metadata separately;
- never copy predicted speaker labels into the reference without review.

### 8.5 Reference Terms JSON

```json
[
  {
    "term_id": "demo_001_term_01",
    "canonical": "pyannote.audio",
    "spoken_forms": ["pyannote", "pyannote audio"],
    "start": 8.2,
    "end": 9.1,
    "speaker": "SPK_B",
    "language": "en",
    "domain": "speaker diarization"
  }
]
```

This file supports candidate recall, Term Error Rate, false rescue analysis,
and multilingual matching.

### 8.6 Reference Overlap / Interruption JSON

Use `reference_events.json`:

```json
[
  {
    "event_id": "demo_001_event_01",
    "type": "interruption",
    "start": 15.1,
    "end": 16.0,
    "speakers": ["SPK_A", "SPK_C"],
    "interrupter": "SPK_C",
    "interrupted": "SPK_A",
    "outcome": "floor_takeover",
    "annotation_status": "adjudicated"
  },
  {
    "event_id": "demo_001_event_02",
    "type": "overlap",
    "start": 27.3,
    "end": 28.0,
    "speakers": ["SPK_A", "SPK_B"],
    "subtype": "cooperative_overlap",
    "annotation_status": "adjudicated"
  }
]
```

### 8.7 Annotation Workflow

1. Create the manifest row.
2. Produce a first-pass human transcript.
3. Add speaker-time anchors.
4. Add domain-term annotations.
5. Add overlap and interruption events.
6. Run a second annotator review on test clips.
7. Adjudicate disagreements.
8. Validate schema and time boundaries with a script.
9. Freeze the test manifest before prompt or glossary tuning.

Planned validation:

```text
scripts/validate_manifest.py
scripts/validate_references.py
```

---

## 9. Mobile Deployment Plan

Mobile ASR evaluation is a mandatory experiment track. Building a complete
native iOS or Android application is not required. The required deliverable is
the Level 1 quantized-model trade-off study with reproducible measurements and
website presentation.

### 9.1 Level 1 - Mobile-Style Benchmark

**Must-have**

- use whisper.cpp quantized models;
- run on available desktop CPU, laptop, or mobile-class hardware;
- record quantization, threads, device, model size, memory, latency, and RTF;
- use phone-recorded audio;
- clearly label desktop emulation versus real-device execution.

Expected files:

```text
experiments/benchmark_mobile_asr.py
configs/mobile_models.yaml
experiments/results/v1/mobile_asr.csv
```

### 9.2 Level 2 - Apple Route

**Should-have when compatible Apple hardware is available. If no compatible
hardware can be obtained, implementation becomes could-have and the blocker
must be documented.**

- evaluate WhisperKit and/or Core ML conversion;
- measure on a named Mac, iPhone, or iPad;
- preserve model version and precision;
- reuse the same mobile benchmark subset;
- avoid building a polished native app before the benchmark works.

### 9.3 Level 3 - Optional Android or ONNX Runtime Demo

**Could-have. It may be promoted to should-have if suitable Android hardware,
an ONNX Whisper model, and implementation time are available.**

- ONNX Runtime Mobile or an Android-compatible whisper.cpp wrapper;
- one minimal inference demo or captured benchmark;
- no requirement for a full production UI;
- only pursue after core evidence pages and real experiments are complete.

### 9.4 Required Mobile Metrics

- WER/CER;
- real-time factor;
- end-to-end latency;
- model size;
- peak memory;
- model load time;
- accuracy-speed trade-off;
- device and runtime metadata.

Peak memory and energy claims must be measured with a documented tool. If they
cannot be measured reliably, report them as unavailable.

---

## 10. Proposed Repository Evolution

The exact file layout may adapt to existing patterns, but the planned
direction is:

```text
backend/
├── interruption.py
├── term_recovery.py
├── correction_audit.py
├── stance.py
├── evidence.py
└── report.py

data/
├── manifests/
│   └── manifest.csv
└── reference/<clip_id>/

experiments/
├── compare_asr_models.py
├── run_pipeline_ablation.py
├── run_correction_ablation.py
├── run_term_recovery.py
├── run_overlap_experiment.py
├── run_multilingual.py
├── benchmark_mobile_asr.py
├── evaluate_cer.py
├── evaluate_interruptions.py
└── results/v1/

webapp/
├── pages/
│   ├── 1_Conversation_Crime_Scene.py
│   ├── 2_Speaker_Timeline_Detective.py
│   ├── 3_Interruption_Map.py
│   ├── 4_Misheard_Word_Rescue.py
│   ├── 5_Hallucination_Watchdog.py
│   ├── 6_Speaker_Stance_Cards.py
│   ├── 7_Evidence_Dashboard.py
│   ├── 8_Multilingual_Demo.py
│   └── 9_Mobile_ASR_Tradeoff.py
└── components/
    ├── conversation_map.py
    ├── evidence_drawer.py
    ├── anchor_inspector.py
    ├── interruption_graph.py
    ├── term_rescue_viewer.py
    ├── correction_audit.py
    └── stance_cards.py
```

Do not create all files at once without an implementing phase that needs them.

---

## 11. Implementation Phases

### Phase 1 - Documentation and Paper Mapping

**Goal:** Establish the differentiated direction before adding features.

Tasks:

- adopt `PRD2.md` as the source of truth;
- update agent instructions;
- update the course anchor paper reading note;
- add a separate note for Diarization-Aware Multi-Speaker ASR via LLMs;
- map every paper to one adaptation and one experiment;
- update research questions and report outline;
- document v0 modules that will be preserved.

Exit criteria:

- no project document describes the final product as only a meeting
  transcription dashboard;
- all six papers have verified local-source notes;
- planned claims are separated from implemented behavior.

### Phase 2 - Real Data and Manifest

**Goal:** Create the evidence base for real evaluation.

#### Phase 2A-REAL Status

**Partially implemented on June 12, 2026.**

Implemented:

- size-capped official-source acquisition scripts;
- 15 real Google FLEURS validation clips: five English, five French, and five
  Mandarin Chinese clips, used as a clearly labeled Common Voice fallback;
- two real 20-second AMI `ES2002a` excerpts with reference transcripts,
  speaker anchors, and derived overlap events;
- source-specific and combined manifests with SHA-256 inventories;
- strict local-file and JSON manifest validation;
- 17 complete rows in `data/manifests/formal_eval_real.csv`;
- raw audio and archives excluded from Git.

Remaining blockers:

- Mozilla Common Voice partial access was unavailable through the attempted
  official Hugging Face endpoint; the current Mozilla Data Collective route
  requires credentials, terms acceptance, and archive-level downloads;
- AISHELL-4 official archives exceed the 500 MB acquisition ceiling;
- no verified file-level AliMeeting route for a few clips plus matching
  annotations has been established;
- demo/mobile data remains pending.

These blockers are documented in `docs/dataset_acquisition.md` and
`docs/manual_dataset_steps.md`. No missing or planned rows are included in the
combined real manifest.

Tasks:

- define and validate `manifest.csv`;
- register public-dataset samples as the primary formal evaluation data;
- support both public-dataset and consent-safe self-recorded manifest entries;
- obtain the minimum English meeting/overlap, Mandarin meeting, multilingual,
  demo, and mobile samples described in Section 7 without committing
  restricted audio;
- create reference transcripts;
- annotate temporal anchors;
- annotate domain terms;
- annotate overlap and interruption events;
- establish development and held-out splits.

Exit criteria:

- at least one complete end-to-end annotated public meeting or overlap sample;
- minimum-source coverage is represented in the manifest, with unavailable
  items explicitly marked rather than fabricated;
- schema validation passes;
- license/access, consent, and redistribution status are recorded.

### Phase 2B - Core Evidence Workflow

**Status: implemented on June 13, 2026, including a real ASR smoke run.**

Implemented:

- `backend/schemas.py` for the v1 `ConversationMap` evidence contract;
- `backend/temporal_anchor.py` for speaker/content/time alignment;
- `backend/events.py` for overlap and conservative interruption candidates;
- `backend/term_rescue.py` for glossary, fuzzy, and phonetic-like candidates;
- `backend/constrained_correction.py` for per-anchor correction and audits;
- `backend/conversation_map.py` for speaker cards, summary, and JSON export;
- `scripts/run_talkweaver_workflow.py` for manifest-aware execution;
- explicit mock, reference-assisted, and real evidence modes;
- successful mock and AMI reference-assisted smoke runs;
- successful real `faster-whisper` CPU smoke execution with
  `asr_mode=real`.

Current limitation:

- real mode fails clearly with fallback disabled when model dependencies are
  unavailable;
- automatic diarization still requires a configured pyannote model and
  `HF_TOKEN`;
- speaker stance output remains extractive and does not infer unsupported
  personality, intent, or position;
- interruption events are timing-based candidates requiring human review.

This phase implements a paper-inspired proxy workflow. It does not reproduce
DiarizationLM, DM-ASR, Diarization-Aware Multi-Speaker ASR via LLMs, or
TagSpeech. Benchmark and ablation runs begin only after this evidence contract
is stable.

### Phase 2C - Real ASR Baseline

**Status: implemented on June 13, 2026.**

Implemented:

- language-aware WER for English/French and CER for Mandarin Chinese;
- dependency-light Levenshtein metrics and conservative text normalization;
- real `faster-whisper` manifest runner with no mock fallback;
- per-clip prediction JSON/TXT artifacts with word timestamps;
- per-clip runtime and real-time factor measurements;
- aggregate summaries grouped by model, language, and dataset;
- real result charts for language error rate and model RTF;
- successful `tiny` and `base` CPU/int8 runs over all 17 manifest clips.

The committed result CSVs and charts are explicitly labeled as small-subset
formal evaluation. Prediction dumps remain local and ignored by Git. The
benchmark is ASR-only and does not validate diarization, overlap reasoning,
RAG, LLM correction, or the complete TalkWeaver method. Protocol and measured
limitations are documented in `docs/asr_benchmark.md`.

### Phase 2C-Fix - ASR Evaluation Reliability

**Status: implemented on June 13, 2026.**

Implemented:

- optional OpenCC Traditional-to-Simplified normalization before Mandarin
  CER, with explicit metadata and graceful fallback;
- separate standard WER and diagnostic disfluency-cleaned WER for AMI;
- configurable `--vad-filter true/false` and `--only-dataset`;
- AMI VAD-disabled diagnostic results for `tiny` and `base`;
- separate warm per-clip RTF and process-level model load timing;
- dataset-and-metric chart separating AMI WER, FLEURS WER, and FLEURS CER;
- updated real CSVs, summaries, charts, tests, and interpretation notes.

The fix changes evaluation normalization and diagnostics, not ASR
predictions. AMI cleaned WER is supplementary and does not replace standard
WER. Local model load timing is not a mobile cold-start measurement.

### Phase 2D - Speaker-Time and Overlap Baseline

**Status: implemented on June 13, 2026, with automatic pyannote pending
model access.**

Implemented:

- lightweight label-permutation-aware speaker/time metrics;
- overlap interval precision, recall, and F1 scoring;
- conservative interruption event matching;
- a 17-clip runner covering `no_diarization`, `reference_assisted`, and
  `pyannote_optional` modes;
- two AMI reference-assisted ConversationMaps that reuse Phase 2C real ASR
  prediction JSON;
- workflow CLI support for prediction JSON reuse, VAD metadata, reference,
  no-diarization, and optional pyannote evidence sources.

The two AMI excerpts provide five reference overlap events and validate the
speaker/time and overlap pipeline. They do not provide human interruption
labels, so interruption scores remain unreported. Reference-assisted results
are oracle workflow checks and must not be described as automatic
diarization performance. Pyannote inference remains pending because no
`HF_TOKEN` model access was configured for this run.

### Phase 2E - TalkWeaver Workflow Ablation

**Status: implemented on June 13, 2026.**

Implemented:

- seven explicit workflow variants from `asr_only` through
  `full_talkweaver`;
- stable loading and strict non-mock validation of Phase 2C prediction JSON;
- reuse of fixed real `base` ASR predictions without model reruns;
- variant-specific temporal anchors, oracle/reference speaker-time,
  overlap/interruption evidence, term retrieval, constrained correction,
  correction audits, speaker cards, and extractive summary;
- WER/CER correction scoring through the same Phase 2C normalization policy;
- 119 real small-subset result rows, a 28-row grouped summary, 119 local
  ConversationMaps, and two result charts.

The full workflow produced 25 anchors, 23 speaker-labeled anchors, four
overlap anchors, five events, 25 correction audits, six review flags, and
zero unsupported changes. The public subset contains no annotated project
technical terms, so conservative retrieval produced zero candidates and
correction changed no text. WER/CER therefore remained equal to the fixed
ASR baseline. This validates evidence completeness and auditability, not
correction accuracy improvement.

### Phase 2F-0 - Secure Optional LLM API Preparation

**Status: implemented on June 13, 2026.**

Prepared for the controlled Phase 2F correction experiment:

- generic `.env` configuration for DeepSeek, Qwen, and OpenAI-compatible
  endpoints;
- a credential-safe `LLMConfig` loader with strict validation and masked
  metadata;
- explicit `rule_fallback`, `llm`, and `llm_with_rule_fallback` modes;
- strict failure behavior that never reports a failed API call as successful
  LLM correction;
- correction audits recording provider, model, prompt version, temperature,
  API use, fallback use, unsupported changes, and review risk;
- an offline rule smoke path and an optional real API smoke command.

No real API call or correction-quality result is claimed by this preparation
phase. Normal tests remain network-free, and deterministic correction remains
available without credentials.

### Phase 3 - Metrics and Experiment Runners

**Goal:** Make all future UI evidence reproducible.

Tasks:

- add CER;
- add interruption metrics;
- strengthen speaker/anchor evaluation;
- implement model-comparison runner;
- implement correction, term, and overlap ablations;
- add run metadata and per-clip outputs;
- preserve compatibility with v0 metrics.

Exit criteria:

- experiment commands run from the manifest;
- mock and real results cannot be confused;
- every aggregate row links to per-clip artifacts.

### Phase 4 - AI Meeting Detective Frontend

**Goal:** Build the final product experience over stable evidence schemas.

Order:

1. Conversation Crime Scene;
2. Speaker Timeline Detective;
3. Interruption Map;
4. Misheard Word Rescue;
5. Hallucination Watchdog;
6. Speaker Stance Cards;
7. Evidence Dashboard.

Tasks:

- redesign navigation and visual language;
- add linked evidence selection;
- add event-level audio playback;
- keep technical details available through drawers or expanders;
- capture desktop and mobile screenshots;
- test missing-data and mock states.

Exit criteria:

- the first screen demonstrates the investigation experience;
- every insight links to evidence;
- no page requires unavailable credentials in mock mode.

### Phase 5 - Real Experiments

**Goal:** Replace demo-only conclusions with measured results.

Tasks:

- run ASR comparison;
- run pipeline ablation;
- run correction ablation;
- run RAG term recovery;
- run overlap-aware correction;
- review error cases;
- freeze charts used in the report and video.

Exit criteria:

- results include sample counts and per-clip data;
- no mock chart appears as real evidence;
- limitations and failed runs are reported.

### Phase 6 - Multilingual and Mobile Experiments

**Goal:** Complete the mandatory multilingual evaluation and Level 1 mobile
ASR benchmark after the core research pipeline works.

Tasks:

- annotate English, Mandarin, and code-switched samples;
- run multilingual comparison;
- run Level 1 whisper.cpp benchmark;
- attempt Level 2 Apple route only when hardware permits;
- add Multilingual Demo and Mobile ASR Trade-off pages.

Exit criteria:

- language metrics are reported appropriately;
- mobile results include device and runtime metadata;
- optional platform work does not block the core submission.

### Phase 7 - Final Report and Video

**Goal:** Present one coherent, evidence-backed story.

Tasks:

- update `PROJECT_REPORT.md`;
- update `BLOG_ARTICLE.md`;
- update literature and paper notes;
- finalize contribution records;
- finalize the 10+ minute video script;
- capture the investigation workflow;
- export charts and case reports;
- verify Git history and reproducibility instructions.

Exit criteria:

- the final narrative begins with a chaotic conversation case;
- research papers motivate specific modules;
- experiments validate specific adaptations;
- limitations are explicit;
- the website supports, rather than replaces, the research contribution.

---

## 12. Scope Control

### 12.1 Must-Have

- preserve the working v0 pipeline and mock mode;
- a real `manifest.csv`;
- public datasets as the primary source for formal evaluation;
- at least one English meeting/overlap sample;
- at least one Mandarin meeting sample;
- a multilingual English/French/Chinese ASR sample set;
- one consent-safe demo clip from a self-recorded or permissive public source;
- one phone-recorded or documented mobile/public-device sample;
- at least one fully annotated real conversation;
- reference transcript and temporal anchors;
- reference overlap/interruption events;
- ASR model comparison;
- pipeline ablation;
- correction ablation;
- RAG term recovery experiment;
- overlap-aware correction experiment;
- multilingual evaluation and Multilingual Demo;
- Level 1 whisper.cpp quantized-model benchmark;
- Mobile ASR Trade-off Study page;
- Conversation Crime Scene;
- Speaker Timeline Detective;
- Interruption Map;
- Misheard Word Rescue;
- Hallucination Watchdog;
- evidence-backed Speaker Stance Cards;
- Evidence Dashboard;
- explicit mock-versus-real labeling;
- final report and video based on measured evidence.

### 12.2 Should-Have

- one or more self-recorded, consent-safe chaotic clips for the final video;
- additional public-dataset samples and held-out coverage;
- fuzzy and phonetic-like term candidate retrieval;
- human correction review decisions;
- CER and code-switch error analysis;
- Apple WhisperKit/Core ML benchmark when compatible hardware is available;
- downloadable evidence report.

### 12.3 Could-Have

- optional speech separation experiment;
- richer interruption subtype classification;
- network-style interaction graph;
- Android or ONNX Runtime minimal demo;
- human annotation helper UI;
- HTML or PDF case report;
- limited live microphone demo after offline analysis is stable.

### 12.4 Do-Not-Do

- do not train a new ASR foundation model;
- do not claim to reproduce DM-ASR or TagSpeech;
- do not claim to reproduce Diarization-Aware Multi-Speaker ASR via LLMs;
- do not make benchmark tables the main product story;
- do not build a full iOS and Android app before the core project works;
- do not omit the required multilingual evaluation or Level 1 mobile ASR
  trade-off study merely because native app development is out of scope;
- do not fabricate experimental results;
- do not delete working v0 modules to make the new UI;
- do not turn RAG into a generic meeting chatbot;
- do not infer personality, intent, emotion, or stance without transcript
  evidence;
- do not present automatic interruption labels as human ground truth;
- do not hide uncertain or inaudible speech with fluent LLM text;
- do not commit private recordings, credentials, restricted datasets, or
  public-dataset audio that is not explicitly redistributable.

---

## 13. Acceptance Criteria

The final project is ready when:

1. `PRD2.md` is followed as the active direction.
2. The existing mock pipeline still runs.
3. At least one real annotated clip can be loaded end to end.
4. The website opens with an investigation experience rather than a pipeline
   configuration screen.
5. A user can click an overlap or interruption event and replay its evidence.
6. A user can inspect raw ASR, retrieved terms, LLM edits, and support status.
7. Every stance, claim, and action item links to source anchors.
8. ASR, ablation, RAG, overlap, multilingual, and mobile result files clearly
   distinguish measured, unavailable, and mock values.
9. The report maps each research adaptation to an experiment.
10. The video demonstrates both the fun interface and the rigorous evidence
    workflow.

---

## 14. Final Video Narrative

The final video should tell this story:

1. Start with a chaotic clip containing interruptions, overlap, and a
   misheard technical term.
2. Open the Conversation Crime Scene and ask: "What actually happened?"
3. Follow speakers through the timeline.
4. Open the Interruption Map and replay the contested turn.
5. Show Misheard Word Rescue and explain why the RAG candidate is evidence,
   not permission to rewrite freely.
6. Open Hallucination Watchdog and reject or flag an unsupported correction.
7. Compare speaker stance cards with their timestamped claims.
8. Explain how DiarizationLM, diarization-aware MS-ASR, DM-ASR, TagSpeech, and
   retrieval-based correction motivated specific design choices.
9. Show real ASR, ablation, term, overlap, multilingual, and mobile trade-off
   results.
10. End with limitations: modular error propagation, overlap ambiguity,
    annotation cost, language coverage, and deployment constraints.

The message is:

```text
TalkWeaver does not merely generate a clean transcript.
It lets users investigate the evidence behind a chaotic conversation.
```

---

## 15. Immediate Next Step

Do not start by rebuilding all Streamlit pages.

The next implementation task should be:

```text
Phase 2: create the real-data manifest and annotation schemas, add validation
scripts, register the minimum public-dataset-first evaluation sources, and
prepare one fully annotated public meeting or overlap sample while preserving
all v0 behavior. A consent-safe self-recorded clip may be added for the demo
or mobile study, but self-recording is not a prerequisite for formal
evaluation.
```

Once the evidence contracts are stable, the new frontend can be built against
real structures instead of temporary UI assumptions.
