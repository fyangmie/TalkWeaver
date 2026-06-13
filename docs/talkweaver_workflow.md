# TalkWeaver Core Workflow

## Purpose

TalkWeaver's core method is the **Temporal-Anchor Evidence-Grounded
Correction Workflow**. It turns lexical, speaker, timing, overlap, retrieval,
and correction evidence into one auditable `ConversationMap`.

```text
audio or reference transcript
  -> ASR segments and words
  -> speaker turns
  -> overlap and interruption candidates
  -> temporal anchors: who said what when
  -> glossary/fuzzy/phonetic-like term candidates
  -> constrained per-anchor correction
  -> unsupported-change audit
  -> speaker evidence cards and extractive summary
  -> ConversationMap JSON
```

This workflow is the method behind the future AI Meeting Detective pages. It
is not a benchmark runner and does not report accuracy results.

## Paper-To-Module Mapping

| Research source | TalkWeaver adaptation | Files | Claim boundary |
| --- | --- | --- | --- |
| Course anchor paper (`xutong_paper`) | Multi-person dialogue management, cross-speech evidence, semantic correction, and Streamlit-oriented output | Existing v0 pipeline plus `backend/conversation_map.py` | Product and engineering inspiration; no paper result reproduction |
| DiarizationLM | Compact ASR-plus-diarization structure for constrained post-processing | `backend/constrained_correction.py`, `backend/temporal_anchor.py` | Proxy adaptation; no DiarizationLM model or full protocol is reproduced |
| DM-ASR | Speaker-time-conditioned processing of one anchor at a time | `backend/temporal_anchor.py`, `backend/constrained_correction.py` | Post-ASR correction proxy; no speech-LLM training or DM-ASR inference |
| Diarization-Aware Multi-Speaker ASR via LLMs | Explicit speaker/time evidence retained through correction | `backend/schemas.py`, `backend/conversation_map.py` | Structural adaptation only |
| TagSpeech | Temporal anchors connect content, speaker identity, and time | `backend/temporal_anchor.py`, `backend/schemas.py` | Transparent JSON proxy; no end-to-end TagSpeech reproduction |
| Retrieval-Augmented ASR Correction | Retrieved domain terms become correction candidates and audit evidence | `backend/term_rescue.py`, existing `backend/rag.py` | Lightweight glossary/fuzzy approximation, not the paper's trained retriever |

## ConversationMap Schema

The root object is defined in `backend/schemas.py`:

```text
ConversationMap
├── clip_id
├── metadata
├── anchors[]
├── events[]
├── term_rescues[]
├── correction_audits[]
├── speaker_cards[]
└── summary
```

Each temporal anchor preserves:

- `anchor_id`, `clip_id`, `start`, and `end`;
- primary `speaker` and all active `speakers`;
- immutable `raw_text` and separately stored `corrected_text`;
- language, overlap, interruption, and confidence fields;
- retrieved terms and correction evidence;
- unsupported changes and `needs_review`.

All schemas are dataclasses with `to_dict()` and `to_json()` helpers. The
exported JSON schema version is `talkweaver.conversation_map.v1`.

## Evidence Modes

### Mock Mode

```bash
python scripts/run_talkweaver_workflow.py \
  --manifest data/manifests/formal_eval_real.csv \
  --clip-id fleurs_en_1548 \
  --mock-models \
  --output outputs/conversation_maps/
```

Metadata is explicit:

```text
asr_mode=mock
diarization_mode=mock
llm_mode=rule_fallback
is_mock=true
```

Mock mode uses deterministic v0 ASR and diarization evidence. It is a smoke
test, not a measured result for the selected manifest audio.

### Reference-Assisted Mode

```bash
python scripts/run_talkweaver_workflow.py \
  --manifest data/manifests/formal_eval_real.csv \
  --clip-id ami_es2002a_01 \
  --asr-source reference \
  --diarization-source reference \
  --output outputs/conversation_maps/
```

This loads the manifest's transcript anchors and events. It is deliberately
labeled:

```text
asr_mode=reference
diarization_mode=reference
reference_assisted=true
```

Reference speaker/time evidence is oracle evidence. It must not be reported
as automatic diarization performance.

### Real ASR With Reference Diarization

Check and install the optional runtime first:

```bash
python scripts/check_optional_dependencies.py
pip install -r requirements-optional.txt
python scripts/check_optional_dependencies.py --strict faster-whisper
```

Then run the CPU smoke path:

```bash
python scripts/run_talkweaver_workflow.py \
  --manifest data/manifests/formal_eval_real.csv \
  --clip-id ami_es2002a_01 \
  --asr-model tiny \
  --device cpu \
  --compute-type int8 \
  --diarization-source reference \
  --output outputs/conversation_maps/
```

Real ASR calls `faster-whisper` with mock fallback disabled. If the package or
model is unavailable, the command exits non-zero. It never relabels mock text
as real ASR, and `asr_mode=real` is written only after the backend reports a
successful `faster_whisper` run. The recommended first configuration is
`--asr-model tiny --device cpu --compute-type int8`.

The first model use downloads CTranslate2 weights from Hugging Face. Model
caches, raw audio, and generated outputs must remain outside Git. Automatic
diarization similarly requires configured
`pyannote.audio` and `HF_TOKEN`; otherwise use the explicitly requested
reference mode during workflow development.

See [`dependency_setup.md`](dependency_setup.md) for Python compatibility,
CPU/GPU setup, cache policy, and troubleshooting.

### Reusing Phase 2C Prediction JSON

Workflow experiments can reuse a saved real prediction without loading the
ASR model again:

```bash
python scripts/run_talkweaver_workflow.py \
  --manifest data/manifests/formal_eval_real.csv \
  --clip-id ami_es2002a_01 \
  --asr-source prediction-json \
  --asr-prediction-json \
    experiments/results/asr_predictions_real/base__ami_es2002a_01.json \
  --diarization-source reference \
  --output outputs/conversation_maps/
```

Accepted diarization evidence sources are:

- `reference`: oracle/reference turns from the manifest;
- `none`: no speaker turns, for a no-diarization ablation;
- `pyannote` (or legacy alias `real`): automatic inference only when the
  package, token, and model access are available.

Real ASR also accepts `--vad-filter true` or `--vad-filter false`. Prediction
JSON metadata records whether VAD was used during the original benchmark.
Neither real ASR nor pyannote silently falls back to mock evidence.

## Event Rules

`backend/events.py` implements two conservative timing rules:

- **overlap:** two or more distinct speaker turns are active simultaneously;
- **interruption candidate:** a later speaker starts before another speaker
  finishes, overlaps by at least the configured threshold, and continues
  after the earlier turn ends.

An interruption event is a floor-taking timing proxy. It does not establish
intent, hostility, or discourse function and requires human review.

## Term Rescue

`backend/term_rescue.py` supports:

- exact glossary terms;
- known ASR error forms such as `piano note -> pyannote`;
- standard-library fuzzy matching;
- a simple normalized consonant-style phonetic proxy;
- English, French, and Mandarin glossary aliases.

Candidates retain score, method, and source anchor IDs. Retrieval does not
automatically authorize replacement. The correction stage must confirm that
the candidate supports the edit.

## Correction And Audit

Correction is performed independently per temporal anchor:

1. timestamps and speaker labels remain immutable;
2. raw text remains stored;
3. only `corrected_text` may change;
4. overlap anchors are conservative correction zones;
5. inserted tokens must be supported by raw text, retrieved terms, or local
   neighboring evidence;
6. unsupported rewrites are rejected and the raw text remains active;
7. every attempted correction receives a `CorrectionAudit`.

`needs_review=true` is set for overlap/unknown anchors and for unsupported,
large, or length-expanding corrections.

Correction supports three explicit modes:

- `rule_fallback`: deterministic glossary substitutions, no API call;
- `llm`: strict API mode with no silent fallback;
- `llm_with_rule_fallback`: API attempt followed by an explicitly recorded
  deterministic fallback if unavailable or rejected.

Every audit records provider, model, prompt version, temperature,
`api_used`, and `fallback_used`. See
[`llm_api_setup.md`](llm_api_setup.md) for secure configuration.

## Current Implementation

Implemented:

- JSON-serializable evidence schemas;
- temporal-anchor builder over existing ASR/diarization contracts;
- overlap and conservative interruption candidates;
- multilingual term candidate retrieval;
- constrained correction and unsupported-change auditing;
- speaker evidence cards with extractive fallback;
- manifest-aware CLI;
- mock and reference-assisted smoke paths;
- `ConversationMap` JSON export.

Not implemented yet:

- controlled technical-term and real-LLM correction ablations beyond the
  completed Phase 2E workflow ablation;
- automatic semantic interruption classification;
- validated stance extraction;
- paper baseline model inference;
- real LLM correction evaluation beyond the configuration smoke path;
- automatic pyannote inference in the current environment;
- the AI Meeting Detective frontend pages that consume this schema.

## Current Environment Status

As of June 13, 2026, real `faster-whisper` CPU inference has been confirmed,
and the Phase 2C `tiny`/`base` ASR-only baseline has run over all 17 local
formal-manifest clips. Mock and reference-assisted workflows remain
available. Automatic diarization still requires a valid pyannote model setup
and `HF_TOKEN`; no supported LLM API key is required for the deterministic
correction fallback.

Recheck the local environment at any time:

```bash
python scripts/check_optional_dependencies.py
```

See [`asr_benchmark.md`](asr_benchmark.md) for the separate ASR-only
evaluation protocol and results.
See [`speaker_overlap_baseline.md`](speaker_overlap_baseline.md) for the
Phase 2D reference speaker-time and event baseline.

## Output

Conversation maps are written to:

```text
outputs/conversation_maps/<clip_id>_conversation_map.json
```

Generated outputs are ignored by Git by default. Small reviewed examples may
be copied into a dedicated documented example directory later if their source
license permits it.

## Why This Is Not A Plain Benchmark

A benchmark reports aggregate model performance. TalkWeaver's method creates
the evidence object that those later experiments will evaluate and that the
frontend will let users investigate. The core contribution is the auditable
connection among speaker identity, time, overlap, retrieved terms, correction
decisions, and review flags.
