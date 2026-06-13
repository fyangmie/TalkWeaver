# AI Meeting Detective Frontend

## Purpose

The Phase 3A Streamlit frontend turns existing TalkWeaver artifacts into an
investigation workspace. It does not run ASR, diarization, or an LLM. It reads
local `ConversationMap` JSON files, measured experiment CSVs, controlled
safety CSVs, and generated charts.

The product story is evidence-first:

- who said what;
- when each speaker was active;
- where speech overlapped;
- which terms were proposed for rescue;
- what correction changed;
- which changes were rejected or marked for review.

## Run

Install the core requirements, then start the v1 application:

```bash
pip install -r requirements.txt
streamlit run webapp/app.py
```

The existing v0 multipage review workspace remains available:

```bash
streamlit run webapp/streamlit_app.py
```

No API key, GPU, `HF_TOKEN`, pyannote model, or raw audio is required to open
the v1 frontend. Missing artifacts produce an actionable warning rather than
a model fallback.

## Pages

### Home / Project Story

Introduces the AI Meeting Detective workflow and separates real public data,
reference-assisted evidence, and controlled fixtures.

### Conversation Crime Scene

Shows case metadata, evidence modes, anchor and speaker counts, overlap
events, review flags, unsupported changes, and the most uncertain temporal
anchor.

### Speaker Timeline Detective

Plots temporal anchors on speaker lanes. Overlap intervals are shaded and
review anchors receive a visible outline. The table can be filtered by
speaker, overlap, and review state, then inspected as raw-versus-corrected
evidence.

### Cross-talk and Overlap Warning

Combines selected-map overlap events with the controlled Phase 2G safety
summary and charts. The page emphasizes conservative rejection instead of
fluent completion of uncertain cross-talk.

### Misheard Word Rescue

Shows controlled Phase 2F glossary, fuzzy, phonetic-like, fused, rule, and
LLM results. It includes retrieval candidates, corrected text, expected
terms, false positives, and review flags.

### Hallucination Watchdog

Displays `CorrectionAudit` records from the selected map and controlled
examples of accepted supported edits, rejected risky edits, and preserved
common-word negative controls.

### Evidence Dashboard

Displays:

- real ASR summary and charts;
- speaker-time/overlap baseline;
- workflow ablation;
- controlled term rescue charts;
- controlled overlap safety charts.

WER and CER, public and controlled evidence, and oracle versus automatic
speaker evidence remain visibly separated.

### Export / Report Preview

Builds a Markdown detective report containing case metadata, temporal
anchors, event warnings, review flags, term candidates, correction-audit
summary, and related chart paths. Local exports are written to:

```text
outputs/reports/<clip_id>_detective_report.md
```

Generated reports remain ignored by Git by default.

## Data Files

The frontend data layer is `webapp/data_loader.py`.

| Artifact | Path | Claim type |
| --- | --- | --- |
| ConversationMaps | `outputs/conversation_maps/` | Per-clip workflow evidence; metadata defines mock, real, or reference-assisted mode |
| ASR summary | `experiments/results/asr_benchmark_summary_real.csv` | Small real public subset |
| Speaker/overlap baseline | `experiments/results/speaker_overlap_baseline_real.csv` | Real references; reference-assisted rows are oracle workflow checks |
| Workflow ablation | `experiments/results/workflow_ablation_real.csv` | Fixed real ASR predictions with explicit evidence flags |
| Term rescue summary | `experiments/results/term_rescue_summary_controlled.csv` | Controlled authored text fixtures |
| Overlap safety summary | `experiments/results/overlap_safety_summary_controlled.csv` | Controlled authored text fixtures |
| Charts | `assets/result_charts/` | Derived from the corresponding CSV scope |

## Missing Data Behavior

Loaders return empty DataFrames with a warning attribute or a ConversationMap
warning object. The UI does not fabricate a replacement artifact and does not
silently run mock models. A generated ConversationMap can be created with the
workflow or reference-map scripts documented in
[`talkweaver_workflow.md`](talkweaver_workflow.md).

## Current Limitations

- The formal evaluation subset contains only 17 clips.
- Only two AMI excerpts provide multi-speaker reference timing.
- Reference speaker-time is oracle-assisted, not automatic diarization.
- Controlled term and overlap fixtures are text safety tests, not acoustic
  generalization evidence.
- Event-level audio seeking is not implemented yet.
- Speaker stance cards remain extractive fallback evidence; unsupported
  personality, emotion, or intent inference is prohibited.
- Multilingual UI comparison and the mandatory whisper.cpp mobile trade-off
  page remain future phases.

## Next Steps

1. Add linked event-to-audio playback.
2. Add a dedicated interruption map once human interruption labels exist.
3. Add evidence-linked speaker stance and action-item cards.
4. Add multilingual comparison and Level 1 mobile ASR trade-off views.
5. Capture desktop and mobile screenshots for the final video.
