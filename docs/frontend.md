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
anchor. When a public clip mostly retains its raw ASR text, the page explains
that the clip has no annotated technical-term target and that conservative
retention is an intentional safety result.

### Speaker Timeline Detective

Plots temporal anchors on speaker lanes. Overlap intervals are shaded and
review anchors receive a visible outline. The table can be filtered by
speaker, overlap, and review state, then inspected as raw-versus-corrected
evidence. The anchor inspector uses a token-level diff and explicitly labels
identical text as retained evidence rather than a missing UI state. The
**Jump to event** selector highlights linked anchors and starts local audio at
the padded event window.

### Speaker Evidence Cards

Builds one evidence card per named speaker. Each card reports speaking time,
anchor count, overlap count, review burden, top extractive terms,
representative raw/corrected quotes, claims, action items, and source anchor
IDs. Current AMI cards use extractive fallback. LLM-assisted cards must be
explicitly labeled and retain anchor links.

### Cross-talk and Overlap Warning

Combines selected-map overlap events with the controlled Phase 2G safety
summary and charts. The page emphasizes conservative rejection instead of
fluent completion of uncertain cross-talk. Four controlled case files show:

- mild overlap where supported correction is allowed but review remains;
- heavy overlap where correction is rejected;
- ambiguous speaker attribution where words are not reassigned;
- physical `rack` context where `RAG` replacement is blocked.

The Event Investigation section filters real ConversationMap events by type,
review state, and speaker. It links each event to raw/corrected anchors and a
local audio window.

### Misheard Word Rescue

Shows controlled Phase 2F glossary, fuzzy, phonetic-like, fused, rule, and
LLM results. It includes retrieval candidates, corrected text, expected
terms, false positives, and review flags. The **Rescue Case Files** selector
visually demonstrates:

- `piano note -> pyannote`;
- `diary station -> speaker diarization`;
- contextual `rack -> RAG`;
- contextual `where -> WER`;
- `temporal anger -> temporal anchor`.

These are controlled technical-term fixtures, not real audio.

### Hallucination Watchdog

Displays `CorrectionAudit` records from the selected map and controlled
examples of accepted supported edits, rejected risky edits, and preserved
common-word negative controls. Each case exposes raw text, corrected or
retained text, unsupported changes, review/rejection status, API use,
fallback use, and the recorded reason.

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

## Local Audio Evidence

ConversationMaps retain the manifest `audio_path`, but raw public/private
audio is ignored by Git. `resolve_local_audio_path()` permits playback only
when that path resolves to an existing file inside the repository.

When available:

- Conversation Crime Scene shows the full clip;
- Speaker Timeline plays the selected anchor or event window;
- Cross-talk Warning plays the selected event window;
- Speaker Evidence Cards play the selected quote window.

Streamlit uses `start_time` and `end_time` for bounded playback. Every view
also prints the approximate time range because browser seek behavior may
vary. Missing audio produces a normal informational notice and never causes a
test or page failure.

## Overlap And Interruption

- **Overlap:** two or more speakers are active at the same time.
- **Interruption candidate:** one speaker begins before another finishes and
  may create floor-taking or cross-talk.

The current AMI subset contains reference overlap labels. It has limited
human interruption labels, so timing-derived interruption candidates must
not be reported as human ground truth. Reference AMI speaker timing remains
oracle-assisted, not automatic diarization.

## Data Files

The frontend data layer is `webapp/data_loader.py`.

| Artifact | Path | Claim type |
| --- | --- | --- |
| ConversationMaps | `outputs/conversation_maps/` | Per-clip workflow evidence; metadata defines mock, real, or reference-assisted mode |
| ASR summary | `experiments/results/asr_benchmark_summary_real.csv` | Small real public subset |
| Speaker/overlap baseline | `experiments/results/speaker_overlap_baseline_real.csv` | Real references; reference-assisted rows are oracle workflow checks |
| Workflow ablation | `experiments/results/workflow_ablation_real.csv` | Fixed real ASR predictions with explicit evidence flags |
| Term rescue cases | `experiments/results/term_rescue_controlled.csv` | Controlled authored correction and retrieval cases |
| Term rescue summary | `experiments/results/term_rescue_summary_controlled.csv` | Controlled authored text fixtures |
| Overlap safety cases | `experiments/results/overlap_safety_controlled.csv` | Controlled correction/rejection cases |
| Overlap safety summary | `experiments/results/overlap_safety_summary_controlled.csv` | Controlled authored text fixtures |
| Charts | `assets/result_charts/` | Derived from the corresponding CSV scope |

## Why Public Clips May Show No Correction

The current AMI and FLEURS subset was selected for real ASR, multilingual,
speaker-time, and overlap evaluation. It does not contain annotated
TalkWeaver technical-term targets. In the real workflow ablation, correction
therefore preserved the fixed ASR predictions instead of forcing edits.

Identical `raw_text` and `corrected_text` on those clips is an auditable
conservative result. It must not be replaced with a fabricated real-audio
correction. Demonstrable correction behavior comes from the separate Phase
2F and Phase 2G controlled fixtures.

## Correction Demo Flow

For a final-video correction demonstration:

1. Open **Speaker Timeline Detective** and show that the selected AMI public
   map retains raw evidence while exposing overlap and review flags.
2. Open **Misheard Word Rescue** and select a controlled Rescue Case File.
   The token diff highlights the misheard phrase and rescued canonical term.
3. Open **Hallucination Watchdog** and compare an accepted correction, a
   strict LLM rejection, and an unchanged negative control.
4. Open **Cross-talk and Overlap Warning** and contrast mild correction with
   heavy-overlap rejection.
5. Select a real AMI event, play its bounded audio window, and inspect linked
   anchors.
6. Open **Speaker Evidence Cards** and verify an extractive statement against
   its timestamped quote.

The controlled fixture label remains visible throughout this flow.

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
- Event-level playback requires the ignored local audio file and browser
  support for seek offsets.
- Speaker cards remain extractive fallback evidence unless an LLM-assisted
  card is explicitly labeled; unsupported personality, emotion, or intent
  inference is prohibited.
- Multilingual UI comparison and the mandatory whisper.cpp mobile trade-off
  page remain future phases.

## Next Steps

1. Add a dedicated interruption map once human interruption labels exist.
2. Add multilingual comparison and Level 1 mobile ASR trade-off views.
3. Capture desktop and mobile screenshots for the final video.
