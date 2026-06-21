# TalkWeaver Workflow Ablation

## Purpose

Phase 2E evaluates the TalkWeaver method over fixed Phase 2C real ASR
predictions. It asks what evidence and audit structures are added as the
system moves from a plain transcript to an AI Meeting Detective
`ConversationMap`.

This is not another ASR model comparison. ASR-only output cannot represent
who spoke, speaker timing, overlap, review uncertainty, retrieved evidence,
or correction provenance. The ablation holds the base-model ASR prediction
constant and changes only the downstream workflow.

## Inputs

- 50 real public-data clips in `data/manifests/formal_eval_real.csv`;
- 50 existing `base` faster-whisper prediction JSON files from Phase 2C;
- generated single-speaker anchors for 30 FLEURS clips;
- reference multi-speaker anchors and derived overlap events for eight AMI
  clips;
- TextGrid-derived speaker-time anchors for 12 AISHELL-4 Mandarin meeting
  clips;
- the local TalkWeaver glossary and deterministic constrained correction.

No ASR model was loaded or rerun. No mock prediction was accepted. Pyannote
was not required because this phase uses explicitly labeled reference
speaker-time evidence.

A second automatic workflow run now uses the 24-clip AMI held-out set, fixed
`base` ASR predictions, and automatic pyannote turns instead of reference
speaker-time evidence.

## Variants

| Variant | Added capability |
| --- | --- |
| `asr_only` | One clip-level raw transcript span |
| `temporal_anchor_only` | Word/segment timing grouped into anchors with unknown speaker |
| `reference_speaker_time` | Oracle/reference speaker-time attribution without overlap semantics |
| `overlap_aware` | Overlap and conservative interruption events plus review flags |
| `term_rescue` | Glossary, fuzzy, and conservative phonetic-like candidates without text replacement |
| `constrained_correction` | Per-anchor deterministic correction and `CorrectionAudit` |
| `full_talkweaver` | Correction evidence, speaker cards, and extractive summary in a complete `ConversationMap` |

Reference speaker-time is an oracle-assisted input. It must not be described
as automatic diarization performance.

## Commands

```bash
python experiments/run_workflow_ablation.py \
  --manifest data/manifests/formal_eval_real.csv \
  --predictions-dir experiments/results/asr_predictions_real \
  --asr-model base \
  --output experiments/results/workflow_ablation_real.csv \
  --maps-dir outputs/conversation_maps/ablation_real \
  --variants all

python experiments/summarize_workflow_ablation.py \
  --input experiments/results/workflow_ablation_real.csv \
  --output experiments/results/workflow_ablation_summary_real.csv

python experiments/plot_workflow_ablation.py \
  --input experiments/results/workflow_ablation_real.csv \
  --output-dir assets/result_charts
```

Use `--dataset "AMI Meeting Corpus"` or `--max-clips 5` for smaller
diagnostic runs.

## Metrics

- anchor, speaker-labeled anchor, overlap-anchor, and event counts;
- term candidates and corrections actually applied;
- correction audits, unsupported inserted tokens, and review flags;
- temporal anchor coverage over ASR segment time;
- original and corrected WER/CER using the Phase 2C normalization policy;
- evidence-availability and module-use flags.

`anchor_coverage` measures timed ASR span covered by emitted anchors. The
word-timestamp variants average slightly below 1.0 because inter-word gaps
and segment padding are not represented as speech anchors.

Term precision and recall are left blank when `reference_terms.json` is
empty. An empty term reference is not scored as perfect retrieval.

## Real Small-Subset Results

The run produced 350 rows: 50 real clips multiplied by seven variants. It
also produced 350 local ConversationMap JSON files.

AMI per-clip means show how the workflow adds meeting evidence:

| Variant | Anchors | Speaker-labeled | Overlap anchors | Events | Term candidates | Audits | Unsupported changes | Needs review |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `asr_only` | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `temporal_anchor_only` | 1.875 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.875 |
| `reference_speaker_time` | 4.500 | 3.500 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| `overlap_aware` | 4.500 | 3.625 | 1.250 | 3.750 | 0.000 | 0.000 | 0.000 | 2.125 |
| `term_rescue` | 4.500 | 3.625 | 1.250 | 3.750 | 0.375 | 0.000 | 0.000 | 2.125 |
| `constrained_correction` | 4.500 | 3.625 | 1.250 | 3.750 | 0.375 | 4.500 | 0.000 | 2.125 |
| `full_talkweaver` | 4.500 | 3.625 | 1.250 | 3.750 | 0.375 | 4.500 | 0.000 | 2.125 |

FLEURS remains single-speaker read speech in this workflow: each clip has one
anchor, no overlap anchors, and no conversation events. This is expected and
is why AMI is the meaningful source for the meeting-evidence line.

AISHELL-4 adds Mandarin meeting anchors but the selected 12 windows currently
have no reference overlap events:

| Variant | Anchors | Speaker-labeled | Overlap anchors | Events | Needs review |
| --- | ---: | ---: | ---: | ---: | ---: |
| `asr_only` | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `temporal_anchor_only` | 2.167 | 0.000 | 0.000 | 0.000 | 2.167 |
| `reference_speaker_time` | 2.833 | 2.583 | 0.000 | 0.000 | 0.250 |
| `full_talkweaver` | 2.833 | 2.583 | 0.000 | 0.000 | 0.250 |

### Correction Accuracy

No correction was applied to the real public subset. Therefore corrected
error rates exactly equal the fixed ASR baseline:

| Dataset / language | Metric | ASR error | Corrected error |
| --- | --- | ---: | ---: |
| AMI Meeting Corpus / English | WER | 0.398364 | 0.398364 |
| AISHELL-4 / Mandarin | CER | 0.609966 | 0.609966 |
| Google FLEURS / English | WER | 0.114374 | 0.114374 |
| Google FLEURS / French | WER | 0.227136 | 0.227136 |
| Google FLEURS / Mandarin | CER | 0.113336 | 0.113336 |

WER and CER are reported separately and should not be combined into a single
cross-language accuracy claim.

This result shows conservative behavior rather than correction benefit. The
rule fallback produced an audit for every corrected-workflow anchor but
introduced zero unsupported changes and did not rewrite unrelated public
speech.

### Term Rescue

The public subset has empty reference term annotations and does not contain
the controlled TalkWeaver technical-error phrases. The expanded real run
produced a small number of candidates but applied no rescues:

- 0.375 term candidates per AMI clip on average;
- 0.1 term candidates per English and French FLEURS clip on average;
- 0 applied rescues;
- no term precision or recall value.

Term recovery benefit must be evaluated later on consent-safe or synthetic
clips containing controlled errors such as `piano note -> pyannote` and
`diary station -> diarization`.

## Charts

- `assets/result_charts/workflow_ablation_completeness.png`
- `assets/result_charts/workflow_ablation_review_flags.png`

The first chart shows evidence structure growing across variants. The second
shows that review flags appear when unknown speaker and overlap evidence
become explicit, while unsupported correction count remains zero.

## Automatic Pyannote Evidence Maps

The automatic workflow is generated with:

```bash
python experiments/run_automatic_pyannote_workflow.py \
  --manifest data/manifests/english_meeting_heldout_real.csv \
  --asr-predictions-dir experiments/results/asr_predictions_english_meeting_heldout_real \
  --pyannote-predictions-dir outputs/diarization/pyannote_heldout_real \
  --asr-model base \
  --output experiments/results/automatic_pyannote_workflow_heldout_real.csv \
  --summary-output experiments/results/automatic_pyannote_workflow_heldout_summary_real.csv \
  --maps-dir outputs/exports/automatic_pyannote_workflow_heldout_real
```

Current summary:

| Clips | Pyannote turns | Anchors | Speaker-labeled anchors | Overlap anchors | Events | Needs review |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 24 | 4.667 | 4.750 | 4.250 | 1.792 | 2.042 | 2.292 |

This is the automatic counterpart to the reference-assisted ablation: it uses
pyannote turns and detected overlap/interruption events, not oracle
speaker-time turns, to build the ConversationMap.

## Interpretation

The ablation validates the engineering method:

1. fixed real ASR predictions can be reused without model reruns;
2. temporal anchors make missing speaker evidence visible;
3. reference speaker-time assigns most anchors to named speakers;
4. overlap-aware processing exposes conversation events and review zones in
   AMI;
5. constrained correction creates an audit trail without forcing edits;
6. the full variant adds speaker cards and extractive meeting evidence.

It does not demonstrate WER/CER improvement or term recovery on this subset.
Those claims require clips with actual correction targets and human term
references.

## Limitations

- The formal subset is only 50 clips.
- The original formal subset has only eight AMI excerpts; the larger held-out
  workflow uses 24 AMI clips from four recordings. AISHELL-4 adds 12 Mandarin
  meeting clips but currently from one recording prefix only.
- Reference speaker-time is oracle evidence, not automatic diarization.
- Current public clips contain no annotated TalkWeaver technical terms.
- Correction used deterministic rule fallback, not an evaluated API LLM.
- Pyannote is now evaluated on 24 held-out clips, but the automatic maps still
  need human interruption labels for discourse-level interruption claims.
- Human interruption labels remain incomplete; Mandarin meeting data is now
  present but still small.
- Generated ConversationMaps remain local and ignored by Git.

The next experimental step should add broader interruption labels, a larger
Mandarin meeting split, and true phone-side whisper.cpp/mobile latency
results.
