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

- 17 real public-data clips in `data/manifests/formal_eval_real.csv`;
- 17 existing `base` faster-whisper prediction JSON files from Phase 2C;
- generated single-speaker FLEURS anchors;
- reference multi-speaker anchors and five overlap events for two AMI clips;
- the local TalkWeaver glossary and deterministic constrained correction.

No ASR model was loaded or rerun. No mock prediction was accepted. Pyannote
was not required because this phase uses explicitly labeled reference
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

The run produced 119 rows: 17 real clips multiplied by seven variants. It
also produced 119 local ConversationMap JSON files.

| Variant | Anchors | Speaker-labeled | Overlap anchors | Events | Term candidates | Audits | Unsupported changes | Needs review |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `asr_only` | 17 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `temporal_anchor_only` | 19 | 0 | 0 | 0 | 0 | 0 | 0 | 19 |
| `reference_speaker_time` | 25 | 22 | 0 | 0 | 0 | 0 | 0 | 3 |
| `overlap_aware` | 25 | 23 | 4 | 5 | 0 | 0 | 0 | 6 |
| `term_rescue` | 25 | 23 | 4 | 5 | 0 | 0 | 0 | 6 |
| `constrained_correction` | 25 | 23 | 4 | 5 | 0 | 25 | 0 | 6 |
| `full_talkweaver` | 25 | 23 | 4 | 5 | 0 | 25 | 0 | 6 |

The full workflow exposed four ASR anchors intersecting the five AMI overlap
events. Six anchors were marked for review: three in each AMI excerpt. The
first AMI clip includes unknown or partial speaker coverage in addition to
overlap uncertainty; the second has three overlap anchors.

### Correction Accuracy

No correction was applied to the real public subset. Therefore corrected
error rates exactly equal the fixed ASR baseline:

| Dataset / language | Metric | ASR error | Corrected error |
| --- | --- | ---: | ---: |
| AMI Meeting Corpus / English | WER | 0.370536 | 0.370536 |
| Google FLEURS / English | WER | 0.154242 | 0.154242 |
| Google FLEURS / French | WER | 0.273839 | 0.273839 |
| Google FLEURS / Mandarin | CER | 0.089651 | 0.089651 |

WER and CER are reported separately and should not be combined into a single
cross-language accuracy claim.

This result shows conservative behavior rather than correction benefit. The
rule fallback produced an audit for every corrected-workflow anchor but
introduced zero unsupported changes and did not rewrite unrelated public
speech.

### Term Rescue

The public subset has empty reference term annotations and does not contain
the controlled TalkWeaver technical-error phrases. After tightening
phonetic-like retrieval to avoid sentence-wide false matches, the real run
produced:

- 0 term candidates;
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

## Interpretation

The ablation validates the engineering method:

1. fixed real ASR predictions can be reused without model reruns;
2. temporal anchors make missing speaker evidence visible;
3. reference speaker-time assigns most anchors to named speakers;
4. overlap-aware processing exposes five conversation events and review
   zones in AMI;
5. constrained correction creates an audit trail without forcing edits;
6. the full variant adds speaker cards and extractive meeting evidence.

It does not demonstrate WER/CER improvement or term recovery on this subset.
Those claims require clips with actual correction targets and human term
references.

## Limitations

- The formal subset is only 17 clips.
- Only two AMI excerpts contain real multi-speaker time evidence.
- Reference speaker-time is oracle evidence, not automatic diarization.
- Current public clips contain no annotated TalkWeaver technical terms.
- Correction used deterministic rule fallback, not an evaluated API LLM.
- Pyannote was not evaluated because `HF_TOKEN` is not configured.
- Human interruption labels and Mandarin meeting data are still absent.
- Generated ConversationMaps remain local and ignored by Git.

The next experimental step should add a controlled technical-term correction
set and run the RAG/correction safety ablation before frontend implementation.
