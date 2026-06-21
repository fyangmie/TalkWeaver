# Speaker-Time and Overlap Baseline

## Purpose

Phase 2D establishes the speaker/time evidence layer required before
TalkWeaver workflow ablation. It tests whether the project can load reference
speaker turns, detect timing events, score predictions against reference
evidence, and keep oracle/reference-assisted results separate from automatic
diarization.

This is a small-subset project baseline. It is not a full DER evaluation and
does not claim automatic diarization accuracy when reference turns are used.

## Data Scope

The formal manifest contains 50 real public-data clips:

- 30 Google FLEURS clips with generated single-speaker anchors;
- 8 AMI Meeting Corpus excerpts with multi-speaker temporal anchors;
- 12 AISHELL-4 Mandarin meeting excerpts with TextGrid-derived temporal
  anchors;
- derived overlap intervals across the eight AMI excerpts. The selected
  AISHELL-4 windows currently contain no reference overlap events.

The current manifest has no human-labeled interruption events. Interruption
precision, recall, and F1 are therefore left blank rather than treating an
empty reference set as a validated perfect score.

## Modes

### `no_diarization`

The entire clip is represented by one `UNKNOWN` speaker turn and no events.
This is a naive baseline. Its time coverage can be high because the unknown
turn spans the clip, but its speaker-label error remains high on labeled
speech.

### `reference_assisted`

Reference speaker turns are loaded from `anchors_path`, then TalkWeaver's
rule-based event detector runs over those turns. This is an oracle
speaker/time workflow sanity check, not automatic diarization performance.

### `pyannote_optional`

Real pyannote inference runs only when `pyannote.audio` is installed and
`HF_TOKEN` is configured. Missing dependencies or model access produce an
explicit skipped row. The runner never substitutes mock diarization.

## Metrics

- **Turn time coverage:** fraction of reference speech-union time intersected
  by any predicted speaker turn.
- **Speaker label error rate:** duration-weighted speaker attribution error
  after a best one-to-one label permutation. This avoids penalizing arbitrary
  names such as `SPEAKER_00` versus `A`.
- **Boundary MAE:** mean absolute start/end difference for greedily matched
  speaker turns.
- **Overlap precision/recall/F1:** interval matching with IoU at least 0.3.
- **Interruption precision/recall/F1:** temporal overlap plus matching speaker
  pair when both references provide a pair.

The speaker-label metric is deliberately simplified. It does not implement
DER collars, false-alarm decomposition, missed-speech decomposition, or the
full diarization evaluation protocol.

## Measured Results

Command:

```bash
python experiments/run_speaker_overlap_baseline.py \
  --manifest data/manifests/formal_eval_real.csv \
  --output experiments/results/speaker_overlap_baseline_real.csv
```

AMI multi-speaker results:

| Mode | Clips | Speaker label error | Time coverage | Boundary MAE | Overlap P/R/F1 |
| --- | ---: | ---: | ---: | ---: | --- |
| `no_diarization` | 8 | 1.000 | 1.000 | N/A | 0.000 / 0.000 / 0.000 |
| `reference_assisted` | 8 | 0.000 | 1.000 | 0.000 s | 0.986 / 0.975 / 0.980 |
| `pyannote_optional` | 8 | 0.109 | 0.968 | 0.375 s | 0.646 / 0.619 / 0.604 |

The reference-assisted scores are expected because the same reference turns
are the input evidence. They validate loading, event generation, metric
plumbing, and output contracts only. They are an upper-bound workflow sanity
check, not evidence of an automatic model reaching perfect diarization.

`pyannote_optional` now runs because `HF_TOKEN` is configured. It is still a
small local run, not a full AMI benchmark.

AISHELL-4 Mandarin meeting sanity results:

| Mode | Clips | Speaker label error | Time coverage | Boundary MAE | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| `no_diarization` | 12 | 1.000 | 1.000 | N/A | one `UNKNOWN` speaker |
| `reference_assisted` | 12 | 0.000 | 1.000 | 0.000 s | oracle reference turns |
| `pyannote_optional` | 12 | 0.053 | 0.983 | 2.752 s | no reference overlap events in selected windows |

FLEURS is single-speaker read speech and should not be interpreted as a
meeting diarization result. On FLEURS, pyannote over-segments some clips, so
the single-speaker speaker-label error is diagnostic only.

Interruption metrics are not reported because the current event references
contain overlap labels only. Rule-based interruption candidates remain
available in ConversationMap output for review, but cannot yet be scored
against human interruption labels.

## Reference-Assisted ConversationMaps

```bash
python experiments/run_reference_workflow_maps.py \
  --manifest data/manifests/formal_eval_real.csv \
  --dataset "AMI Meeting Corpus" \
  --output-dir outputs/conversation_maps/reference_assisted_real
```

The command generated two local ConversationMaps:

```text
outputs/conversation_maps/reference_assisted_real/
├── ami_es2002a_01_conversation_map.json
├── ami_es2002a_02_conversation_map.json
└── ...
```

Both use the existing Phase 2C real `base` ASR prediction JSON and reference
speaker/time evidence. Their metadata records:

```text
asr_mode=real_prediction_json
diarization_mode=reference
evaluation_scope=small_subset
```

Generated map files remain ignored by Git by default.

## Limitations

- Only eight AMI excerpts from one meeting and 12 AISHELL-4 excerpts from one
  Mandarin meeting recording currently have real multi-speaker time labels.
- No human interruption reference exists in the current subset.
- Reference-assisted scores cannot be interpreted as automatic diarization
  accuracy.
- pyannote requires accepted model access and an `HF_TOKEN`; the current run
  is a small local diagnostic.
- The simple label-mapped metric is useful for project diagnostics but is not
  a substitute for standard DER tooling.

The next experiment should reuse the frozen Phase 2C prediction JSON and this
speaker/time evidence layer for the TalkWeaver workflow ablation.
