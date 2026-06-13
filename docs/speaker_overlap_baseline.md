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

The formal manifest contains 17 real public-data clips:

- 15 Google FLEURS clips with generated single-speaker anchors;
- 2 AMI Meeting Corpus excerpts with multi-speaker temporal anchors;
- 5 human/reference overlap intervals across the two AMI excerpts.

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
| `no_diarization` | 2 | 1.000 | 1.000 | N/A | 0.000 / 0.000 / 0.000 |
| `reference_assisted` | 2 | 0.000 | 1.000 | 0.000 s | 1.000 / 1.000 / 1.000 |
| `pyannote_optional` | 0 run | N/A | N/A | N/A | N/A |

The reference-assisted scores are expected because the same reference turns
are the input evidence. They validate loading, event generation, metric
plumbing, and output contracts only. They are an upper-bound workflow sanity
check, not evidence of an automatic model reaching perfect diarization.

`pyannote_optional` was skipped for all 17 clips because `HF_TOKEN` was not
configured. The Python package was present, but model access was not assumed.

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
└── ami_es2002a_02_conversation_map.json
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

- Only two AMI excerpts currently have real multi-speaker time labels.
- No Mandarin meeting sample is locally available yet.
- No human interruption reference exists in the current subset.
- Reference-assisted scores cannot be interpreted as automatic diarization
  accuracy.
- pyannote requires accepted model access and an `HF_TOKEN`; it was skipped.
- The simple label-mapped metric is useful for project diagnostics but is not
  a substitute for standard DER tooling.

The next experiment should reuse the frozen Phase 2C prediction JSON and this
speaker/time evidence layer for the TalkWeaver workflow ablation.
