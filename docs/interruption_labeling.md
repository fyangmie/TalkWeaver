# Interruption Labeling

## Purpose

Overlap is a timing fact, but interruption is a discourse label. TalkWeaver
therefore does not treat rule-based floor-taking candidates as ground truth.
The new workflow generates candidate rows for human review.

## Commands

```bash
python scripts/generate_interruption_label_candidates.py \
  --manifest data/manifests/english_meeting_heldout_real.csv \
  --output data/reference/public/english_meeting_heldout/interruption_label_candidates.csv

python scripts/validate_interruption_labels.py \
  --labels data/reference/public/english_meeting_heldout/interruption_label_candidates.csv \
  --manifest data/manifests/english_meeting_heldout_real.csv

python experiments/evaluate_interruption_labels.py \
  --labels data/reference/public/english_meeting_heldout/interruption_label_candidates.csv \
  --output experiments/results/interruption_label_summary_heldout_real.csv
```

## Label Schema

Required columns:

```text
clip_id,start,end,interrupter,interrupted,label,annotator
```

Allowed labels:

```text
interruption
backchannel
overlap_only
uncertain
```

The generated file contains 10 candidates from the 24-clip AMI held-out subset.
After human review on 2026-06-21, all 10 candidate windows were confirmed as
event-level `interruption`.

Important scope note: the human annotator verified that an interruption event
occurred in each candidate window. The `interrupter` and `interrupted` anonymous
speaker-pair fields are inherited from AMI reference speaker timing and were not
independently voice-identified by the annotator.

`experiments/evaluate_interruption_labels.py` intentionally does not fabricate
recall or F1. With the current reviewed file it reports candidate precision over
the 10 reviewed candidates:

```text
total_candidates=10
reviewed_candidates=10
candidate_precision=1.0
recall=
f1=
```

Recall/F1 require either exhaustive timeline labels or sampled non-candidate
negatives.
