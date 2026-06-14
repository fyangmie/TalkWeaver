# TalkWeaver EvidenceGate

## Purpose

EvidenceGate is TalkWeaver's trained lightweight model component. It decides
whether a proposed transcript correction should be:

- `accept`;
- `reject`;
- `needs_review`.

It sits after evidence collection and correction proposal:

```text
ASR text
-> temporal / speaker / overlap evidence
-> glossary retrieval
-> rule or LLM correction proposal
-> EvidenceGate accept / reject / needs_review
-> ConversationMap audit trail
```

EvidenceGate does not transcribe audio, perform diarization, retrieve terms,
or generate corrected prose. It does not fine-tune faster-whisper or
DeepSeek. It learns a small decision boundary over the audit evidence emitted
by those components.

## Claim Boundary

This is a **controlled / semi-synthetic correction-safety experiment**.

The seed rows come from:

- Phase 2F controlled technical-term rescue;
- Phase 2G controlled overlap correction safety.

The augmented rows are deterministic rule-generated perturbations. No raw
audio, API call, private transcript, ASR training, or LLM training is used.
The results measure policy distillation on this controlled task. They do not
establish real-world audio generalization.

## Dataset

The normalized seed dataset contains 255 correction decisions:

| Source | Rows |
| --- | ---: |
| Phase 2F term rescue | 175 |
| Phase 2G overlap safety | 80 |
| Total | 255 |

The label policy produced:

| Label | Original rows |
| --- | ---: |
| accept | 160 |
| reject | 51 |
| needs_review | 44 |

Twenty-three seed rows are logged as ambiguous policy cases. They are routed
to review rather than silently treated as safe.

Transparent augmentation adds six perturbations per source case group:

- safe preservation;
- supported correction;
- invented unsupported claim;
- forbidden speaker reassignment;
- heavy-overlap review;
- ambiguous mild-overlap review.

This adds 270 rows for a total of 525:

| Label | Augmented dataset rows |
| --- | ---: |
| accept | 250 |
| reject | 141 |
| needs_review | 134 |

Every row records `is_augmented` and `augmentation_type`. The process is
implemented in `experiments/augment_evidence_gate_examples.py`; it is not a
hidden generative-data step.

## Label Policy

The policy is conservative:

1. Explicit rejection, invented content, changed speaker attribution,
   forbidden changes, unsupported changes, strict correction validation
   failure, or safety-policy failure maps to `reject`.
2. An unresolved audit review flag maps to `needs_review`.
3. A supported correction that does not worsen reference text error maps to
   `accept`.
4. An unchanged negative control maps to `accept`.
5. An unresolved expected term maps to `needs_review`.

The policy and reason are stored as `expected_label` and `label_reason`.

## Features

EvidenceGate uses numeric evidence already available at correction-decision
time:

- retrieval, expected-term, true-positive, false-positive, and missed-term
  counts;
- term precision, recall, and F1;
- text error before/after and error delta;
- changed-token count and ratio;
- edit-distance ratio;
- overlap, heavy-overlap, and uncertainty indicators;
- upstream review and rejection flags;
- unsupported, forbidden, invented-content, and speaker-change indicators;
- API, LLM-variant, rule-variant, and negative-control indicators;
- a deterministic context-risk score.

The strongest features in the current gradient-boosting model are:

1. `correction_rejected_input_flag`;
2. `uncertainty_score`;
3. `missed_term_count`;
4. `context_risk_score`;
5. `needs_review_input_flag`.

These are intentionally audit-oriented. Because the labels are derived from
related safety rules, the current task is closer to **policy distillation**
than independent semantic safety discovery. This explains why controlled
scores are high and is a major limitation, not a result to hide.

## Leakage Prevention

Each seed case receives:

```text
template_group = source_experiment + ":" + case_id
```

All variants and augmented rows from one template group remain in exactly one
split. The random-seed-42 group split is:

| Split | Rows | Template groups |
| --- | ---: | ---: |
| train | 361 | 31 |
| validation | 63 | 6 |
| test | 101 | 8 |

No `template_group` occurs in more than one split. This prevents direct
case-variant leakage. It does not remove broader structural similarity from
rule-based augmentation, so external validation remains necessary.

## Models

The experiment trains:

- class-balanced logistic regression;
- class-balanced random forest;
- gradient boosting.

The resulting joblib files are approximately 4 KB, 696 KB, and 372 KB. They
are small enough to keep with the project and can also be regenerated from
the committed controlled datasets.

## Controlled Results

All three trained classifiers produced the same controlled test result:

| Model | Macro F1 | False accept | Unsafe accept | Review recall | Reject recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic regression | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 |
| Random forest | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 |
| Gradient boosting | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 |

Ties are reported as ties. The frontend selects gradient boosting
deterministically by model name only after macro F1 and unsafe-accept rate
tie.

### Baselines

| Baseline | Macro F1 | False accept | Unsafe accept | Review recall | Reject recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Rule policy baseline | 0.928 | 0.156 | 0.053 | 0.769 | 0.947 |
| Raw LLM-variant signal | 0.331 | 0.933 | 0.895 | 0.038 | 0.105 |
| Always accept | 0.238 | 1.000 | 1.000 | 0.000 | 0.000 |
| Always review | 0.136 | 0.000 | 0.000 | 1.000 | 0.000 |

`always_review` has zero unsafe accepts but is operationally useless because
it rejects automation entirely. The rule baseline remains strong and
interpretable. EvidenceGate's practical value must therefore be tested on
future independently human-labeled proposals, not inferred from this
controlled score alone.

## Metrics

- **Accuracy:** fraction of all decisions classified correctly.
- **Macro F1:** equal-weight mean F1 across accept, reject, and needs-review.
- **False accept rate:** non-accept ground-truth rows predicted accept.
- **Unsafe accept rate:** reject ground-truth rows predicted accept.
- **Reject recall:** rejected rows correctly rejected.
- **Needs-review recall:** review rows correctly routed to review.
- **Accept precision:** predicted accepts that are truly accepted.

False and unsafe accepts receive special attention because accepting an
unsupported fluent correction is the failure mode TalkWeaver is designed to
expose.

## Reproduce

```bash
python experiments/build_evidence_gate_dataset.py \
  --term-input experiments/results/term_rescue_controlled.csv \
  --overlap-input experiments/results/overlap_safety_controlled.csv \
  --output data/controlled_evidence_gate/evidence_gate_examples.csv

python experiments/augment_evidence_gate_examples.py \
  --input data/controlled_evidence_gate/evidence_gate_examples.csv \
  --output data/controlled_evidence_gate/evidence_gate_examples_augmented.csv

python experiments/train_evidence_gate.py \
  --input data/controlled_evidence_gate/evidence_gate_examples_augmented.csv \
  --output-dir experiments/results/evidence_gate \
  --models logistic_regression random_forest gradient_boosting \
  --group-split-column template_group \
  --random-seed 42

python experiments/evaluate_evidence_gate.py \
  --predictions experiments/results/evidence_gate/evidence_gate_predictions.csv \
  --output experiments/results/evidence_gate/evidence_gate_eval_summary.csv

python experiments/plot_evidence_gate.py \
  --metrics experiments/results/evidence_gate/evidence_gate_metrics.csv \
  --predictions experiments/results/evidence_gate/evidence_gate_predictions.csv \
  --feature-importance experiments/results/evidence_gate/evidence_gate_feature_importance.csv \
  --output-dir assets/result_charts
```

## Artifacts

```text
data/controlled_evidence_gate/evidence_gate_examples.csv
data/controlled_evidence_gate/evidence_gate_examples_augmented.csv
experiments/results/evidence_gate/evidence_gate_predictions.csv
experiments/results/evidence_gate/evidence_gate_metrics.csv
experiments/results/evidence_gate/evidence_gate_eval_summary.csv
experiments/results/evidence_gate/evidence_gate_feature_importance.csv
experiments/results/evidence_gate/evidence_gate_split_summary.csv
models/evidence_gate/evidence_gate_logistic_regression.joblib
models/evidence_gate/evidence_gate_random_forest.joblib
models/evidence_gate/evidence_gate_gradient_boosting.joblib
assets/result_charts/evidence_gate_*.png
```

## Frontend

The **EvidenceGate Model** page shows:

- model and baseline comparison;
- macro F1 and unsafe-accept metrics;
- group-aware split sizes;
- confusion matrix;
- feature importance;
- class recall;
- accepted, rejected, and needs-review case files.

The Evidence Dashboard includes the same charts, and detective report exports
include the best available controlled EvidenceGate summary.

## Limitations And Next Validation

- Seed data is controlled authored text, not raw-audio correction decisions.
- Augmentation is deterministic and structurally simpler than real failures.
- Labels and features share explicit audit-policy signals.
- The original seed set is small.
- No ASR or LLM backbone is fine-tuned.
- There is no independent human-adjudicated external test set yet.
- Perfect controlled scores must not be described as real deployment
  performance.

The next meaningful EvidenceGate step is to collect blind, human-labeled
correction proposals from held-out multilingual and overlap audio, then test
the frozen model without changing its label policy or thresholds.
