# TalkWeaver EvidenceGate

> **Leakage warning:** The initial EvidenceGate result is a
> policy-distillation sanity check. Perfect scores likely reflect label-proxy
> features and should not be interpreted as real-world generalization.

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
The results measure policy distillation and controlled risk prediction. They
do not establish real-world audio generalization. The final report must not
use the initial `1.000` scores as a primary model-performance claim.

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

## Three Feature Sets

### Audit-aware

Uses every original feature, including reference-derived correctness values
and final audit outcomes. It answers only:

> Can a classifier reproduce the controlled safety policy when given fields
> that substantially encode that policy?

This is the **policy-distillation sanity check**. It is not a deployable
pre-decision gate and is expected to score optimistically.

### Evidence-only

Excludes reference-derived correctness and final audit decisions. It uses:

- retrieval candidate and proposed correction counts;
- changed-token and edit-distance magnitude;
- overlap, heavy-overlap, and uncertainty;
- a pre-decision context-risk score;
- API/rule/LLM metadata;
- language and source-experiment indicators.

### Risk-only

Uses the strictest proposal-time subset:

- changed-token count and ratio;
- edit-distance ratio;
- overlap and heavy-overlap;
- uncertainty and pre-decision context risk;
- API/rule/LLM metadata;
- language.

Neither strict feature set sees `needs_review`, `correction_rejected`,
`safety_pass`, unsupported changes, forbidden changes, invented content,
speaker-attribution outcomes, reference term scores, or post-correction text
error.

## Feature Leakage Audit

The audit classifies 41 model features and related dataset fields:

| Category | Count |
| --- | ---: |
| Allowed pre-decision | 19 |
| Risky reference-derived | 4 |
| Direct label proxy | 14 |
| Final audit outcome | 4 |

Flagged fields include:

- `safety_pass`, `needs_review`, and `correction_rejected`;
- `needs_review_input_flag` and `correction_rejected_input_flag`;
- unsupported, forbidden, invented-content, and speaker-change outcomes;
- reference term precision/recall/F1;
- true-positive, false-positive, and missed-term counts;
- `text_error_after` and `error_delta`.

The original `context_risk_score` also included final audit outcomes. It has
been replaced with a pre-decision score based only on overlap, uncertainty,
and proposed edit magnitude.

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

## Initial Policy-Distillation Sanity Check

The audit-aware models retain the original grouped-test result:

| Model | Macro F1 | False accept | Unsafe accept | Review recall | Reject recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic regression | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 |
| Random forest | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 |
| Gradient boosting | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 |

These scores are now labeled **policy-distillation sanity check** everywhere.
They demonstrate that sklearn models can reproduce the authored policy when
given label-proxy inputs. They do not demonstrate correction-safety
generalization.

## Strict Grouped-Test Results

Removing proxy and reference-derived features lowers grouped-test results:

| Feature set / best model | Macro F1 | False accept | Unsafe accept | Review recall | Reject recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Evidence-only / random forest | 0.976 | 0.000 | 0.000 | 0.962 | 1.000 |
| Risk-only / random forest | 0.924 | 0.133 | 0.000 | 0.769 | 0.947 |

These rows still share Phase 2F/2G source structure and deterministic
augmentation patterns. They remain optimistic internal validation.

## Independent Heldout

The independent set contains 90 manually authored proposals across 30 new
template groups:

- 30 accept;
- 30 reject;
- 30 needs-review;
- 75 English, 6 French, 6 Mandarin Chinese, and 3 German examples.

It includes technical corrections, ordinary-word negative controls,
`WER/where`, `RAG/rack`, and `DER/dear` ambiguities, heavy-overlap partial
utterances, speaker-attribution risk, fluent hallucinations, no-change cases,
and multilingual examples. It does not reuse augmentation templates and
intentionally omits final audit outcome fields.

| Feature set / model | Macro F1 | False accept | Unsafe accept | Review recall | Reject recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Audit-aware / random forest | 0.312 | 0.400 | 0.433 | 0.000 | 0.567 |
| Evidence-only / random forest | **0.325** | 0.217 | 0.167 | 0.033 | 0.833 |
| Evidence-only / logistic regression | 0.316 | **0.100** | **0.067** | 0.000 | **0.933** |
| Risk-only / logistic regression | 0.295 | 0.200 | 0.133 | 0.000 | 0.867 |

The independent result is intentionally not flattering. No strict model
reliably identifies `needs_review`; recall is `0.000` to `0.033`. The best
macro F1 is only `0.325`, and even the safest strict model still accepts
`6.7%` of truly reject examples. EvidenceGate is therefore not ready for
main-workflow integration.

### Baselines

On independent heldout, `always_accept` and `always_review` both have macro
F1 `0.167`. The pre-decision rule baseline reaches macro F1 `0.269` but has
unsafe-accept rate `0.733`. The old audit-rule baseline is not a valid
deployment comparator because its strongest inputs are absent by design.

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

python experiments/audit_evidence_gate_features.py
python experiments/build_evidence_gate_heldout.py
python experiments/train_evidence_gate.py --feature-set audit_aware
python experiments/train_evidence_gate.py --feature-set evidence_only
python experiments/train_evidence_gate.py --feature-set risk_only
python experiments/evaluate_evidence_gate_heldout.py
python experiments/plot_evidence_gate_validation.py
```

## Artifacts

```text
data/controlled_evidence_gate/evidence_gate_examples.csv
data/controlled_evidence_gate/evidence_gate_examples_augmented.csv
data/controlled_evidence_gate/evidence_gate_independent_heldout.csv
experiments/results/evidence_gate/evidence_gate_feature_leakage_audit.csv
experiments/results/evidence_gate/evidence_gate_<feature_set>_*.csv
experiments/results/evidence_gate/evidence_gate_validation_metrics.csv
experiments/results/evidence_gate/evidence_gate_validation_predictions.csv
models/evidence_gate/evidence_gate_<feature_set>_<model>.joblib
assets/result_charts/evidence_gate_*.png
```

## Frontend

The **EvidenceGate Model** page now leads with the leakage warning and shows:

- the audit-aware, evidence-only, and risk-only distinction;
- the leakage audit;
- grouped versus independent metrics;
- strict heldout confusion matrix;
- unsafe-accept comparison;
- diagnostic heldout errors rather than success-only examples.

The Evidence Dashboard includes the same charts, and detective report exports
include the best available controlled EvidenceGate summary.

## Limitations And Next Validation

- Seed data is controlled authored text, not raw-audio correction decisions.
- Augmentation is deterministic and structurally simpler than real failures.
- The audit-aware labels and features share explicit policy signals.
- The original seed set is small.
- No ASR or LLM backbone is fine-tuned.
- The new heldout is manually authored but not independently annotated by
  multiple humans.
- Perfect controlled scores must not be described as real deployment
  performance.
- Strict models currently fail to recover the needs-review class.

## Final Report Wording

Use this wording:

> EvidenceGate is a trained lightweight correction-safety model. The
> audit-aware version can reproduce the controlled safety policy, while
> evidence-only and risk-only evaluations test whether correction risk can be
> predicted without direct audit labels. These results are controlled and do
> not demonstrate real-audio generalization.

The next meaningful step is blind, multi-annotator labeling of correction
proposals from held-out multilingual and overlap audio, followed by a frozen
external evaluation without changing labels, features, or thresholds.
