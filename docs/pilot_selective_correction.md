# Selective LLM-ASR Correction Feasibility Pilot

## Scope

This Phase R0 pilot tests one question before TalkWeaver commits to a larger
human-labeled benchmark:

> Is an LLM self-judge already sufficient for deciding when an ASR correction
> should be accepted, rejected, or deferred, or do explicit cross-speech
> evidence and abstention still add value?

The working paper-style title is:

> **When Not to Correct: Evidence-Calibrated LLM-ASR Correction with
> Abstention in Overlapped Multi-Speaker Meetings**

This is a feasibility pilot, not paper-level evidence. It does not use raw
audio, does not evaluate real-audio generalization, and does not replace a
future human annotation study.

## Pilot Dataset

`data/pilot/selective_correction_pilot.csv` contains 72 explicitly authored
correction proposals:

- 24 suggested `accept` labels;
- 24 suggested `reject` labels;
- 24 suggested `needs_review` labels.

The eight categories each contain nine proposals:

| Category | Proposals |
|---|---:|
| Technical term recovery | 9 |
| Ordinary-word negative control | 9 |
| Weak retrieval evidence | 9 |
| Heavy overlap | 9 |
| Partial utterance | 9 |
| Speaker-attribution risk | 9 |
| Fluent hallucination | 9 |
| No-change case | 9 |

Languages are English (66), Mandarin Chinese (3), and French (3).

`suggested_gold_label` is an author-provided pilot label. It is not a final
human annotation. `human_checked_label` and `human_checked_rationale` remain
blank. Evaluation therefore records `label_source=pilot_auto_labeled`.

The dataset can be regenerated with:

```bash
python experiments/build_selective_correction_pilot.py \
  --output data/pilot/selective_correction_pilot.csv
```

## Methods

### Always Accept

Accept every proposed correction. This measures maximum coverage without
safety control.

### Always Review

Defer every proposal to a human. This is safe but has zero automatic coverage.

### EccoGate

`backend/eccogate.py` is a deterministic proposal-time gate. It does not call
an LLM and does not inspect labels or post-hoc correctness fields. It uses:

- retrieved term support;
- whether the retrieved term matches the stated context;
- overlap and heavy-overlap flags;
- speaker ambiguity;
- partial-utterance evidence;
- changed-token and edit-size risk;
- unsupported text expansion;
- speaker-attribution changes.

It outputs `accept`, `reject`, or `needs_review` with support/risk scores and a
short explanation.

### LLM Self-Judge

`experiments/run_pilot_llm_self_judge.py` supports:

- `no_evidence`: raw ASR and proposed correction only;
- `with_evidence`: adds context, retrieved terms, overlap, speaker ambiguity,
  and partial-utterance flags.

The script requires a configured real API. It has no rule fallback. A failed
API call cannot be recorded as an LLM result.

## Metrics

- **Macro F1:** equal-weighted F1 over accept, reject, and needs-review.
- **Unsafe-accept rate:** predicted accept among gold reject or needs-review
  proposals.
- **Needs-review recall:** review-needed proposals correctly deferred.
- **Accept precision:** accepted proposals whose pilot label is accept.
- **Reject recall:** rejected proposals correctly rejected.
- **Coverage:** fraction not deferred to needs-review.
- **Per-category unsafe-accept rate:** unsafe accepts separated by hard-case
  category.

## Current Pilot Results

The current committed run contains no LLM predictions. The local DeepSeek
configuration was detected, but external API access was not authorized in the
automated environment. No fallback rows were created.

| Method | Macro F1 | Unsafe accept | Review recall | Accept precision | Reject recall | Coverage |
|---|---:|---:|---:|---:|---:|---:|
| Always accept | 0.167 | 1.000 | 0.000 | 0.333 | 0.000 | 1.000 |
| Always review | 0.167 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| EccoGate | 0.819 | 0.000 | 0.667 | 1.000 | 0.792 | 0.708 |

EccoGate accepted all 24 suggested-safe proposals, deferred 16 of 24
review-needed proposals, and rejected 19 of 24 suggested-unsafe proposals.
It made no unsafe accepts on these authored fixtures. Eight review cases were
conservatively rejected and five reject cases were conservatively deferred.

These values are a policy sanity check on auto-labeled proposals. They must
not be presented as real-world correction performance.

## Interpretation Rules

The pilot is intended to decide the next research step:

1. If the LLM with evidence dominates EccoGate on safety and coverage,
   EccoGate should not be claimed as the main method. The research should
   focus on LLM self-judging and evidence conditioning.
2. If the LLM still unsafe-accepts hard overlap, attribution, or hallucination
   cases, evidence-calibrated abstention remains a promising direction.
3. If EccoGate is safer but has lower accuracy or coverage, frame the method
   as a safety-coverage trade-off rather than a universal accuracy gain.

The current result cannot answer the LLM comparison because the real API run
is missing. It supports only a provisional conclusion: a transparent
proposal-time abstention policy can avoid unsafe acceptance on the authored
pilot while retaining 70.8% coverage.

## Reproduction

Run the real LLM conditions after API access is authorized:

```bash
python experiments/run_pilot_llm_self_judge.py \
  --input data/pilot/selective_correction_pilot.csv \
  --mode no_evidence \
  --output experiments/results/pilot/llm_self_judge_pilot_predictions.csv

python experiments/run_pilot_llm_self_judge.py \
  --input data/pilot/selective_correction_pilot.csv \
  --mode with_evidence \
  --output experiments/results/pilot/llm_self_judge_pilot_predictions.csv \
  --append
```

Evaluate and plot:

```bash
python experiments/run_pilot_selective_correction_eval.py \
  --input data/pilot/selective_correction_pilot.csv \
  --llm-predictions experiments/results/pilot/llm_self_judge_pilot_predictions.csv \
  --output experiments/results/pilot/selective_correction_pilot_results.csv \
  --summary-output experiments/results/pilot/selective_correction_pilot_summary.csv

python experiments/plot_pilot_selective_correction.py \
  --summary experiments/results/pilot/selective_correction_pilot_summary.csv \
  --results experiments/results/pilot/selective_correction_pilot_results.csv \
  --output-dir assets/result_charts
```

Generated charts:

- `pilot_unsafe_accept_rate.png`;
- `pilot_needs_review_recall.png`;
- `pilot_macro_f1.png`;
- `pilot_category_failure.png`.

## Limitations and Next Decision

- Labels are suggested pilot labels, not independently adjudicated labels.
- Cases are controlled text proposals, not measured real-audio errors.
- The dataset is small and intentionally balanced.
- EccoGate uses lexical context cues designed for this feasibility study.
- Multilingual coverage is illustrative, not statistically meaningful.
- The LLM comparison is still required before committing to a full benchmark.

Recommended next step: run both real LLM judge modes, manually check a
stratified subset of at least 20 proposals, and then decide whether to proceed
to a full human-labeled benchmark or pivot to an LLM self-judge analysis.
