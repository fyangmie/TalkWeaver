# Binary Safe-to-Apply Correction Benchmark

## Scope

Phase R1 reformulates TalkWeaver's correction-safety pilot as a binary task:

```text
safe_to_apply
do_not_apply
```

The question is intentionally narrower than the Phase R0 three-way pilot:

> Can a policy decide whether an LLM-ASR correction is safe to apply under
> term misrecognition, overlap, partial speech, and speaker ambiguity?

This is still a controlled feasibility benchmark. It is not a paper-level
result and does not demonstrate real-audio generalization.

## Why Binary

The earlier `accept / reject / needs_review` formulation exposed a labeling
problem: `needs_review` depends on reviewer tolerance and operational cost.
Without independent human annotation, that middle class is difficult to
define reproducibly.

The binary formulation uses a measurable rule whenever a reference transcript
exists:

```text
safe_to_apply if error_after + margin < error_before
do_not_apply otherwise
```

The default margin is `0.01`. Unsupported, invented, forbidden, or
speaker-attribution-changing content receives a `do_not_apply` safety override
even when lexical error decreases.

This reduces subjectivity, but it removes explicit abstention. A future system
can reintroduce review as a calibrated uncertainty band around the binary
decision threshold.

## Benchmark Construction

Run:

```bash
python experiments/build_binary_safe_apply_benchmark.py \
  --output data/pilot/binary_safe_apply_benchmark.csv \
  --margin 0.01
```

The current benchmark contains 327 proposals:

| Source | Rows | Label source |
|---|---:|---|
| Controlled term rescue | 175 | `controlled_reference` |
| Controlled overlap safety | 80 | `controlled_reference` |
| Phase R0 pilot | 72 | `pilot_suggested_if_no_reference` |

Overall labels:

- `safe_to_apply`: 80;
- `do_not_apply`: 247.

Of these, 255 rows use controlled reference transcripts and measured WER/CER.
The 72 R0 rows have no reference transcript. They remain a secondary slice:
R0 `accept` maps to `safe_to_apply`, while `reject` and `needs_review` map to
`do_not_apply`.

Existing ASR prediction JSON is scanned only when it contains a reference
transcript and an actual proposed correction. The current Phase 2C files are
ASR-only predictions, so no fake correction rows are added.

## Compared Methods

### Always Apply

Apply every correction. This has maximum coverage and exposes the upper bound
on unsafe application.

### Never Apply

Block every correction. This prevents unsafe application but loses every
beneficial correction.

### Retrieval Only

Apply when a newly introduced term appears in the retrieved candidates. This
tests whether retrieval evidence alone is sufficient permission to edit.

### Overlap-Unaware Policy

Run the same proposal-time lexical and edit-risk policy as EccoGate while
withholding overlap, speaker ambiguity, and partial-utterance flags.

### Binary EccoGate

Use retrieval support, context compatibility, edit size, overlap, heavy
overlap, partial speech, and speaker-attribution risk. It does not inspect the
reference transcript or call an LLM.

### LLM Self-Judge

Two optional real-API modes are supported:

- `no_evidence`: raw ASR and proposed correction only;
- `with_evidence`: adds context, retrieved terms, overlap, heavy overlap,
  speaker ambiguity, and partial-utterance flags.

The LLM prompt never receives `reference_text`, `binary_label`, or error
scores. There is no rule fallback.

## Metrics

- **Accuracy:** overall binary correctness.
- **Macro F1:** equal-weighted F1 for both labels.
- **Safe-apply precision:** fraction of applied corrections that are safe.
- **Safe-apply recall:** fraction of beneficial corrections applied.
- **Unsafe-apply rate:** applied among gold `do_not_apply` proposals.
- **False-block rate:** blocked among gold `safe_to_apply` proposals.
- **Coverage:** fraction predicted `safe_to_apply`.
- **Error delta when applied:** mean `error_before - error_after` among applied
  rows that have a reference transcript.
- **Per-category unsafe-apply rate:** failure rate separated by proposal type.

## Current Results

Both LLM modes ran with the configured `deepseek/deepseek-chat` API. All 654
rows have `api_used=true`; no fallback predictions were generated.

| Method | Macro F1 | Unsafe apply | False block | Coverage | Applied error delta |
|---|---:|---:|---:|---:|---:|
| Always apply | 0.197 | 1.000 | 0.000 | 1.000 | 0.101 |
| Never apply | 0.430 | 0.000 | 1.000 | 0.000 | n/a |
| Retrieval only | 0.816 | 0.198 | 0.037 | 0.385 | 0.442 |
| Overlap-unaware policy | 0.884 | 0.077 | 0.125 | 0.272 | 0.378 |
| Binary EccoGate | 0.883 | 0.053 | 0.188 | 0.239 | 0.397 |
| LLM self-judge, no evidence | 0.379 | 0.522 | 0.713 | 0.465 | 0.070 |
| LLM self-judge, with evidence | 0.596 | 0.324 | 0.425 | 0.385 | 0.153 |

Binary EccoGate reduces unsafe application relative to retrieval-only and the
overlap-unaware policy. Its cost is lower coverage and more false blocking.
This is a safety-coverage trade-off, not a universal accuracy improvement.

Evidence conditioning materially improves the LLM judge:

- macro F1 rises from `0.379` to `0.596`;
- unsafe application falls from `0.522` to `0.324`;
- false blocking falls from `0.713` to `0.425`;
- 75 incorrect no-evidence decisions are fixed;
- 3 previously correct decisions are harmed.

The remaining unsafe-application rate is too high for automatic application.
The strongest LLM failure categories are ordinary-word negative controls,
overlap correction, and single-speaker safety cases. This supports the
research premise that evidence helps, but LLM self-judging alone is not a
sufficient safety gate.

### Source Sensitivity

The reference-derived controlled slice and R0 suggested-label slice behave
differently. On the 255 controlled-reference rows:

- binary EccoGate macro F1 is `0.877`, unsafe application `0.065`;
- evidence-conditioned LLM macro F1 is `0.550`, unsafe application `0.392`;
- retrieval-only reaches macro F1 `0.989`, partly because the controlled term
  fixtures explicitly expose the same retrieval evidence used to generate
  their proposals.

That retrieval-only result should not be interpreted as independent
generalization. It identifies a source-design dependency that must be removed
with a new heldout benchmark.

## Commands

```bash
python experiments/run_binary_llm_self_judge.py \
  --input data/pilot/binary_safe_apply_benchmark.csv \
  --mode no_evidence \
  --output experiments/results/binary_safe_apply/llm_self_judge_binary_predictions.csv

python experiments/run_binary_llm_self_judge.py \
  --input data/pilot/binary_safe_apply_benchmark.csv \
  --mode with_evidence \
  --output experiments/results/binary_safe_apply/llm_self_judge_binary_predictions.csv \
  --append

python experiments/run_binary_safe_apply_experiment.py \
  --input data/pilot/binary_safe_apply_benchmark.csv \
  --llm-predictions experiments/results/binary_safe_apply/llm_self_judge_binary_predictions.csv \
  --output experiments/results/binary_safe_apply/binary_safe_apply_results.csv \
  --summary-output experiments/results/binary_safe_apply/binary_safe_apply_summary.csv

python experiments/plot_binary_safe_apply_results.py \
  --summary experiments/results/binary_safe_apply/binary_safe_apply_summary.csv \
  --results experiments/results/binary_safe_apply/binary_safe_apply_results.csv \
  --output-dir assets/result_charts
```

## Limitations

- Controlled term and overlap rows are text fixtures, not measured public
  audio corrections.
- The R0 secondary slice uses suggested labels because no reference exists.
- Multiple pipeline variants share source cases, so this is not an independent
  train/test generalization benchmark.
- Retrieval candidates and controlled term proposals are coupled in part of
  the benchmark, making retrieval-only optimistic on that source.
- The class distribution is intentionally realistic but imbalanced.
- Error improvement does not capture every semantic or discourse failure.
- Multilingual coverage remains small.
- No human adjudication is required in R1, but future publication-quality work
  still needs independent review.

## Next Step

If the binary LLM and EccoGate results remain promising, the next phase should:

1. freeze a smaller independent heldout set;
2. collect human checks for a stratified subset;
3. calibrate a confidence threshold;
4. reintroduce `needs_review` only as a thresholded uncertain region rather
   than a primary subjective class.

The Phase R0 three-way EvidenceGate remains exploratory. The binary task is
the current focused pilot and is more viable as a reproducible research
formulation, but the current data is not yet an independent benchmark.
