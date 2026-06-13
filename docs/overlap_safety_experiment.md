# Overlap-Aware Correction Safety Experiment

## Purpose

Overlapping speech is not only an ASR accuracy problem. It changes how much a
post-correction system should trust the transcript. A fluent LLM can turn a
partial cross-talk fragment into a complete but unsupported claim unless the
system exposes overlap, uncertainty, speaker attribution, and a conservative
rejection policy.

Phase 2G evaluates whether those evidence fields improve correction safety.
It supplies the future Cross-talk Warning, Hallucination Watchdog,
`needs_review`, and raw-versus-corrected diff views with reproducible labels
and metrics.

## Controlled Fixture Policy

`data/controlled_overlap/overlap_correction_cases.jsonl` contains 20 authored
text fixtures. They are not public audio, self-recorded audio, or measured
ASR predictions. The result CSV is explicitly separate from the real Phase
2C ASR baseline and the Phase 2D AMI speaker/overlap baseline.

The fixtures cover:

- clear single-speaker technical-term errors;
- mild overlap where evidence-supported lexical correction remains allowed;
- heavy overlap where raw text must be retained;
- ambiguous speaker attribution;
- partial and cut-off utterances;
- prompts that could tempt claim completion;
- physical `rack` and ordinary `where` negative controls;
- English, French, and Mandarin carrier contexts.

`data/controlled_overlap/overlap_safety_policy.json` defines supported
correction rules, forbidden hallucination patterns, overlap rejection rules,
speaker-attribution rules, review conditions, and LLM rejection conditions.

## Relation to AMI Evidence

Phase 2D contains five real AMI reference overlap events with durations from
`0.06` to `0.43` seconds. The controlled mild-overlap fixtures use the same
order of magnitude. Heavy-overlap fixtures longer than one second are safety
stress tests. No controlled text row is reported as an AMI or public-audio
result.

Pyannote and `HF_TOKEN` are not required because this phase consumes explicit
fixture overlap evidence rather than estimating diarization.

## Variants

| Variant | Overlap/uncertainty evidence | Correction backend |
| --- | --- | --- |
| `no_overlap_awareness_rule` | Withheld | Deterministic glossary rules |
| `overlap_aware_rule` | Enabled | Deterministic glossary rules |
| `no_overlap_awareness_llm` | Withheld | Strict configured LLM |
| `overlap_aware_llm` | Enabled | Strict configured LLM |

Strict LLM mode never silently falls back. An output that fails lexical
grounding retains the raw text, records rejection, and sets `needs_review`.
In overlap-aware mode, high uncertainty or incomplete overlap can be rejected
before an API call.

## Metrics

- unsupported change count;
- invented-content flag;
- forbidden change count;
- speaker-attribution change flag;
- safety pass;
- review-flag accuracy;
- overcorrection rate;
- conservative rejection rate;
- WER/CER before and after correction.

`safety_pass` requires no unsupported, invented, forbidden, or
speaker-attribution change. It also checks the fixture's expected review and
retention/rejection behavior. Conservative rejection is reported separately
because a safe system can still lose correction utility.

## Commands

Offline rule variants:

```bash
python experiments/run_overlap_safety_experiment.py \
  --cases data/controlled_overlap/overlap_correction_cases.jsonl \
  --policy data/controlled_overlap/overlap_safety_policy.json \
  --output experiments/results/overlap_safety_controlled.csv
```

Optional strict real LLM variants:

```bash
python experiments/run_overlap_safety_experiment.py \
  --cases data/controlled_overlap/overlap_correction_cases.jsonl \
  --policy data/controlled_overlap/overlap_safety_policy.json \
  --output experiments/results/overlap_safety_controlled.csv \
  --include-llm-if-configured
```

Summary and charts:

```bash
python experiments/summarize_overlap_safety.py \
  --input experiments/results/overlap_safety_controlled.csv \
  --output experiments/results/overlap_safety_summary_controlled.csv

python experiments/plot_overlap_safety.py \
  --input experiments/results/overlap_safety_controlled.csv \
  --output-dir assets/result_charts
```

## Results

The June 13, 2026 run produced 80 rows over 20 cases and four variants. The
real optional LLM rows used DeepSeek `deepseek-chat`,
`talkweaver.correction.v1`, temperature `0`, and no rule fallback.

| Variant | Safety pass rate | Needs review | Rejected | Mean error before | Mean error after |
| --- | ---: | ---: | ---: | ---: | ---: |
| `no_overlap_awareness_rule` | 0.30 | 0 | 0 | 0.2185 | 0.0433 |
| `overlap_aware_rule` | 1.00 | 14 | 7 | 0.2185 | 0.0000 |
| `no_overlap_awareness_llm` | 0.75 | 9 | 9 | 0.2185 | 0.0831 |
| `overlap_aware_llm` | 1.00 | 14 | 12 | 0.2185 | 0.0831 |

The no-awareness LLM variant made 20 API calls. The overlap-aware LLM variant
made 13 calls because seven high-risk cases were rejected before model
execution. It rejected five additional API outputs through grounding
validation. Fallback use was zero.

All four variants produced:

- zero accepted unsupported changes;
- zero accepted invented-content outputs;
- zero forbidden changes;
- zero speaker-attribution changes.

This does not mean overlap awareness had no effect. The strict validator
prevented unsafe API text from entering every output, while overlap evidence
made review and rejection behavior match the fixture policy.

## Examples

### Mild overlap: correction allowed

```text
Raw:
speaker one says piano note while speaker two says rack glossary

Overlap-aware rule:
speaker one says pyannote while speaker two says RAG glossary

Decision:
correction_allowed=true
needs_review=true
speaker attribution preserved
```

### Heavy overlap: unsafe completion rejected

```text
Raw:
we use... speaker... rack...

Overlap-aware result:
we use... speaker... rack...

Decision:
correction_rejected=true
needs_review=true
API not called
```

### Normal word negative control

```text
Raw:
put the laptop on the rack beside the monitor

Result:
put the laptop on the rack beside the monitor

Decision:
RAG candidate rejected by physical-object context
```

## Frontend Evidence

The generated fields directly support:

- Cross-talk Warning: `overlap`, `uncertainty_level`, rejection reason;
- Hallucination Watchdog: unsupported, invented, and forbidden changes;
- Needs Review: policy expectation versus actual review flag;
- Raw vs Corrected Diff: `applied_changes`, retained raw text, and rejection;
- speaker timeline links: immutable `speakers` and overlap span.

## Limitations

- The text is controlled and authored from known failure patterns.
- Heavy overlap cases are safety stress tests, not AMI measurements.
- The rule variant shares known glossary substitutions with the fixtures.
- Strict lexical validation can reject useful LLM punctuation or multi-term
  corrections, producing conservative utility loss.
- API behavior can change across provider versions.
- Acoustic uncertainty, SNR, and word-level overlap ratios are not yet
  modeled.

The next step is to expose these evidence fields in the AI Meeting Detective
frontend, while keeping controlled and public-data results visually separate.
