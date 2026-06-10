# Experiment Plan

## Objective

Evaluate the four research questions without presenting mock output as real
evidence.

## Data

1. Consented self-recorded English project meetings with technical terms.
2. Controlled synthetic overlap created from isolated speaker clips.
3. Optional public meeting or overlap datasets, subject to license review.
4. Human reference transcript, speaker labels, overlap regions, and domain-term
   annotations for every evaluated clip.

## Comparison Groups

| Group | Configuration | Primary purpose |
| --- | --- | --- |
| A | Whisper only | ASR baseline |
| B | Preprocessing + Whisper | Test RQ4 |
| C | Whisper + diarization + alignment | Speaker baseline |
| D | Structured LLM correction | Test RQ1 |
| E | Structured correction + RAG glossary | Test RQ3 |
| F | Overlap-aware vs no-overlap-flag correction | Test RQ2 |

## Metrics

- WER for transcript accuracy.
- WDER or a clearly defined speaker-attribution error.
- Term Error Rate and retrieval precision/recall.
- Overlap-region WER and error categories.
- Hallucinated correction count under a written annotation policy.
- End-to-end and per-stage latency.

## Procedure

1. Freeze a test manifest before comparing variants.
2. Record audio duration, speakers, overlap ratio, noise condition, and domain
   terms.
3. Run each configuration with fixed seeds and logged versions.
4. Save raw intermediate artifacts for audit.
5. Score against references with the scripts under `experiments/`.
6. Review unsupported edits in a blinded or consistently ordered process.
7. Save CSV results and generate charts only from measured rows.

## Mock Smoke Test

`python experiments/run_ablation.py --mock` creates the expected CSV schema
with empty metrics and `mock_demo_not_measured` labels. It verifies plumbing
only.

## Result Files

- `experiments/results/ablation_results.csv`
- `experiments/results/term_error_results.csv`
- `experiments/results/latency_results.csv`
- `assets/result_charts/wer_comparison.png`
- `assets/result_charts/wder_comparison.png`
- `assets/result_charts/term_error_comparison.png`
- `assets/result_charts/latency_comparison.png`

## Reporting Rules

- Never fill missing results with estimates.
- Separate development, validation, and test data.
- Report failures and excluded samples.
- Keep mock, pilot, and final results in distinct tables.
- Tie every conclusion to a comparison and metric.
