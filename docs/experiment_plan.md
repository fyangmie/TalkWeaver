# Experiment Plan

## Objective

Evaluate the four TalkWeaver research questions without mixing deterministic
demonstration output with real model evidence.

## Data and Annotation

The real study should use consented self-recorded meetings or appropriately
licensed public meeting data. Each clip needs:

- verbatim reference transcript;
- speaker label per temporal segment or word;
- overlap intervals and active speakers;
- required domain terms;
- noise condition, duration, language, and speaker count.

Synthetic overlap may supplement but must not replace real conversational
overlap. Development and test clips must be separated before model tuning.

## Comparison Groups

| Group | Configuration | RQ |
| --- | --- | --- |
| A | Whisper only | baseline |
| B | + preprocessing | RQ4 |
| C | + diarization + alignment | speaker baseline |
| D | + structured LLM correction | RQ1 |
| E | + RAG glossary | RQ3 |
| F | + overlap-aware correction | RQ2 |

## Metrics

### WER

`experiments/evaluate_wer.py` accepts literal text, TXT, or TalkWeaver JSON.
It uses jiwer when installed and otherwise uses token-level Levenshtein
distance after lowercase word normalization.

### Simplified Speaker Error / WDER Approximation

`experiments/evaluate_wder.py` aligns reference temporal anchors to predicted
anchors by interval overlap, then computes duration-weighted active-speaker-set
error. This is a project-level approximation. It does not replace standard
DER/WDER scoring with label mapping, collars, and word alignment.

### Term Error Rate

`experiments/evaluate_terms.py` identifies required terms in the reference and
reports missing-term rate, precision, and recall. The main comparison is
Whisper only versus structured correction versus correction plus RAG.

### Overlap and Hallucination

Overlap error is the fraction of reference anchors whose overlap flag is
wrong. A hallucinated correction is a segment rejected by TalkWeaver's lexical
grounding validator or manually judged unsupported by transcript/audio.

### Latency

`experiments/evaluate_latency.py` measures preprocessing, ASR, diarization,
alignment, overlap detection, RAG retrieval, LLM correction, and summary.
Mock timings are labeled and should not be extrapolated to model inference.

## Procedure

1. Freeze a CSV or JSON test manifest.
2. Record package, model, prompt, hardware, and decoding versions.
3. Run all groups on the same clips with fixed settings.
4. Preserve raw ASR, diarization, overlap, retrieval, prompt, correction, and
   summary artifacts.
5. Score WER, speaker error, TER, overlap error, hallucination, and latency.
6. Review overlap and correction failures using a fixed annotation rubric.
7. Aggregate per clip and report mean, variance, and sample count.
8. Generate charts only from rows whose data origin is explicit.

## Mock Demonstration

```bash
python experiments/run_ablation.py --mock
python experiments/plot_results.py
```

The mock run scores a fixed intended transcript and four fixed temporal
anchors against the deterministic pipeline. Every row uses `is_mock=true`.
The resulting zeros are expected on exact synthetic matches and are not
generalization claims.

## Outputs

- `experiments/results/ablation_results.csv`
- `experiments/results/term_error_results.csv`
- `experiments/results/latency_results.csv`
- `assets/result_charts/wer_comparison.png`
- `assets/result_charts/wder_comparison.png`
- `assets/result_charts/term_error_comparison.png`
- `assets/result_charts/latency_comparison.png`
- `assets/result_charts/hallucination_comparison.png`

## Real Experiment Example

```bash
python experiments/evaluate_wer.py \
  --reference data/reference/meeting.txt \
  --hypothesis outputs/corrected_transcripts/meeting_corrected.json

python experiments/evaluate_wder.py \
  --reference data/reference/meeting_temporal.json \
  --hypothesis outputs/corrected_transcripts/meeting_corrected.json

python experiments/evaluate_terms.py \
  --reference data/reference/meeting.txt \
  --whisper outputs/transcripts/meeting_raw_asr.json \
  --rag outputs/corrected_transcripts/meeting_corrected.json

python experiments/evaluate_latency.py \
  --audio data/demo/meeting.wav
```

## Reporting Rules

- Never relabel mock rows as real.
- Do not fill missing values with estimates.
- State that the speaker metric is simplified.
- Report excluded clips and failed model runs.
- Tie each conclusion to one research question and one controlled comparison.
