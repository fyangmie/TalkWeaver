# Real ASR Benchmark

## Purpose

Phase 2C establishes a real ASR-only baseline before TalkWeaver workflow
ablations. It measures what `faster-whisper` produces directly from the
formal manifest audio, without diarization, overlap reasoning, RAG retrieval,
LLM correction, or ConversationMap post-processing.

These results are a **small-subset formal evaluation**, not full FLEURS or
AMI corpus performance.

## Evaluation Subset

The frozen local manifest is:

```text
data/manifests/formal_eval_real.csv
```

It contains 17 real public-data clips and 190.6 seconds of audio:

| Dataset | Language | Clips | Duration | Metric |
| --- | --- | ---: | ---: | --- |
| Google FLEURS validation fallback | English | 5 | 39.9 s | WER |
| Google FLEURS validation fallback | French | 5 | 52.2 s | WER |
| Google FLEURS validation fallback | Mandarin Chinese | 5 | 58.5 s | CER |
| AMI Meeting Corpus excerpts | English | 2 | 40.0 s | WER |

The AMI excerpts contain natural multi-speaker meeting speech and overlap.
The FLEURS clips provide the multilingual ASR track. Raw audio is local and
ignored by Git.

## Models And Runtime Protocol

The June 13, 2026 run used:

```text
faster-whisper 1.2.1
models: tiny, base
device: cpu
compute type: int8
beam size: 5
VAD filter: enabled
word timestamps: enabled
```

One model instance is reused for all 17 clips. Per-clip `runtime_seconds`
starts immediately before `model.transcribe` and includes generator
materialization and timestamp serialization. One-time model initialization is
recorded in prediction metadata and CSV notes but excluded from per-clip RTF.

## Commands Used

Run the real baseline:

```bash
python experiments/run_asr_benchmark.py \
  --manifest data/manifests/formal_eval_real.csv \
  --models tiny base \
  --device cpu \
  --compute-type int8 \
  --output experiments/results/asr_benchmark_real.csv \
  --predictions-dir experiments/results/asr_predictions_real
```

Create aggregate results:

```bash
python experiments/summarize_asr_results.py \
  --input experiments/results/asr_benchmark_real.csv \
  --output experiments/results/asr_benchmark_summary_real.csv
```

Create charts:

```bash
python experiments/plot_asr_results.py \
  --input experiments/results/asr_benchmark_real.csv \
  --output-dir assets/result_charts
```

## Metrics

### Word Error Rate

English, French, and AMI meeting clips use:

```text
WER = (substitutions + deletions + insertions) / reference words
```

Text is lowercased, common punctuation is removed, and whitespace is
normalized. Internal apostrophes, hyphens, periods, and underscores are
retained when they connect alphanumeric characters, which avoids splitting
terms such as `faster-whisper` and `pyannote.audio`.

### Character Error Rate

Mandarin Chinese uses:

```text
CER = character edit distance / reference characters
```

Whitespace and punctuation are removed before scoring. This avoids imposing
an external Chinese word-segmentation policy.

Both metrics use a local Levenshtein implementation. WER may use `jiwer` when
it is already installed, but `jiwer` is not required.

### Runtime And RTF

```text
RTF = runtime_seconds / duration_seconds
```

An RTF below 1 means inference completed faster than the clip duration on the
measured machine. These are local CPU measurements, not mobile-device
measurements.

## Small-Subset Results

Language-level means combine the available datasets for that language:

| Model | English WER, 7 clips | French WER, 5 clips | Mandarin CER, 5 clips | Mean RTF, 17 clips |
| --- | ---: | ---: | ---: | ---: |
| tiny | 0.3123 | 0.4387 | 0.4336 | 0.0470 |
| base | 0.2160 | 0.2738 | 0.3475 | 0.0762 |

Dataset-specific English results show why aggregate interpretation must remain
conservative:

| Model | FLEURS English WER | AMI overlap excerpt WER |
| --- | ---: | ---: |
| tiny | 0.2962 | 0.3527 |
| base | 0.1542 | 0.3705 |

`base` improves the FLEURS language subsets but is slightly worse than
`tiny` on the two AMI excerpts. With only two short meeting clips, this is an
error-analysis observation rather than evidence of general model ordering.

## Outputs

Committed small artifacts:

```text
experiments/results/asr_benchmark_real.csv
experiments/results/asr_benchmark_summary_real.csv
assets/result_charts/asr_error_by_language.png
assets/result_charts/asr_rtf_by_model.png
```

Local prediction artifacts:

```text
experiments/results/asr_predictions_real/
```

Each model/clip pair has:

- JSON metadata, original and normalized text, metric result, segments, and
  word timestamps;
- a readable TXT hypothesis.

The prediction directory is ignored by Git. It can be regenerated from the
manifest and model cache.

## Limitations

- The study covers 17 clips, not complete corpus test partitions.
- FLEURS is the documented multilingual fallback, not Common Voice.
- There is no Mandarin meeting clip yet; Mandarin results are single-speaker
  FLEURS CER.
- AMI has only two 20-second overlap excerpts.
- Per-clip RTF excludes one-time model initialization and model download.
- Runtime reflects the current CPU environment and cannot be reported as a
  mobile result.
- No statistical significance test is meaningful at this sample size.
- ASR-only results do not validate TalkWeaver diarization, overlap-aware
  correction, RAG recovery, or hallucination controls.

Do not claim these values as full FLEURS, full AMI, or state-of-the-art model
performance.
