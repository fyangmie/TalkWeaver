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
Chinese script normalization: OpenCC t2s
```

One model instance is reused for all 17 clips. Per-clip `runtime_seconds`
starts immediately before `model.transcribe` and includes generator
materialization and timestamp serialization. One-time model initialization is
recorded as `cold_model_load_seconds` and excluded from per-clip RTF.
Therefore the reported RTF is a warm inference measure. The load measurement
is a local process-level model constructor time with weights already cached;
it is not mobile cold-start latency and does not include initial model
download.

## Commands Used

Run the real baseline:

```bash
python experiments/run_asr_benchmark.py \
  --manifest data/manifests/formal_eval_real.csv \
  --models tiny base \
  --device cpu \
  --compute-type int8 \
  --vad-filter true \
  --output experiments/results/asr_benchmark_real.csv \
  --predictions-dir experiments/results/asr_predictions_real
```

Run the AMI VAD diagnostic:

```bash
python experiments/run_asr_benchmark.py \
  --manifest data/manifests/formal_eval_real.csv \
  --models tiny base \
  --device cpu \
  --compute-type int8 \
  --vad-filter false \
  --only-dataset "AMI Meeting Corpus" \
  --output experiments/results/asr_benchmark_ami_no_vad_real.csv \
  --predictions-dir experiments/results/asr_predictions_ami_no_vad_real
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
an external Chinese word-segmentation policy. When
`opencc-python-reimplemented` is installed, both references and hypotheses
are converted from Traditional to Simplified Chinese before CER. Each row
records:

```text
script_normalized=true
normalization_notes=OpenCC t2s normalization applied before Mandarin CER scoring.
```

Without OpenCC, the evaluator keeps the original script and records a warning
that Traditional/Simplified differences may inflate CER. OpenCC remains an
optional dependency:

```bash
pip install opencc-python-reimplemented
```

Both metrics use a local Levenshtein implementation. WER may use `jiwer` when
it is already installed, but `jiwer` is not required.

### AMI Diagnostic Cleaned WER

AMI manual references contain fillers, backchannels, repetitions, and
punctuation markers that may be rendered differently by ASR. The benchmark
therefore retains standard WER and adds a diagnostic:

```text
WER_DISFLUENCY_CLEANED
```

The diagnostic removes `um`, `uh`, `mm-hmm`, `mm`, `hmm`, and related
variants, then collapses immediately repeated words in both reference and
hypothesis. It does not replace standard WER and is not used for FLEURS.

### VAD Diagnostic

`--vad-filter true` remains the default baseline. `--vad-filter false` allows
an AMI-only diagnostic because VAD may truncate low-energy meeting speech or
backchannels. The CSV records `vad_filter` for every row.

### Warm Runtime, Cold Load, And RTF

```text
RTF = runtime_seconds / duration_seconds
```

An RTF below 1 means inference completed faster than the clip duration on the
measured machine. These are local CPU measurements, not mobile-device
measurements. `cold_model_load_seconds` is reported separately. It must not be
treated as mobile startup latency because filesystem cache state, process
startup, model download, hardware, and runtime packaging differ.

## Small-Subset Results

Language-level means combine the available datasets for that language:

| Model | English WER, 7 clips | French WER, 5 clips | Mandarin CER with OpenCC, 5 clips | Mean warm RTF, 17 clips |
| --- | ---: | ---: | ---: | ---: |
| tiny | 0.3123 | 0.4387 | 0.2761 | 0.0473 |
| base | 0.2160 | 0.2738 | 0.0897 | 0.0758 |

OpenCC changes only the evaluation normalization, not model predictions. The
previous script-sensitive Mandarin means were `0.4336` for `tiny` and
`0.3475` for `base`; those values were inflated by Traditional/Simplified
character differences.

Dataset-specific results show why aggregate interpretation must remain
conservative:

| Model | FLEURS WER, English + French | FLEURS Mandarin CER | AMI standard WER | AMI cleaned WER |
| --- | ---: | ---: | ---: | ---: |
| tiny | 0.3675 | 0.2761 | 0.3527 | 0.2596 |
| base | 0.2140 | 0.0897 | 0.3705 | 0.2885 |

`base` improves FLEURS multilingual accuracy, while `tiny` is faster. AMI is
unstable: only two short clips are available, the references preserve
disfluencies, and the VAD-enabled `base` hypothesis for `ami_es2002a_01`
stops after "what we're".

The AMI VAD diagnostic was:

| Model | VAD=true standard / cleaned WER | VAD=false standard / cleaned WER |
| --- | ---: | ---: |
| tiny | 0.3527 / 0.2596 | 0.3371 / 0.2404 |
| base | 0.3705 / 0.2885 | 0.2612 / 0.1538 |

Disabling VAD restored the remainder of the truncated `base` hypothesis and
substantially reduced its two-clip AMI error. This is a diagnostic finding,
not a general recommendation to disable VAD.

## Outputs

Committed small artifacts:

```text
experiments/results/asr_benchmark_real.csv
experiments/results/asr_benchmark_summary_real.csv
experiments/results/asr_benchmark_ami_no_vad_real.csv
assets/result_charts/asr_error_by_language.png
assets/result_charts/asr_error_by_dataset.png
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
- Standard AMI WER is reference-style sensitive; cleaned WER is diagnostic.
- VAD conclusions are based on only two clips.
- Per-clip warm RTF excludes model initialization and model download.
- Local `cold_model_load_seconds` is not mobile cold-start latency.
- Runtime reflects the current CPU environment and cannot be reported as a
  mobile result.
- No statistical significance test is meaningful at this sample size.
- ASR-only results do not validate TalkWeaver diarization, overlap-aware
  correction, RAG recovery, or hallucination controls.

Do not claim these values as full FLEURS, full AMI, or state-of-the-art model
performance.
