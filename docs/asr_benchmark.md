# Real ASR Benchmark

## Purpose

Phase 2C establishes a real ASR-only baseline before TalkWeaver workflow
ablations. It measures what `faster-whisper` produces directly from the
formal manifest audio, without diarization, overlap reasoning, RAG retrieval,
LLM correction, or ConversationMap post-processing.

These results are a **small-subset formal evaluation**, not full FLEURS, AMI,
or AISHELL-4 corpus performance.

## Evaluation Subset

The frozen local manifest is:

```text
data/manifests/formal_eval_real.csv
```

It contains 50 real public-data clips and 698.34 seconds of audio:

| Dataset | Language | Clips | Duration | Metric |
| --- | --- | ---: | ---: | --- |
| Google FLEURS validation fallback | English | 10 | 80.0 s | WER |
| Google FLEURS validation fallback | French | 10 | 97.38 s | WER |
| Google FLEURS validation fallback | Mandarin Chinese | 10 | 120.96 s | CER |
| AMI Meeting Corpus excerpts | English | 8 | 160.0 s | WER |
| AISHELL-4 meeting excerpts | Mandarin Chinese | 12 | 240.0 s | CER |

The AMI and AISHELL-4 excerpts contain natural multi-speaker meeting speech.
The FLEURS clips provide the read-speech multilingual ASR track. Raw audio is
local and ignored by Git.

A larger Mandarin meeting benchmark subset is also frozen separately:

```text
data/manifests/aishell4_benchmark_60x20.csv
```

It contains 60 AISHELL-4 clips, 20 seconds each, from all 20 locally extracted
test recordings. The subset totals 1200 seconds, caps each recording at three
clips, contains 29 multi-speaker clips, and marks 10 clips with reference
overlap. This is stronger than the earlier 12-clip sanity check, but it is
still a fixed subset rather than full AISHELL-4 test-set performance.

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

One model instance is reused for all 50 clips. Per-clip `runtime_seconds`
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

python experiments/run_asr_benchmark.py \
  --manifest data/manifests/english_meeting_heldout_real.csv \
  --models tiny base \
  --device cpu \
  --compute-type int8 \
  --vad-filter true \
  --output experiments/results/asr_benchmark_english_meeting_heldout_real.csv \
  --predictions-dir experiments/results/asr_predictions_english_meeting_heldout_real

python experiments/summarize_asr_results.py \
  --input experiments/results/asr_benchmark_english_meeting_heldout_real.csv \
  --output experiments/results/asr_benchmark_english_meeting_heldout_summary_real.csv

python experiments/run_asr_benchmark.py \
  --manifest data/manifests/english_meeting_heldout_real.csv \
  --models small \
  --device cpu \
  --compute-type int8 \
  --vad-filter true \
  --output experiments/results/asr_benchmark_english_meeting_heldout_small_real.csv \
  --predictions-dir experiments/results/asr_predictions_english_meeting_heldout_small_real

python experiments/summarize_asr_results.py \
  --input experiments/results/asr_benchmark_english_meeting_heldout_small_real.csv \
  --output experiments/results/asr_benchmark_english_meeting_heldout_small_summary_real.csv

python experiments/run_asr_benchmark.py \
  --manifest data/manifests/aishell4_benchmark_60x20.csv \
  --models tiny base small \
  --device cpu \
  --compute-type int8 \
  --vad-filter true \
  --output experiments/results/asr_benchmark_aishell4_60x20_real.csv \
  --predictions-dir experiments/results/asr_predictions_aishell4_60x20_real

python experiments/summarize_asr_results.py \
  --input experiments/results/asr_benchmark_aishell4_60x20_real.csv \
  --output experiments/results/asr_benchmark_aishell4_60x20_summary_real.csv
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

The June 21, 2026 expanded run produced 100 model/clip rows: 50 clips times
two model sizes.

| Model | FLEURS English WER, 10 clips | FLEURS French WER, 10 clips | FLEURS Mandarin CER, 10 clips | AMI standard WER, 8 clips | AMI cleaned WER, 8 clips | AISHELL-4 CER, 12 clips | Mean warm RTF, 50 clips |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tiny | 0.2104 | 0.3873 | 0.2226 | 0.4323 | 0.3775 | 0.6795 | 0.0506 |
| base | 0.1144 | 0.2271 | 0.1133 | 0.3984 | 0.3312 | 0.6100 | 0.0654 |

OpenCC changes only the evaluation normalization, not model predictions.
`base` improves accuracy on every current subset, while `tiny` is faster.
Meeting speech remains much harder than read FLEURS speech: the AMI standard
WER is about 0.40 even for `base`, and the diagnostic cleaned WER is still
0.3312. Mandarin meeting speech is also substantially harder than FLEURS
Mandarin read speech.

The earlier AMI VAD diagnostic showed that VAD can truncate low-energy meeting
speech. That diagnostic used only two AMI clips and is not treated as the
expanded formal result. A publishable VAD claim should rerun the 8-clip AMI
subset, and ideally a larger AMI split, with both VAD settings.

## AMI Held-Out Result

The June 20, 2026 AMI held-out run uses 24 additional 20-second meeting clips,
six each from `ES2002a`, `ES2002b`, `ES2002c`, and `ES2002d`.

| Model | Clips | WER | Cleaned WER | Mean warm RTF |
| --- | ---: | ---: | ---: | ---: |
| tiny | 24 | 0.3666 | 0.3144 | 0.0360 |
| base | 24 | 0.3493 | 0.2898 | 0.0717 |
| small | 24 | 0.2986 | 0.2336 | 0.1762 |

Artifacts:

```text
experiments/results/asr_benchmark_english_meeting_heldout_real.csv
experiments/results/asr_benchmark_english_meeting_heldout_summary_real.csv
experiments/results/asr_benchmark_english_meeting_heldout_small_real.csv
experiments/results/asr_benchmark_english_meeting_heldout_small_summary_real.csv
```

## Earnings-22 V3 Blind ASR Result

The six-file v3 blind Earnings-22 subset provides the ASR input for the RAG
v3 safety-gate experiment.

| Model | Clips | WER | Cleaned WER | Mean warm RTF |
| --- | ---: | ---: | ---: | ---: |
| tiny | 6 | 0.2519 | 0.2187 | 0.0255 |
| base | 6 | 0.2121 | 0.1777 | 0.0453 |

Artifacts:

```text
experiments/results/asr_benchmark_earnings22_v3_blind_6x180.csv
experiments/results/asr_benchmark_earnings22_v3_blind_6x180_summary.csv
```

## AISHELL-4 Mandarin Meeting Benchmark Subset

The June 21, 2026 AISHELL-4 test archive was fully downloaded and extracted
locally. Raw FLAC files and the 5.24 GB archive are ignored by Git. The
reported benchmark uses a fixed 60-clip subset: three 20-second excerpts from
each of the 20 test recordings. References are parsed from AISHELL-4 TextGrid
files.

| Model | Clips | CER | Mean warm RTF |
| --- | ---: | ---: | ---: |
| tiny | 60 | 0.6483 | 0.0636 |
| base | 60 | 0.5369 | 0.0711 |
| small | 60 | 0.4818 | 0.1367 |

Artifacts:

```text
data/manifests/aishell4_benchmark_60x20.csv
experiments/results/asr_benchmark_aishell4_60x20_real.csv
experiments/results/asr_benchmark_aishell4_60x20_summary_real.csv
```

The earlier 12-clip `mandarin_meeting_real.csv` manifest remains useful as a
small sanity track inside the 50-clip formal ASR manifest, but the 60-clip
subset is the stronger Mandarin meeting result to cite in the paper.

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

- The formal ASR study covers 50 clips, not complete corpus test partitions.
- The larger AISHELL-4 result covers 60 fixed clips from 20 recordings, not
  the complete AISHELL-4 test partition.
- FLEURS is the documented multilingual fallback, not Common Voice.
- AMI has only eight 20-second overlap excerpts from one meeting recording.
- Standard AMI WER is reference-style sensitive; cleaned WER is diagnostic.
- VAD conclusions remain diagnostic until rerun on the expanded AMI subset.
- Per-clip warm RTF excludes model initialization and model download.
- Local `cold_model_load_seconds` is not mobile cold-start latency.
- Runtime reflects the current CPU environment and cannot be reported as a
  mobile result.
- No statistical significance test is meaningful at this sample size.
- ASR-only results do not validate TalkWeaver diarization, overlap-aware
  correction, RAG recovery, or hallucination controls.

Do not claim these values as full FLEURS, full AMI, or state-of-the-art model
performance.
