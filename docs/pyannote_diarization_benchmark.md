# Pyannote Diarization Benchmark

## Purpose

This track is the automatic diarization counterpart to the existing
reference-assisted speaker/time evidence layer. It runs pyannote without mock
fallback and scores predicted turns against AMI reference speaker anchors.
The same scorer can also evaluate Mandarin meeting clips when speaker/time
reference anchors are available.

## Command

```bash
python experiments/run_pyannote_diarization_benchmark.py \
  --manifest data/manifests/english_meeting_heldout_real.csv \
  --output experiments/results/pyannote_diarization_heldout_real.csv \
  --summary-output experiments/results/pyannote_diarization_heldout_summary_real.csv \
  --predictions-dir outputs/diarization/pyannote_heldout_real

python experiments/run_pyannote_diarization_benchmark.py \
  --manifest data/manifests/aishell4_benchmark_60x20.csv \
  --output experiments/results/pyannote_diarization_aishell4_60x20_real.csv \
  --summary-output experiments/results/pyannote_diarization_aishell4_60x20_summary_real.csv \
  --predictions-dir outputs/diarization/pyannote_aishell4_60x20_real
```

## Metrics

Primary standard metrics use `pyannote.metrics`:

- DER with `collar=0.25`, `skip_overlap=false`;
- JER with `collar=0.25`, `skip_overlap=false`;
- diagnostic DER/JER with `skip_overlap=true`.

The existing project-level speaker label error is retained only as an
auxiliary diagnostic.

## Current Result

The current run uses a 24-clip AMI held-out subset balanced across
`ES2002a`, `ES2002b`, `ES2002c`, and `ES2002d`:

```text
metric_status=ok
num_clips=24
mean_der=0.106035
mean_jer=0.307202
mean_der_skip_overlap=0.081325
mean_jer_skip_overlap=0.209765
mean_overlap_f1=0.490214
mean_rtf=0.692477
```

Artifacts:

```text
experiments/results/pyannote_diarization_heldout_real.csv
experiments/results/pyannote_diarization_heldout_summary_real.csv
outputs/diarization/pyannote_heldout_real/
```

## AISHELL-4 Mandarin Meeting Benchmark Subset

The fixed AISHELL-4 benchmark subset contains 60 clips from 20 test
recordings. The diarization script scores only clips with at least two
reference speakers, producing 29 multi-speaker scored rows.

```text
metric_status=ok
num_clips=29
mean_der=0.326501
mean_jer=0.712577
mean_der_skip_overlap=0.326214
mean_jer_skip_overlap=0.693182
mean_project_speaker_label_error=0.362507
mean_overlap_f1=0.261905
mean_rtf=0.504550
```

Artifacts:

```text
experiments/results/pyannote_diarization_aishell4_60x20_real.csv
experiments/results/pyannote_diarization_aishell4_60x20_summary_real.csv
outputs/diarization/pyannote_aishell4_60x20_real/
```

This is a real AISHELL-4 benchmark subset result, not a full AISHELL-4
test-set score. It is strong enough to support a Mandarin meeting diarization
failure-mode discussion, especially the gap between speaker coverage and
overlap detection.

Implementation note: pyannote's optional `torchcodec` decoder is not working
in the current local environment, so `backend/diarization.py` loads WAV files
with `soundfile` and passes an in-memory waveform to the pyannote pipeline.
The torchcodec warning may still print during import, but the benchmark rows
are real pyannote outputs, not mock fallback.
