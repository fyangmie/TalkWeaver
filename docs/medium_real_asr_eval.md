# Medium-Scale Real ASR Evaluation Runbook

This runbook prepares a medium formal ASR evaluation that can be run on a GPU
machine while keeping raw audio, model weights, and per-clip prediction JSONs
out of Git.

## Goal

The medium run expands the existing 17-clip formal subset into an approximately
100-clip real-audio ASR benchmark:

- 30 English FLEURS/Common Voice fallback clips;
- 30 French FLEURS/Common Voice fallback clips;
- 30 Mandarin FLEURS/Common Voice fallback clips;
- 10 AMI meeting excerpts.

The main reported metrics are WER for English/French/AMI and CER for Mandarin.
This benchmark primarily supports ASR accuracy claims. TalkWeaver's term rescue
and correction-audit claims still need their own focused evaluation.

## Local CPU Smoke

Use the smoke run to verify the scripts and output shape before pushing or
moving to a GPU machine. It runs one clip for each dataset/language pair.

Install optional runtime dependencies:

```bash
pip install faster-whisper opencc-python-reimplemented socksio
```

`socksio` is only needed when the machine downloads Hugging Face models through
a SOCKS proxy.

Prepare a small local smoke manifest:

```bash
python scripts/download_common_voice_subset.py \
  --languages en fr zh-CN \
  --max-clips-per-language 1 \
  --output-root data/raw/public/smoke_multilingual \
  --reference-root data/reference/public/smoke_multilingual \
  --manifest-out data/manifests/smoke_multilingual_real.csv

python scripts/download_meeting_subset.py \
  --dataset ami \
  --max-clips 1 \
  --output-root data/raw/public/smoke_english_meeting \
  --reference-root data/reference/public/smoke_english_meeting \
  --manifest-out data/manifests/smoke_english_meeting_real.csv

python scripts/build_formal_eval_manifest.py \
  --inputs \
    data/manifests/smoke_multilingual_real.csv \
    data/manifests/smoke_english_meeting_real.csv \
  --output data/manifests/smoke_eval_real.csv
```

Run CPU smoke:

```bash
python experiments/run_asr_benchmark.py \
  --manifest data/manifests/smoke_eval_real.csv \
  --models tiny \
  --device cpu \
  --compute-type int8 \
  --vad-filter true \
  --max-rows-per-dataset-language 1 \
  --benchmark-scope "medium formal evaluation smoke" \
  --output experiments/results/asr_medium/asr_benchmark_medium_smoke.csv \
  --predictions-dir .cache/talkweaver/asr_predictions_medium_smoke
```

Summarize and plot smoke results:

```bash
python experiments/summarize_asr_results.py \
  --input experiments/results/asr_medium/asr_benchmark_medium_smoke.csv \
  --output experiments/results/asr_medium/asr_benchmark_medium_smoke_summary.csv \
  --benchmark-scope "medium formal evaluation smoke"

python experiments/plot_asr_results.py \
  --input experiments/results/asr_medium/asr_benchmark_medium_smoke.csv \
  --output-dir assets/result_charts/asr_medium_smoke \
  --title-scope "Medium Formal Subset Smoke"
```

## GPU Full Run

Prepare the medium manifest on the GPU machine:

```bash
python scripts/download_common_voice_subset.py \
  --languages en fr zh-CN \
  --max-clips-per-language 30 \
  --output-root data/raw/public/medium_multilingual \
  --reference-root data/reference/public/medium_multilingual \
  --manifest-out data/manifests/medium_multilingual_real.csv

python scripts/download_meeting_subset.py \
  --dataset ami \
  --max-clips 10 \
  --output-root data/raw/public/medium_english_meeting \
  --reference-root data/reference/public/medium_english_meeting \
  --manifest-out data/manifests/medium_english_meeting_real.csv

python scripts/build_formal_eval_manifest.py \
  --inputs \
    data/manifests/medium_multilingual_real.csv \
    data/manifests/medium_english_meeting_real.csv \
  --output data/manifests/medium_eval_real.csv

python experiments/validate_manifest.py \
  --manifest data/manifests/medium_eval_real.csv \
  --require-real-files
```

Run faster-whisper on GPU:

```bash
python experiments/run_asr_benchmark.py \
  --manifest data/manifests/medium_eval_real.csv \
  --models tiny base small \
  --device cuda \
  --compute-type float16 \
  --vad-filter true \
  --benchmark-scope "medium formal evaluation" \
  --output experiments/results/asr_medium/asr_benchmark_medium_real.csv \
  --predictions-dir .cache/talkweaver/asr_predictions_medium_real
```

Summarize and plot:

```bash
python experiments/summarize_asr_results.py \
  --input experiments/results/asr_medium/asr_benchmark_medium_real.csv \
  --output experiments/results/asr_medium/asr_benchmark_medium_summary_real.csv \
  --benchmark-scope "medium formal evaluation"

python experiments/plot_asr_results.py \
  --input experiments/results/asr_medium/asr_benchmark_medium_real.csv \
  --output-dir assets/result_charts/asr_medium \
  --title-scope "Medium Formal Subset"
```

Optional workflow ablation on the same ASR predictions:

```bash
python experiments/run_workflow_ablation.py \
  --manifest data/manifests/medium_eval_real.csv \
  --predictions-dir .cache/talkweaver/asr_predictions_medium_real \
  --asr-model small \
  --output experiments/results/asr_medium/workflow_ablation_medium_real.csv \
  --maps-dir outputs/conversation_maps/medium_real \
  --variants all
```

## What To Commit

Commit:

- scripts and tests;
- manifest CSVs and reference TXT/JSON files when their licenses permit;
- aggregate CSV summaries;
- final charts;
- this runbook.

Do not commit:

- `data/raw/**` audio;
- `.cache/talkweaver/**` prediction JSONs and TXT files;
- model weights or Hugging Face caches;
- `.venv`.

## Disk Budget

Approximate disk use for the GPU run:

- raw medium audio and references: under 1 GB for the planned subset;
- faster-whisper tiny/base/small model cache: several GB depending on cache
  format and existing local models;
- prediction JSON/TXT cache: usually under a few hundred MB for 100 short clips.

Keep at least 10 GB free on the GPU machine before running the full benchmark.
