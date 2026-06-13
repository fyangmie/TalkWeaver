# TalkWeaver

**An Overlap-Aware Multi-Speaker ASR System with Diarization-Structured LLM Correction**

*RAG-Enhanced Domain Term Recovery for Noisy Meeting Speech*

> The original prototype has been reframed into TalkWeaver: AI Meeting Detective. PRD2.md is now the source of truth for the final project direction.

The final title is **TalkWeaver: AI Meeting Detective for Chaotic
Multi-Speaker Conversations**, with the subtitle **An evidence-grounded
conversation map for overlap, interruptions, misheard terms, and speaker
stances.** The working pipeline documented below is the v0 foundation and
will be preserved while the new investigation experience is implemented.

## Final Direction

TalkWeaver is now **AI Meeting Detective**. Multilingual evaluation and the
Level 1 whisper.cpp mobile ASR trade-off experiment are required research
tracks; a full native iOS or Android app is not required. Official and proxy
baseline feasibility is documented in
[`docs/baseline_feasibility.md`](docs/baseline_feasibility.md).

TalkWeaver is a research-inspired machine learning final project for noisy,
multi-speaker meetings. Its main focus is the interaction between automatic
speech recognition (ASR), speaker diarization, overlapping speech, and
constrained LLM correction. Retrieval-augmented generation (RAG) is a
supporting module used primarily to recover domain terms.

This repository includes the Phase 8 research pipeline, evaluation suite, and
review dashboard: audio
preprocessing, ASR, speaker diarization, word-speaker alignment, overlap
detection, local TF-IDF domain-term retrieval, diarization-structured
correction, and extractive meeting understanding. It can run faster-whisper
and pyannote.audio when their dependencies and credentials are available. The
deterministic mock pipeline requires no GPU, model download, vector database,
or external API key.

## Research Motivation

A meeting transcript is useful only when the system can answer who spoke,
when they spoke, what was said, and whether the evidence is uncertain because
speakers overlapped. Standard ASR does not solve all of those problems.
TalkWeaver therefore keeps timestamps, speaker labels, overlap flags,
confidence, raw text, corrected text, and retrieved terminology together in a
temporal-anchor transcript.

## Research Questions

1. Can diarization-structured prompting improve speaker-attributed transcript
   readability and speaker consistency?
2. Can overlap-aware uncertainty control reduce hallucinated corrections in
   overlapping speech regions?
3. Can RAG-based domain glossary retrieval reduce ASR errors on technical
   terms?
4. Does local audio preprocessing improve ASR robustness under noisy meeting
   conditions?

## Paper-Inspired Design

- **DiarizationLM:** compact diarization-structured prompting.
- **DM-ASR:** speaker-time conditioned segment correction.
- **TagSpeech / temporal-anchor work:** grounded "who spoke what and when"
  transcript records.
- **Retrieval-augmented ASR correction:** glossary candidates for rare and
  technical terms.

The local reference folder now contains the course anchor thesis,
DiarizationLM, Diarization-Aware Multi-Speaker ASR via LLMs, DM-ASR,
TagSpeech, and retrieval-augmented ASR correction. Their v1 product and
experiment mapping is defined in `PRD2.md`; the older literature notes still
need to be synchronized during the documentation phase.

## Architecture

```text
Meeting Audio
  -> preprocessing
  -> ASR
  -> speaker diarization
  -> word-speaker alignment
  -> overlap and confidence analysis
  -> temporal-anchor transcript
  -> domain-term retrieval
  -> constrained segment-level LLM correction
  -> secondary summary and action items
  -> evaluation and Streamlit review
```

The generated architecture figure is stored at `assets/architecture.png`.

## Installation

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

`ffmpeg` and `libsndfile` are listed in `packages.txt`. Standard PCM WAV files
can be processed with the Python standard library; `soundfile` handles common
libsndfile formats, while `pydub` plus FFmpeg provides a fallback for MP3 and
M4A.

Install the optional ASR, denoising, and diarization packages when needed:

```bash
python scripts/check_optional_dependencies.py
pip install -r requirements-optional.txt
```

Real ASR remains optional so mock and reference-assisted workflows work
without model packages. `faster-whisper` downloads CTranslate2 model weights
from Hugging Face on first use; these caches must not be committed. Start with
`--asr-model tiny --device cpu --compute-type int8`. GPU execution additionally
requires a compatible CUDA/cuDNN setup.
Real diarization uses
`pyannote/speaker-diarization-community-1` and requires an accepted model
license plus a Hugging Face access token in `HF_TOKEN`.

See [`docs/dependency_setup.md`](docs/dependency_setup.md) for the minimal,
CPU, GPU, cache, and troubleshooting policies.

## Environment Variables

```env
HF_TOKEN=
LLM_PROVIDER=deepseek
LLM_API_KEY=replace_me
LLM_MODEL=deepseek-chat
LLM_BASE_URL=https://api.deepseek.com
LLM_TEMPERATURE=0
LLM_TIMEOUT_SECONDS=30
ASR_MODEL_SIZE=medium
USE_MOCK_ASR=false
USE_MOCK_DIARIZATION=false
USE_MOCK_LLM=true
```

Never commit `.env` or credentials.

## Optional LLM API Setup

External LLM correction is optional. The deterministic rule fallback remains
the default for tests and reproducible offline runs.

```bash
cp .env.example .env
python scripts/run_llm_correction_smoke.py --mode rule_fallback
```

After setting `LLM_API_KEY`, provider, model, and base URL in the local
`.env`, run the strict smoke test:

```bash
python scripts/run_llm_correction_smoke.py --mode llm
```

Use `--mode llm_with_rule_fallback` only when an explicitly recorded fallback
is acceptable. Correction audits store provider, model, prompt version,
temperature, API usage, and fallback status, but never the key. Do not send
private or restricted transcripts to an external provider. See
[`docs/llm_api_setup.md`](docs/llm_api_setup.md) for configuration, cost,
privacy, and failure semantics.

## Downloading Small Formal Evaluation Subsets

Phase 2A-REAL provides size-capped acquisition scripts for real public
evaluation data. Raw audio and archives stay under `data/raw/public/` and are
ignored by Git.

```bash
python scripts/download_common_voice_subset.py \
  --languages en fr zh-CN --max-clips-per-language 5
python scripts/download_meeting_subset.py --dataset auto --max-clips 2
python scripts/download_mandarin_meeting_subset.py --dataset auto --max-clips 2

python scripts/build_formal_eval_manifest.py \
  --inputs \
    data/manifests/common_voice_multilingual_real.csv \
    data/manifests/english_meeting_real.csv \
    data/manifests/mandarin_meeting_real.csv \
  --output data/manifests/formal_eval_real.csv

python experiments/validate_manifest.py \
  --manifest data/manifests/formal_eval_real.csv \
  --require-real-files
```

The current reproducible subset contains 15 Google FLEURS clips, five each
for English, French, and Mandarin Chinese, plus two 20-second AMI meeting
excerpts with speaker anchors and overlap events. Common Voice partial access
was unavailable through the attempted official endpoint, so FLEURS is
explicitly labeled as the multilingual fallback. AISHELL-4 exceeds the 500 MB
download cap and AliMeeting still requires a verified manual subset route.

See [`docs/dataset_acquisition.md`](docs/dataset_acquisition.md) for exact
sources, licenses, commands, checksums, and current results, and
[`docs/manual_dataset_steps.md`](docs/manual_dataset_steps.md) for access
blockers. No raw dataset audio is committed.

## Running TalkWeaver Core Workflow

The Phase 2B method builds an auditable temporal-anchor `ConversationMap`
before any benchmark or ablation is run. It combines ASR text, speaker-time
evidence, overlap/interruption candidates, retrieved terminology, constrained
correction, and unsupported-change audits.

Mock smoke test:

```bash
python scripts/run_talkweaver_workflow.py \
  --manifest data/manifests/formal_eval_real.csv \
  --clip-id fleurs_en_1548 \
  --mock-models \
  --output outputs/conversation_maps/
```

Reference-assisted AMI workflow:

```bash
python scripts/run_talkweaver_workflow.py \
  --manifest data/manifests/formal_eval_real.csv \
  --clip-id ami_es2002a_01 \
  --asr-source reference \
  --diarization-source reference \
  --output outputs/conversation_maps/
```

Real ASR with reference diarization:

```bash
python scripts/check_optional_dependencies.py
pip install -r requirements-optional.txt

python scripts/run_talkweaver_workflow.py \
  --manifest data/manifests/formal_eval_real.csv \
  --clip-id ami_es2002a_01 \
  --asr-model tiny \
  --device cpu \
  --compute-type int8 \
  --diarization-source reference \
  --output outputs/conversation_maps/
```

Real model commands disable mock fallback and exit clearly when a dependency
or credential is unavailable. Reference-assisted mode is marked as
oracle/reference evidence, not automatic diarization. Output is written to
`outputs/conversation_maps/<clip_id>_conversation_map.json`.

See [`docs/talkweaver_workflow.md`](docs/talkweaver_workflow.md) for the
paper-to-module mapping, schema, evidence modes, correction rules, and current
limitations, and [`docs/dependency_setup.md`](docs/dependency_setup.md) for
optional real-ASR setup. This is a paper-inspired proxy workflow; it is not
claimed as a reproduction of DiarizationLM, DM-ASR, Diarization-Aware MS-ASR,
or TagSpeech.

## Running Real ASR Benchmark

Phase 2C evaluates real `faster-whisper` ASR before any TalkWeaver workflow
ablation. The current small formal subset contains 17 FLEURS/AMI clips and
uses WER for English/French and CER for Mandarin Chinese.

```bash
python experiments/run_asr_benchmark.py \
  --manifest data/manifests/formal_eval_real.csv \
  --models tiny base \
  --device cpu \
  --compute-type int8 \
  --vad-filter true \
  --output experiments/results/asr_benchmark_real.csv \
  --predictions-dir experiments/results/asr_predictions_real

python experiments/summarize_asr_results.py \
  --input experiments/results/asr_benchmark_real.csv \
  --output experiments/results/asr_benchmark_summary_real.csv

python experiments/plot_asr_results.py \
  --input experiments/results/asr_benchmark_real.csv \
  --output-dir assets/result_charts
```

Per-clip predictions remain local and ignored by Git. The small result CSVs
and charts are stored under `experiments/results/` and
`assets/result_charts/`. See
[`docs/asr_benchmark.md`](docs/asr_benchmark.md) for the protocol, measured
results, OpenCC Mandarin normalization, AMI cleaned-WER/VAD diagnostics, warm
RTF versus model-load timing, and limitations. These values are not
full-dataset performance claims.

## Running Speaker/Overlap Baseline

Phase 2D validates TalkWeaver's speaker-time evidence and overlap scoring
before workflow ablation:

```bash
python experiments/run_speaker_overlap_baseline.py \
  --manifest data/manifests/formal_eval_real.csv \
  --output experiments/results/speaker_overlap_baseline_real.csv

python experiments/run_reference_workflow_maps.py \
  --manifest data/manifests/formal_eval_real.csv \
  --dataset "AMI Meeting Corpus" \
  --output-dir outputs/conversation_maps/reference_assisted_real
```

The baseline compares a single-`UNKNOWN` no-diarization mode, an
oracle/reference-assisted mode, and optional real pyannote inference.
Reference-assisted results are workflow checks, not automatic diarization
claims. Pyannote is skipped explicitly unless its package and `HF_TOKEN`
model access are available. See
[`docs/speaker_overlap_baseline.md`](docs/speaker_overlap_baseline.md) for
metric definitions, measured AMI results, and limitations.

## Running TalkWeaver Workflow Ablation

Phase 2E holds the real Phase 2C `base` ASR predictions fixed and compares
seven downstream evidence variants:

```bash
python experiments/run_workflow_ablation.py \
  --manifest data/manifests/formal_eval_real.csv \
  --predictions-dir experiments/results/asr_predictions_real \
  --asr-model base \
  --output experiments/results/workflow_ablation_real.csv \
  --maps-dir outputs/conversation_maps/ablation_real \
  --variants all

python experiments/summarize_workflow_ablation.py \
  --input experiments/results/workflow_ablation_real.csv \
  --output experiments/results/workflow_ablation_summary_real.csv

python experiments/plot_workflow_ablation.py \
  --input experiments/results/workflow_ablation_real.csv \
  --output-dir assets/result_charts
```

The real run produced 119 rows across 17 clips and seven variants. It adds
speaker-time, overlap, review, retrieval, correction-audit, speaker-card, and
summary evidence without rerunning ASR. The public subset contains no
annotated technical-term failures, so no term rescue or text correction was
applied; WER/CER remained unchanged and unsupported changes remained zero.
See [`docs/workflow_ablation.md`](docs/workflow_ablation.md) for variant
definitions, measured results, charts, and claim limits.

## Running Controlled Term Rescue Experiment

Phase 2F uses 25 text-only controlled technical-term fixtures. These fixtures
are not public audio and are not reported as measured ASR output. They isolate
retrieval, correction, and Hallucination Watchdog behavior that the current
public subset cannot test.

Run the fully offline variants:

```bash
python experiments/run_term_rescue_experiment.py \
  --cases data/controlled_terms/term_rescue_cases.jsonl \
  --terms data/controlled_terms/reference_terms.json \
  --output experiments/results/term_rescue_controlled.csv \
  --candidates-output experiments/results/term_candidates_controlled.jsonl
```

Add strict real LLM rows only when the local `.env` is valid:

```bash
python experiments/run_term_rescue_experiment.py \
  --cases data/controlled_terms/term_rescue_cases.jsonl \
  --terms data/controlled_terms/reference_terms.json \
  --output experiments/results/term_rescue_controlled.csv \
  --candidates-output experiments/results/term_candidates_controlled.jsonl \
  --include-llm-if-configured
```

Summarize and plot:

```bash
python experiments/summarize_term_rescue.py \
  --input experiments/results/term_rescue_controlled.csv \
  --output experiments/results/term_rescue_summary_controlled.csv

python experiments/plot_term_rescue.py \
  --input experiments/results/term_rescue_controlled.csv \
  --output-dir assets/result_charts
```

The completed controlled run used DeepSeek `deepseek-chat` for the optional
LLM variant. Fused retrieval reached term F1 `1.0` with zero false-positive
terms on four common-word negative controls. Rule correction reduced the
mean controlled text error from `0.2880` to `0.0000`. Strict LLM correction
reduced it to `0.0812`; four API outputs were rejected by grounding checks,
retained the raw text, and were marked for review. These are controlled
fixture results, not public-dataset ASR claims. See
[`docs/term_rescue_experiment.md`](docs/term_rescue_experiment.md).

## Running Overlap-Aware Correction Safety Experiment

Phase 2G evaluates whether overlap and uncertainty evidence makes correction
more conservative. It uses 20 text-only controlled fixtures informed by the
duration range of the five AMI reference overlap events from Phase 2D. The
fixtures remain separate from public-audio ASR results.

Run the offline rule comparison:

```bash
python experiments/run_overlap_safety_experiment.py \
  --cases data/controlled_overlap/overlap_correction_cases.jsonl \
  --policy data/controlled_overlap/overlap_safety_policy.json \
  --output experiments/results/overlap_safety_controlled.csv
```

Add strict real LLM variants only when `.env` is configured:

```bash
python experiments/run_overlap_safety_experiment.py \
  --cases data/controlled_overlap/overlap_correction_cases.jsonl \
  --policy data/controlled_overlap/overlap_safety_policy.json \
  --output experiments/results/overlap_safety_controlled.csv \
  --include-llm-if-configured
```

Summarize and plot:

```bash
python experiments/summarize_overlap_safety.py \
  --input experiments/results/overlap_safety_controlled.csv \
  --output experiments/results/overlap_safety_summary_controlled.csv

python experiments/plot_overlap_safety.py \
  --input experiments/results/overlap_safety_controlled.csv \
  --output-dir assets/result_charts
```

The completed controlled run produced 80 rows. Safety pass rate increased
from `0.30` to `1.00` for rule correction and from `0.75` to `1.00` for
DeepSeek `deepseek-chat` when overlap/uncertainty evidence was enabled.
No accepted output contained unsupported changes, invented content, or
speaker-attribution changes. Overlap-aware variants exposed 14 review cases;
the rule variant rejected 7 risky corrections and the LLM variant rejected
12, including 7 pre-model policy rejections. See
[`docs/overlap_safety_experiment.md`](docs/overlap_safety_experiment.md).

## Quickstart: Mock Mode

Mock mode is deterministic, requires no GPU or external API keys, and labels
its artifacts as demo data.

```bash
python scripts/run_pipeline.py --mock
python scripts/run_stage.py --stage preprocess \
  --audio data/demo/demo_meeting.wav --mock
python scripts/run_stage.py --stage asr --mock
python scripts/run_stage.py --stage diarization --mock
python scripts/run_stage.py --stage align --mock
python scripts/run_stage.py --stage overlap --mock
python scripts/run_stage.py --stage rag --mock
python scripts/run_stage.py --stage correction --mock
python scripts/run_stage.py --stage summarize --mock
python experiments/run_ablation.py --mock
python experiments/plot_results.py
```

Generated demo outputs are written under `outputs/` and
`experiments/results/`.

## Running AI Meeting Detective Frontend

Phase 3A adds the investigation-oriented frontend over existing
`ConversationMap` JSON, experiment CSV, and chart artifacts:

```bash
streamlit run webapp/app.py
```

The app provides:

- Home / Project Story;
- Conversation Crime Scene;
- Speaker Timeline Detective;
- Cross-talk and Overlap Warning;
- Misheard Word Rescue;
- Hallucination Watchdog;
- Evidence Dashboard;
- Export / Report Preview.

It requires no API key and does not execute models. Real public-data results,
controlled text fixtures, and reference-assisted oracle speaker evidence are
visually distinguished. Markdown reports are exported locally to
`outputs/reports/<clip_id>_detective_report.md`.

Public AMI/FLEURS maps may show identical raw and corrected text because the
current public subset has no annotated technical-term correction targets.
TalkWeaver deliberately avoids unsupported edits. Use **Misheard Word
Rescue**, **Hallucination Watchdog**, and **Cross-talk and Overlap Warning**
to inspect controlled correction, rejection, and negative-control case files
with token-level diffs.

See [`docs/frontend.md`](docs/frontend.md) for consumed artifacts, page
behavior, claim boundaries, and limitations.

## Legacy Streamlit Review Workspace

```bash
streamlit run webapp/streamlit_app.py
```

The multipage app is a review dashboard for the research pipeline rather than
a generic meeting chatbot. It starts without a GPU, Hugging Face token, or LLM
API key and can generate all required artifacts in deterministic mock mode.

### UI Workflow

1. Open **Audio Input** to upload WAV, MP3, M4A, FLAC, or OGG audio. Uploads
   are saved locally under `outputs/uploads/`, played in the browser, decoded
   for metadata, and displayed as a downsampled waveform.
2. Open **Pipeline** and choose `Mock / demo` or `Real audio`. The stage
   toggles define the requested review plan; the integrated runner preserves
   required intermediate dependencies.
3. Select **Run Pipeline**. Real mode uses the uploaded path and falls back to
   clearly labeled mock ASR or diarization components if optional packages or
   credentials are unavailable.
4. Open **Transcript Review** to compare raw ASR, speaker-attributed temporal
   anchors, overlap intervals, and constrained corrections.
5. Use **RAG Domain-Term Recovery** for retrieved glossary terms, the
   extractive summary, sourced action items, and transcript-grounded QA.
6. Open **Metrics** to inspect result CSVs and generated charts. Mock scaffold
   rows remain explicitly unmeasured.

### Pages

- **Overview:** current execution mode, speaker timeline, overlap count, and
  correction audit.
- **Audio Input:** persisted upload, playback, metadata, and waveform.
- **Pipeline:** mock/real controls, stage selection, fallback status, and
  exported artifact paths.
- **Transcript Review:** raw ASR, diarization turns, overlap warnings,
  temporal-anchor fields, and raw-versus-corrected text.
- **RAG Domain-Term Recovery:** auxiliary terminology evidence and secondary
  transcript understanding.
- **Metrics:** WER, WDER or speaker attribution error, Term Error Rate,
  overlap analysis, latency, CSV tables, and result charts.

Mock mode can be started either from the Overview page, the Audio Input empty
state, or the Pipeline page. Its transcript, overlap, and correction outputs
are deterministic demo data and must not be reported as model evaluation.

## CLI

```bash
python scripts/run_pipeline.py --mock
python scripts/run_stage.py --stage preprocess \
  --audio data/demo/demo_meeting.wav --mock
python scripts/run_stage.py --stage asr \
  --audio data/demo/demo_meeting.wav --mock
python scripts/run_stage.py --stage diarization --mock
python scripts/run_stage.py --stage align --mock
python scripts/run_stage.py --stage overlap --mock
python scripts/run_stage.py --stage rag --mock
python scripts/run_stage.py --stage correction --mock
python scripts/run_stage.py --stage summarize --mock
python scripts/generate_synthetic_overlap.py --mock
```

Process a real audio file and save `data/processed/demo_meeting_mono_16k.wav`:

```bash
python scripts/run_stage.py \
  --stage preprocess \
  --audio data/demo/demo_meeting.wav
```

Request optional spectral-gating denoising:

```bash
python scripts/run_stage.py \
  --stage preprocess \
  --audio data/demo/demo_meeting.wav \
  --denoise
```

Run the ASR baseline directly and export raw JSON, Markdown, and metadata under
`outputs/transcripts/`:

```bash
python scripts/run_stage.py \
  --stage asr \
  --audio data/processed/demo_meeting_mono_16k.wav \
  --model-size medium \
  --language en
```

Run the integrated Phase 3 path:

```bash
python scripts/run_pipeline.py --audio data/demo/demo_meeting.wav
```

If `faster-whisper` is absent, ASR does not crash. It exports deterministic
segments labeled `mock_fallback`, including the reason. If `pyannote.audio` or
`HF_TOKEN` is unavailable, diarization likewise falls back to deterministic
two-speaker turns. Use `--mock` to request demo mode explicitly.

### Phase 2 Output

The raw transcript JSON is an array of segments:

```json
[
  {
    "start": 0.0,
    "end": 3.2,
    "text": "Today we are testing speaker diarization.",
    "words": [
      {"word": "Today", "start": 0.0, "end": 0.4}
    ]
  }
]
```

Each run also creates a readable Markdown transcript and a metadata JSON file
recording the ASR mode, model, language, device, and any fallback reason.

### Phase 3 Diarization and Overlap

Mock diarization contains two speakers and one deliberate overlap from
3.00-3.40 seconds. The output is clearly labeled `mock_demo` and must not be
treated as a real diarization result.

```bash
python scripts/run_stage.py --stage diarization --mock
python scripts/run_stage.py --stage overlap --mock
python scripts/run_stage.py --stage align --mock
```

The stage commands create or reuse mock ASR and diarization artifacts when
needed. Speaker turns and overlap warnings are exported under
`outputs/diarization/`. Speaker-attributed JSON and Markdown are exported under
`outputs/transcripts/`.

Alignment assigns each ASR word using the midpoint of its start and end
timestamps. A midpoint inside one turn receives that speaker; a midpoint
inside multiple turns receives `speaker: "OVERLAP"` and all active speakers;
an uncovered midpoint receives `speaker: "UNKNOWN"`.

The temporal-anchor transcript uses this structure:

```json
{
  "start": 3.0,
  "end": 3.2,
  "speaker": "OVERLAP",
  "speakers": ["SPEAKER_00", "SPEAKER_01"],
  "raw_text": "The",
  "corrected_text": "",
  "overlap": true,
  "confidence": 0.55,
  "retrieved_terms": []
}
```

`corrected_text` and `retrieved_terms` remain empty in Phase 3. They are
reserved for the later constrained LLM and domain-term recovery phases.

### Phase 4/5 RAG and Correction

TalkWeaver loads every Markdown file under `docs/knowledge_base/`, splits
glossary rows and explanatory paragraphs into local chunks, and ranks them
with an in-process TF-IDF cosine index. No external vector database is
required.

```bash
python scripts/run_stage.py --stage rag --mock
python scripts/run_stage.py --stage correction --mock
python scripts/run_stage.py --stage summarize --mock
python scripts/run_pipeline.py --mock
```

Each correction prompt contains the timestamp, speaker label, active speakers,
overlap flag, confidence, raw text, and retrieved terms. Correction runs
segment by segment and keeps an audit trail. The deterministic fallback
supports:

```text
piano note -> pyannote
diary station -> diarization
where -> WER
the ear -> DER
rack -> RAG
```

If `USE_MOCK_LLM=true`, `--mock` is supplied, or no configured provider key is
available, correction uses deterministic glossary rules. When API correction
is enabled, TalkWeaver supports OpenAI-compatible chat-completion endpoints
for OpenAI, DeepSeek, and Qwen. Returned text is rejected if it adds
unsupported vocabulary or excessive content, then falls back to deterministic
rules.

Outputs are written to:

- `outputs/transcripts/*_rag_enriched.json`
- `outputs/corrected_transcripts/*_corrected.json`
- `outputs/corrected_transcripts/*_corrected.md`
- `outputs/summaries/*_summary.json`
- `outputs/summaries/*_summary.md`

Overlap segments remain uncertain even after correction. Meeting summaries
are extractive, action items retain their source speaker and timestamp, and
the secondary QA helper returns a supporting transcript segment rather than
generating unsupported answers.

## Experiments

Run the deterministic experiment workflow:

```bash
python experiments/run_ablation.py --mock
python experiments/plot_results.py
pytest
```

The ablation groups are:

- A: Whisper only
- B: preprocessing + Whisper
- C: Whisper + diarization + alignment
- D: structured LLM correction
- E: structured LLM correction + RAG glossary
- F: overlap-aware correction versus correction without overlap flags

Required metrics are WER, WDER or speaker-attribution error, Term Error Rate,
overlap error analysis, hallucinated correction count, and latency. The WDER
column is a clearly documented project-level temporal speaker-error
approximation, not a full DER/WDER implementation.

Use the individual evaluators with real references:

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

python experiments/evaluate_latency.py --audio data/demo/meeting.wav
```

Generated result files:

- `experiments/results/ablation_results.csv`
- `experiments/results/term_error_results.csv`
- `experiments/results/latency_results.csv`
- `assets/result_charts/wer_comparison.png`
- `assets/result_charts/wder_comparison.png`
- `assets/result_charts/term_error_comparison.png`
- `assets/result_charts/latency_comparison.png`
- `assets/result_charts/hallucination_comparison.png`

Every included row currently uses `is_mock=true`. The numbers are calculated
from the deterministic built-in reference, so they test metric direction and
artifact plumbing only. Real experiments must replace or separately store
results from annotated audio; mock charts must not be cited as model quality.

## Final Project Status

- [x] Research-oriented repository structure
- [x] Audio preprocessing and faster-whisper integration
- [x] Pyannote diarization with deterministic fallback
- [x] Word-speaker alignment and overlap detection
- [x] Temporal-anchor transcript export
- [x] Diarization-structured correction and audit trail
- [x] Local TF-IDF domain-term recovery
- [x] Multi-page Streamlit review dashboard
- [x] WER, speaker-error approximation, TER, overlap, hallucination, latency
- [x] Mock ablation CSVs and five labeled charts
- [x] Project report, blog article, literature notes, and video script
- [x] AI Meeting Detective Phase 3A data layer and core investigation pages
- [x] Inspect the local course anchor paper and related reference PDFs
- [ ] Synchronize the full literature notes with the PRD2 paper mapping
- [ ] Collect or license real evaluation audio and annotations
- [ ] Run reference-backed groups A-F and replace demo-only conclusions
- [ ] Add final screenshots, member contributions, and recorded video

## Repository Guide

- `backend/`: pipeline interfaces and deterministic mock implementations.
- `webapp/`: Streamlit entry point, pages, and visual components.
- `scripts/`: command-line entry points.
- `experiments/`: evaluation and ablation entry points.
- `docs/`: literature, research questions, experiment plan, and knowledge base.
- `data/`: local demo, raw, processed, reference, and synthetic data.
- `outputs/`: generated transcripts, metrics, summaries, and exports.

## Screenshots and Charts

Place final UI captures in `assets/screenshots/` and generated evaluation
figures in `assets/result_charts/`. Placeholder directories are tracked with
`.gitkeep`.

Planned final captures:

- `assets/screenshots/01_upload.png`
- `assets/screenshots/02_pipeline.png`
- `assets/screenshots/03_overlap_review.png`
- `assets/screenshots/04_rag_terms.png`
- `assets/screenshots/05_metrics.png`

## Limitations

- Real ASR requires the optional `faster-whisper` package and model weights.
- Denoising is skipped with a warning unless `noisereduce` is installed.
- Real diarization requires `pyannote.audio`, model access, and `HF_TOKEN`.
- Diarization falls back to clearly labeled mock turns when its dependency or
  token is unavailable.
- TF-IDF retrieval is intentionally lightweight and does not perform semantic
  embedding retrieval.
- API correction requires a configured provider key and network access.
- Lexical grounding reduces hallucination risk but does not replace human
  review, especially for overlapping speech.
- Mock transcripts and metrics demonstrate interfaces, not model quality.
- The course anchor paper is now available under `参考文献/`, but the older
  v0 report and reading note still require a PRD2-aligned revision.
- Literature metadata and links require verification before final submission.
- Meeting summarization is secondary and must not replace the main
  diarization/overlap research evaluation.

## License

This project is available under the MIT License. Dataset and model licenses
must be reviewed separately before redistribution.
