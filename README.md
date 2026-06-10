# TalkWeaver

**An Overlap-Aware Multi-Speaker ASR System with Diarization-Structured LLM Correction**

*RAG-Enhanced Domain Term Recovery for Noisy Meeting Speech*

TalkWeaver is a research-inspired machine learning final project for noisy,
multi-speaker meetings. Its main focus is the interaction between automatic
speech recognition (ASR), speaker diarization, overlapping speech, and
constrained LLM correction. Retrieval-augmented generation (RAG) is a
supporting module used primarily to recover domain terms.

This repository is in Phase 1. It contains the project structure, research
foundation, deterministic mock pipeline, and UI placeholders. Heavy model
integration and real experiments are intentionally deferred.

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

The literature files are research templates at this stage. Sources and claims
must be verified before the final report.

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

`ffmpeg` and `libsndfile` are listed in `packages.txt` for later audio support.
Heavy packages such as `faster-whisper`, `pyannote.audio`, and LLM SDKs will be
optional and lazily imported in later phases.

## Environment Variables

```env
HF_TOKEN=
OPENAI_API_KEY=
DEEPSEEK_API_KEY=
QWEN_API_KEY=
ASR_MODEL_SIZE=medium
USE_MOCK_ASR=false
USE_MOCK_DIARIZATION=false
USE_MOCK_LLM=true
```

Never commit `.env` or credentials.

## Quickstart: Mock Mode

Mock mode is deterministic, requires no GPU or external API keys, and labels
its artifacts as demo data.

```bash
python scripts/run_pipeline.py --mock
python scripts/run_stage.py --stage asr --mock
python experiments/run_ablation.py --mock
python experiments/plot_results.py
```

Generated demo outputs are written under `outputs/` and
`experiments/results/`.

## Streamlit

```bash
streamlit run webapp/streamlit_app.py
```

The multipage app currently demonstrates the intended workflow and can run the
deterministic mock pipeline. Audio processing and real model controls are
Phase 2 and Phase 3 work.

## CLI

```bash
python scripts/run_pipeline.py --mock
python scripts/run_stage.py --stage asr --mock
python scripts/run_stage.py --stage diarization --mock
python scripts/generate_synthetic_overlap.py --mock
```

Real-audio mode is reserved for the next implementation phases:

```bash
python scripts/run_pipeline.py --audio data/demo/demo_meeting.wav
```

## Experiments

The planned ablation groups are:

- A: Whisper only
- B: preprocessing + Whisper
- C: Whisper + diarization + alignment
- D: structured LLM correction
- E: structured LLM correction + RAG glossary
- F: overlap-aware correction versus correction without overlap flags

Required metrics are WER, WDER or speaker-attribution error, Term Error Rate,
overlap error analysis, hallucinated correction count, and latency. Current
CSV output is labeled `mock_demo` and must not be cited as a real result.

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

## Limitations

- Real ASR, diarization, denoising, and LLM APIs are not integrated yet.
- Mock transcripts and metrics demonstrate interfaces, not model quality.
- The required `project/xutong_paper.pdf` was not available during Phase 1.
- Literature metadata and links require verification before final submission.
- Meeting summarization is secondary and must not replace the main
  diarization/overlap research evaluation.

## License

This project is available under the MIT License. Dataset and model licenses
must be reviewed separately before redistribution.
