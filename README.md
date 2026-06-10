# TalkWeaver

**An Overlap-Aware Multi-Speaker ASR System with Diarization-Structured LLM Correction**

*RAG-Enhanced Domain Term Recovery for Noisy Meeting Speech*

TalkWeaver is a research-inspired machine learning final project for noisy,
multi-speaker meetings. Its main focus is the interaction between automatic
speech recognition (ASR), speaker diarization, overlapping speech, and
constrained LLM correction. Retrieval-augmented generation (RAG) is a
supporting module used primarily to recover domain terms.

This repository includes the Phase 3 audio preprocessing, ASR, speaker
diarization, word-speaker alignment, and overlap-detection baseline. It can
normalize real meeting audio to mono 16 kHz PCM WAV, run faster-whisper when
installed, and run pyannote.audio when both the package and `HF_TOKEN` are
available. The deterministic mock pipeline remains available without a GPU,
model download, or external credentials.

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

`ffmpeg` and `libsndfile` are listed in `packages.txt`. Standard PCM WAV files
can be processed with the Python standard library; `soundfile` handles common
libsndfile formats, while `pydub` plus FFmpeg provides a fallback for MP3 and
M4A.

Install the optional ASR, denoising, and diarization packages when needed:

```bash
pip install -r requirements-optional.txt
```

`faster-whisper` model weights are downloaded on first use when a model name is
selected. GPU execution additionally requires a compatible CUDA/cuDNN setup.
Real diarization uses
`pyannote/speaker-diarization-community-1` and requires an accepted model
license plus a Hugging Face access token in `HF_TOKEN`.

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
python scripts/run_stage.py --stage preprocess \
  --audio data/demo/demo_meeting.wav --mock
python scripts/run_stage.py --stage asr --mock
python scripts/run_stage.py --stage diarization --mock
python scripts/run_stage.py --stage align --mock
python scripts/run_stage.py --stage overlap --mock
python experiments/run_ablation.py --mock
python experiments/plot_results.py
```

Generated demo outputs are written under `outputs/` and
`experiments/results/`.

## Streamlit

```bash
streamlit run webapp/streamlit_app.py
```

The multipage app demonstrates the Phase 3 mock workflow. Real audio
preprocessing, ASR, and diarization are exposed through the CLI. LLM
correction, RAG term recovery, and summarization are intentionally not run yet.

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

- Real ASR requires the optional `faster-whisper` package and model weights.
- Denoising is skipped with a warning unless `noisereduce` is installed.
- Real diarization requires `pyannote.audio`, model access, and `HF_TOKEN`.
- Diarization falls back to clearly labeled mock turns when its dependency or
  token is unavailable.
- LLM correction and RAG term recovery are not integrated yet.
- Mock transcripts and metrics demonstrate interfaces, not model quality.
- The required `project/xutong_paper.pdf` was not available during Phase 1.
- Literature metadata and links require verification before final submission.
- Meeting summarization is secondary and must not replace the main
  diarization/overlap research evaluation.

## License

This project is available under the MIT License. Dataset and model licenses
must be reviewed separately before redistribution.
