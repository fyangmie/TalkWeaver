# PRD.md — TalkWeaver Final Project Product Requirements Document

> Project codename: **TalkWeaver**  
> Final title: **TalkWeaver: An Overlap-Aware Multi-Speaker ASR System with Diarization-Structured LLM Correction**  
> Subtitle: **RAG-Enhanced Domain Term Recovery for Noisy Meeting Speech**  
> Target: Machine Learning final project with research-driven engineering implementation, GitHub repository, experiments, Streamlit demo, and English presentation video.

---

## 0. Executive Summary

TalkWeaver is a research-inspired machine learning final project focused on **speaker diarization**, **overlapping speech / cross-speech**, and **LLM + ASR synergy**.

The system takes a noisy multi-speaker meeting audio file as input and produces:

1. raw ASR transcript;
2. speaker-attributed transcript;
3. overlap/cross-speech warnings;
4. diarization-structured transcript format;
5. constrained LLM-corrected transcript;
6. RAG-enhanced domain term recovery;
7. meeting summary, action items, and QA;
8. evaluation metrics and ablation charts.

The project must **not** be implemented as a simple Whisper demo or a generic meeting summary app. Its main research focus is:

> How can recent ideas from LLM-based diarization correction, diarization-aware multi-speaker ASR, and RAG-based ASR correction be adapted into a lightweight, reproducible engineering pipeline for noisy, overlapping multi-speaker meetings?

---

## 1. Project Context and Course Requirements

The teacher's project topic emphasizes:

- Speaker diarization;
- Multi-speaker overlapping speech / cross-speech;
- LLM + ASR synergy;
- Reading the provided `xutong_paper.pdf` under the project folder;
- Exploring related recent research;
- Identifying limitations or missing components;
- Proposing improvements;
- Designing and testing more advanced or innovative approaches;
- Optional RAG integration.

Therefore, TalkWeaver must be built as:

```text
Recent paper survey
→ problem / limitation identification
→ paper-inspired component design
→ working engineering pipeline
→ experiments and ablation study
→ GitHub repository
→ English video presentation
```

---

## 2. Main Project Direction

### 2.1 Main Topic

The main topic is:

```text
Topic 1:
Speaker diarization, cross-speech, and LLM + ASR synergy
```

### 2.2 Supporting Topic

A supporting module is:

```text
Topic 3:
Local audio preprocessing for better ASR performance
```

### 2.3 Auxiliary Innovation

RAG is included as an auxiliary module:

```text
RAG is used for domain-specific ASR error correction and meeting understanding.
RAG must not become the main project topic.
```

---

## 3. What This Project Is and Is Not

### 3.1 This Project Is

TalkWeaver is:

- an overlap-aware multi-speaker ASR system;
- a diarization-structured LLM correction pipeline;
- a lightweight engineering adaptation of recent multi-speaker ASR research;
- a Streamlit-based review and evaluation dashboard;
- a research-style project with literature review, research questions, experiments, metrics, and limitations.

### 3.2 This Project Is Not

TalkWeaver is not:

- a simple Whisper transcription demo;
- a generic RAG meeting chatbot;
- a mobile ASR deployment project;
- a pet translation project;
- a pure Streamlit UI project;
- an unsupported claim of outperforming top conference models;
- a project that fabricates experimental results.

---

## 4. Research Motivation

Real-world meetings are difficult for ASR systems because they often contain:

- multiple speakers;
- background noise;
- interruptions;
- overlapping speech;
- domain-specific technical terms;
- incomplete or incorrect speaker attribution;
- ASR errors on rare words and terminology.

A standard ASR system only answers:

```text
What was said?
```

However, real multi-speaker meeting understanding requires:

```text
Who spoke?
When did they speak?
What did they say?
Did multiple speakers overlap?
Can errors be safely corrected?
What does the meeting mean?
```

---

## 5. Required Paper-Driven Research Foundation

The repository must include a literature review and paper reading notes. The project should use the following paper categories.

### 5.1 Required Anchor Paper

The provided course paper:

```text
project/xutong_paper.pdf
```

Codex must:

1. check whether this file exists;
2. if it exists, summarize it in `docs/paper_reading_notes/00_xutong_paper.md`;
3. identify its key problem, method, limitation, and how TalkWeaver responds to it;
4. if the file is absent, create a placeholder note saying the file was not available locally and must be added by the student later;
5. never invent details from the paper if it cannot be read.

### 5.2 Recommended Related Papers / Works

Use these works as research inspiration and cite them in `docs/literature_review.md` and `PROJECT_REPORT.md`.

| Paper / Work                                 | Status                     | Key Idea                                                     | Our Adaptation                                               |
| -------------------------------------------- | -------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| DiarizationLM                                | INTERSPEECH 2024           | Use compact textual representation of ASR + diarization output for LLM post-processing | Diarization-structured prompting with timestamps, speakers, overlap flags, confidence, and retrieved terms |
| Diarization-aware Multi-Speaker ASR via LLMs | 2025 arXiv / frontier work | Use diarization structure to improve multi-speaker ASR, especially overlapping speech | Lightweight reproducible pipeline using Whisper + pyannote + structured prompting |
| DM-ASR                                       | 2026 arXiv / frontier work | Speaker- and time-conditioned queries for multi-speaker ASR  | Speaker/time-conditioned segment-wise LLM correction         |
| TagSpeech                                    | 2026 arXiv / frontier work | Temporal anchor grounding for “who spoke what and when”      | Temporal-anchor JSON transcript format                       |
| Retrieval-Augmented ASR Correction           | 2024 arXiv / frontier work | Retrieve candidate rare entities to correct ASR errors       | RAG-based domain term recovery for ASR/diarization terms     |

### 5.3 Required Literature Review Files

Create and maintain:

```text
docs/literature_review.md
docs/research_questions.md
docs/paper_reading_notes/00_xutong_paper.md
docs/paper_reading_notes/01_diarizationlm.md
docs/paper_reading_notes/02_dm_asr.md
docs/paper_reading_notes/03_tagspeech.md
docs/paper_reading_notes/04_rag_asr_correction.md
```

Each paper note must follow this template:

```markdown
# Paper Title

## Source
Venue/status/year/link.

## Problem
What problem does this paper solve?

## Key Idea
What is the core technical idea?

## Limitation
What is still missing or difficult?

## Our Adaptation
How TalkWeaver adapts this idea.

## Implementation Mapping
Relevant project files.
```

---

## 6. Research Questions

TalkWeaver must explicitly answer the following research questions.

### RQ1: Diarization-Structured Prompting

```text
Can diarization-structured prompting improve speaker-attributed transcript readability and speaker consistency?
```

### RQ2: Overlap-Aware Uncertainty

```text
Can overlap-aware uncertainty control reduce hallucinated corrections in overlapping speech regions?
```

### RQ3: RAG-Based Domain Term Recovery

```text
Can RAG-based domain glossary retrieval reduce ASR errors on technical terms?
```

### RQ4: Audio Preprocessing

```text
Does local audio preprocessing improve ASR robustness under noisy meeting conditions?
```

---

## 7. Proposed Contributions

The final report and video should describe the project contributions as follows.

### Contribution 1: Diarization-Structured Transcript Format

Convert ASR and diarization outputs into a structured transcript:

```json
{
  "start": 12.40,
  "end": 15.80,
  "speaker": "SPEAKER_01",
  "raw_text": "The rack system improves where.",
  "corrected_text": "The RAG system improves WER.",
  "overlap": true,
  "confidence": 0.62,
  "retrieved_terms": ["RAG", "WER"]
}
```

### Contribution 2: Speaker-Time Conditioned LLM Correction

Correct transcript segment by segment using:

- speaker label;
- start/end timestamp;
- raw text;
- overlap flag;
- confidence;
- retrieved domain terms.

### Contribution 3: Overlap-Aware Uncertainty Control

Detect overlapping speech and require conservative correction in overlap regions.

### Contribution 4: RAG-Based Domain Term Recovery

Retrieve domain terms such as:

- pyannote.audio;
- speaker diarization;
- overlapping speech;
- ASR;
- WER;
- DER;
- WDER;
- RAG;
- faster-whisper;
- VAD;
- LLM correction.

Use retrieved terms to correct common ASR mistakes:

```text
piano note → pyannote
diary station → diarization
where → WER
the ear → DER
rack → RAG
```

### Contribution 5: Multi-Level Evaluation

Evaluate with:

- WER;
- WDER or speaker attribution error;
- Term Error Rate;
- overlap error analysis;
- hallucinated correction count;
- latency.

---

## 8. System Workflow

The final pipeline must follow this order:

```text
Raw Meeting Audio
→ Audio Preprocessing
→ ASR Transcription
→ Speaker Diarization
→ Word-Speaker Alignment
→ Overlap Detection
→ Temporal-Anchor Transcript Export
→ RAG Domain-Term Retrieval
→ Diarization-Structured LLM Correction
→ Meeting Summary / Action Items / QA
→ Evaluation Metrics
→ Streamlit Review Dashboard
```

---

## 9. Module Requirements

### 9.1 Audio Preprocessing

File:

```text
backend/preprocessing.py
```

Required features:

- load wav/mp3/m4a when possible;
- convert to mono;
- resample to 16kHz;
- normalize volume;
- optional denoising;
- optional VAD / silence trimming;
- save processed audio.

### 9.2 ASR Transcription

File:

```text
backend/asr.py
```

Required features:

- use `faster-whisper` when installed;
- support model size configuration;
- output segment-level timestamps;
- output word-level timestamps when available;
- export JSON and Markdown;
- include fallback/mock output mode for development environments.

### 9.3 Speaker Diarization

File:

```text
backend/diarization.py
```

Required features:

- use `pyannote.audio` when `HF_TOKEN` is available;
- provide mock diarization mode when no token is available;
- export speaker turn segments;
- support multiple speakers;
- avoid crashing when pyannote is unavailable.

### 9.4 Word-Speaker Alignment

File:

```text
backend/alignment.py
```

Required features:

- assign ASR words to speakers using timestamp midpoint;
- group speaker-attributed words into readable segments;
- mark unknown speaker when no segment matches;
- support overlap detection integration.

### 9.5 Overlap Detection

Files:

```text
backend/overlap.py
backend/confidence.py
```

Required features:

- detect overlapping speaker segments;
- mark overlap regions;
- assign confidence or uncertainty labels;
- export overlap warnings.

### 9.6 Prompting

File:

```text
backend/prompting.py
```

Required features:

- create diarization-structured prompts;
- implement speaker-time conditioned prompts;
- include retrieved domain terms;
- include overlap uncertainty constraints;
- preserve timestamps and speaker labels.

### 9.7 LLM Correction

File:

```text
backend/llm_correction.py
```

Required features:

- support API-based LLM correction if API key exists;
- support safe mock/rule-based correction if no API key exists;
- preserve timestamps;
- preserve speaker labels unless strong evidence exists;
- avoid hallucination;
- mark uncertain overlap segments;
- use retrieved terms.

### 9.8 RAG Domain Term Recovery

File:

```text
backend/rag.py
```

Required features:

- load markdown documents from `docs/knowledge_base/`;
- load domain terms;
- implement simple TF-IDF retrieval fallback;
- optionally support vector database later;
- return candidate correction terms for each transcript segment.

### 9.9 Meeting Understanding

Files:

```text
backend/summarizer.py
backend/rag.py
```

Required features:

- generate meeting summary;
- generate action items;
- generate keywords;
- support simple QA over transcript;
- keep this secondary to the ASR/diarization focus.

### 9.10 Export

File:

```text
backend/export.py
```

Required features:

- export raw transcript;
- export speaker-attributed transcript;
- export corrected transcript;
- export summary;
- export metrics;
- support JSON and Markdown.

### 9.11 Pipeline Orchestration

File:

```text
backend/pipeline.py
```

Required features:

- run the full pipeline;
- allow stage-by-stage execution;
- save intermediate outputs;
- handle missing dependencies gracefully;
- provide clear logs.

---

## 10. Streamlit Web App Requirements

The UI must be a multi-page Streamlit app.

Main entry:

```text
webapp/streamlit_app.py
```

Pages:

```text
webapp/pages/1_Upload.py
webapp/pages/2_Pipeline.py
webapp/pages/3_Review_Transcript.py
webapp/pages/4_RAG_Assistant.py
webapp/pages/5_Metrics.py
```

### 10.1 Upload Page

Must include:

- project title;
- audio uploader;
- audio preview/player;
- basic audio information.

### 10.2 Pipeline Page

Must include options:

- enable preprocessing;
- enable ASR;
- enable diarization;
- enable overlap detection;
- enable RAG;
- enable LLM correction;
- select mock mode vs real mode;
- run full pipeline.

### 10.3 Review Transcript Page

Must display:

- raw transcript;
- speaker-attributed transcript;
- corrected transcript;
- overlap warnings;
- timestamped segments.

### 10.4 RAG Assistant Page

Must display:

- retrieved domain terms;
- meeting summary;
- action items;
- QA input box.

This page must not dominate the project narrative.

### 10.5 Metrics Page

Must display:

- WER;
- WDER or speaker attribution error;
- Term Error Rate;
- latency;
- ablation charts;
- result CSV preview.

---

## 11. CLI Requirements

Implement command-line scripts:

```text
scripts/run_pipeline.py
scripts/run_stage.py
scripts/generate_synthetic_overlap.py
scripts/export_submission_assets.py
```

Example commands:

```bash
python scripts/run_pipeline.py --audio data/demo/demo_meeting.wav --mock
python scripts/run_stage.py --stage asr --audio data/demo/demo_meeting.wav
python scripts/run_stage.py --stage diarization --audio data/demo/demo_meeting.wav --mock
python scripts/generate_synthetic_overlap.py
python experiments/run_ablation.py --mock
python experiments/plot_results.py
```

---

## 12. Experiments

The experiments must align with the research questions.

### Experiment 1: Preprocessing

Comparison:

```text
Raw audio + Whisper
vs
Preprocessed audio + Whisper
```

Metrics:

- WER;
- latency.

### Experiment 2: Structured Prompting

Comparison:

```text
Plain transcript correction
vs
Diarization-structured transcript correction
```

Metrics:

- WDER or speaker attribution consistency;
- readability/manual error notes.

### Experiment 3: Overlap Uncertainty

Comparison:

```text
LLM correction without overlap flag
vs
LLM correction with overlap uncertainty
```

Metrics:

- overlap segment error;
- hallucinated correction count;
- manual error analysis.

### Experiment 4: RAG Term Recovery

Comparison:

```text
Whisper only
vs
Whisper + LLM correction
vs
Whisper + LLM correction + RAG glossary
```

Metrics:

- Term Error Rate;
- WER.

---

## 13. Evaluation Scripts

Create:

```text
experiments/evaluate_wer.py
experiments/evaluate_wder.py
experiments/evaluate_terms.py
experiments/evaluate_latency.py
experiments/run_ablation.py
experiments/plot_results.py
```

Expected outputs:

```text
experiments/results/ablation_results.csv
experiments/results/term_error_results.csv
experiments/results/latency_results.csv
assets/result_charts/wer_comparison.png
assets/result_charts/wder_comparison.png
assets/result_charts/term_error_comparison.png
assets/result_charts/latency_comparison.png
```

Never fabricate metrics. If real experiments are not yet run, label outputs as mock/demo results.

---

## 14. Data Strategy

Use three types of data.

### 14.1 Self-Recorded Demo Audio

Record 3-6 students simulating a project meeting.

Include:

- English speech;
- technical terms;
- interruptions;
- overlap;
- background noise.

Example phrases:

```text
We should use pyannote for speaker diarization.
The RAG glossary can reduce term errors.
Whisper may confuse WER with where.
Two speakers are talking at the same time now.
```

### 14.2 Synthetic Overlap Data

Use scripts to generate controlled overlap examples by mixing speaker audio clips and adding noise.

### 14.3 Public Datasets

Optional but desirable:

- AMI Meeting Corpus;
- LibriMix / Libri2Mix;
- VoxConverse;
- AliMeeting.

If public data is not used, explain why and rely on self-recorded and synthetic controlled data.

---

## 15. Repository Structure

The project must use this structure:

```text
TalkWeaver/
├── README.md
├── PROJECT_REPORT.md
├── BLOG_ARTICLE.md
├── requirements.txt
├── packages.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── .gitignore
├── LICENSE
│
├── webapp/
│   ├── streamlit_app.py
│   ├── components/
│   │   ├── audio_player.py
│   │   ├── waveform_viewer.py
│   │   ├── speaker_timeline.py
│   │   ├── transcript_viewer.py
│   │   └── metrics_dashboard.py
│   └── pages/
│       ├── 1_Upload.py
│       ├── 2_Pipeline.py
│       ├── 3_Review_Transcript.py
│       ├── 4_RAG_Assistant.py
│       └── 5_Metrics.py
│
├── backend/
│   ├── __init__.py
│   ├── config.py
│   ├── pipeline.py
│   ├── job_manager.py
│   ├── preprocessing.py
│   ├── asr.py
│   ├── diarization.py
│   ├── alignment.py
│   ├── overlap.py
│   ├── confidence.py
│   ├── prompting.py
│   ├── llm_correction.py
│   ├── rag.py
│   ├── summarizer.py
│   └── export.py
│
├── scripts/
│   ├── run_pipeline.py
│   ├── run_stage.py
│   ├── prepare_demo_audio.py
│   ├── generate_synthetic_overlap.py
│   ├── batch_transcribe.py
│   ├── batch_diarize.py
│   └── export_submission_assets.py
│
├── experiments/
│   ├── run_ablation.py
│   ├── evaluate_wer.py
│   ├── evaluate_wder.py
│   ├── evaluate_terms.py
│   ├── evaluate_latency.py
│   ├── plot_results.py
│   └── results/
│
├── notebooks/
│   ├── 00_literature_mapping.ipynb
│   ├── 01_asr_baseline.ipynb
│   ├── 02_speaker_diarization.ipynb
│   ├── 03_overlap_analysis.ipynb
│   ├── 04_rag_correction.ipynb
│   └── 05_ablation_study.ipynb
│
├── docs/
│   ├── PRD.md
│   ├── AGENTS.md
│   ├── literature_review.md
│   ├── research_questions.md
│   ├── experiment_plan.md
│   ├── model_cards.md
│   ├── glossary.md
│   ├── video_script.md
│   ├── contribution.md
│   ├── github_url.md
│   ├── paper_reading_notes/
│   └── knowledge_base/
│       ├── project_requirement.md
│       ├── asr_background.md
│       ├── diarization_background.md
│       ├── rag_background.md
│       └── domain_terms.md
│
├── data/
│   ├── demo/
│   ├── raw/
│   ├── processed/
│   ├── reference/
│   └── synthetic/
│
├── assets/
│   ├── architecture.png
│   ├── screenshots/
│   ├── result_charts/
│   └── demo_gifs/
│
└── outputs/
    ├── transcripts/
    ├── diarization/
    ├── corrected_transcripts/
    ├── summaries/
    ├── metrics/
    └── exports/
```

---

## 16. Environment Strategy

Use:

```text
Local RTX 5060:
- development;
- Streamlit demo;
- small/medium faster-whisper;
- video recording.

Server GPU:
- large model inference;
- pyannote batch diarization;
- ablation experiments;
- chart generation.

GitHub:
- synchronization;
- commit history;
- final code review.
```

---

## 17. Required Documentation

Create and maintain:

```text
README.md
PROJECT_REPORT.md
BLOG_ARTICLE.md
docs/literature_review.md
docs/research_questions.md
docs/experiment_plan.md
docs/video_script.md
docs/contribution.md
docs/github_url.md
```

### 17.1 README.md

Must include:

- project overview;
- research motivation;
- paper-inspired design;
- system architecture;
- installation;
- environment variables;
- quickstart;
- CLI usage;
- Streamlit usage;
- mock mode;
- experiment commands;
- screenshots/charts;
- limitations.

### 17.2 PROJECT_REPORT.md

Must include:

1. Abstract;
2. Introduction;
3. Related Work;
4. Research Gaps;
5. Method;
6. System Architecture;
7. Experiments;
8. Results;
9. Error Analysis;
10. Limitations;
11. Future Work;
12. Conclusion.

### 17.3 BLOG_ARTICLE.md

A readable story-style article:

```text
When Whisper Meets a Noisy Meeting:
Building an Overlap-Aware Multi-Speaker ASR System
```

### 17.4 video_script.md

10+ minute English video script.

### 17.5 contribution.md

Team contribution table.

### 17.6 github_url.md

Repository URL and note that commit history is reviewable.

---

## 18. GitHub Requirements

The teacher will check commit history.

Target:

```text
Minimum: 40 meaningful commits
Recommended: 60+ meaningful commits
Ideal: 70+ meaningful commits
```

Commit style examples:

```bash
git commit -m "docs: add literature review and research questions"
git commit -m "chore: initialize TalkWeaver project structure"
git commit -m "feat: implement preprocessing and ASR baseline"
git commit -m "feat: add diarization alignment and overlap detection"
git commit -m "feat: add diarization-structured LLM correction"
git commit -m "feat: add RAG-based domain term recovery"
git commit -m "feat: build multi-page Streamlit review dashboard"
git commit -m "exp: add ablation study and evaluation charts"
git commit -m "docs: add final report and video script"
```

Do not commit:

- API keys;
- `.env`;
- large raw audio;
- large videos;
- model weights;
- private tokens.

---

## 19. Mock Mode Requirement

The project must run even without:

- Hugging Face token;
- OpenAI/DeepSeek/Qwen key;
- CUDA GPU;
- pyannote model access.

Mock mode must support:

- fake but structured ASR output;
- fake diarization segments;
- deterministic overlap warnings;
- rule-based domain term correction;
- demo metrics.

Mock/demo outputs must be clearly labeled as mock/demo and not presented as real experiment results.

---

## 20. Acceptance Criteria

The project is considered acceptable when:

1. repository structure is complete;
2. `streamlit run webapp/streamlit_app.py` starts successfully;
3. `python scripts/run_pipeline.py --mock` runs successfully;
4. raw transcript, speaker transcript, corrected transcript, and summary can be generated;
5. overlap warnings are displayed;
6. RAG domain terms are retrieved;
7. experiment scripts can generate at least demo metrics;
8. README contains usage instructions;
9. PROJECT_REPORT contains paper-driven method and experiments;
10. GitHub history contains meaningful incremental commits.

The project is considered strong when:

1. real faster-whisper ASR works;
2. real pyannote diarization works;
3. real or API-based LLM correction works;
4. self-recorded meeting demo is processed;
5. ablation results are generated from real data;
6. result charts are displayed in the app;
7. video clearly explains related work, gaps, method, demo, and experiments.

---

## 21. Final Submission Package

The final ZIP should contain:

```text
TalkWeaver_Final_Submission/
├── TalkWeaver_video.mp4
├── source_code/
│   └── TalkWeaver/
├── contribution.md
├── github_url.md
└── README_submission.md
```

Upload the ZIP to Google Drive and send only the sharing link to the professor.

---

## 22. Final Narrative for Report and Video

Use this narrative:

```text
We start from the project topic of speaker diarization, cross-speech, and LLM-ASR synergy. After reviewing recent work such as DiarizationLM, DM-ASR, TagSpeech, and RAG-based ASR correction, we identify four practical gaps: ASR/diarization misalignment, overlapping speech uncertainty, LLM hallucination during correction, and domain-term recognition errors.

TalkWeaver adapts recent research ideas into a lightweight engineering pipeline. Instead of training a new Speech-LLM, we combine faster-whisper, pyannote, temporal-anchor transcript formatting, overlap-aware uncertainty control, constrained LLM correction, and RAG-based domain term recovery.

Our goal is not to outperform state-of-the-art research models, but to build a reproducible, testable, and visually demonstrable system that explores how these ideas work in real noisy multi-speaker meeting scenarios.
```