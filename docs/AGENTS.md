# AGENTS.md — Codex Instructions for TalkWeaver

> This file is intended for Codex / AI coding agents.  
> Place this file in the repository root as `AGENTS.md`.  
> If a copy is also needed in `docs/AGENTS.md`, keep it synchronized.

---

## 1. Mission

You are building **TalkWeaver**, a research-inspired machine learning final project.

Final title:

```text
TalkWeaver:
An Overlap-Aware Multi-Speaker ASR System
with Diarization-Structured LLM Correction
```

Subtitle:

```text
RAG-Enhanced Domain Term Recovery for Noisy Meeting Speech
```

The project must focus on:

- speaker diarization;
- overlapping speech / cross-speech;
- LLM + ASR synergy;
- recent paper-driven improvement;
- RAG-based domain term recovery as an auxiliary module;
- experiments and evaluation;
- GitHub-quality engineering.

Do **not** implement this as a simple Whisper demo or a generic RAG meeting chatbot.

---

## 2. Non-Negotiable Project Focus

The main topic is:

```text
Topic 1:
Speaker diarization, cross-speech, and LLM + ASR synergy.
```

Supporting module:

```text
Topic 3:
Local audio preprocessing for better ASR performance.
```

RAG is only an auxiliary enhancement:

```text
RAG is used mainly for ASR domain-term correction and secondary meeting understanding.
```

Do not let the project drift into:

- generic meeting summarization;
- generic chatbot;
- mobile deployment;
- pet translation;
- pure UI building;
- pure API usage.

---

## 3. Research-Driven Design Requirement

The project must be built as:

```text
recent paper review
→ research gap identification
→ paper-inspired component adaptation
→ engineering pipeline
→ experiments
→ report and video
```

### Required literature files

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

### Important

If `project/xutong_paper.pdf` exists, read and summarize it. If it does not exist, create a placeholder note saying the file was not available and must be added by the student. Never invent paper details.

---

## 4. Core Research Questions

Implement and document the project around these research questions:

```text
RQ1:
Can diarization-structured prompting improve speaker-attributed transcript readability and speaker consistency?

RQ2:
Can overlap-aware uncertainty control reduce hallucinated corrections in overlapping speech regions?

RQ3:
Can RAG-based domain glossary retrieval reduce ASR errors on technical terms?

RQ4:
Does local audio preprocessing improve ASR robustness under noisy meeting conditions?
```

---

## 5. Paper-Inspired Components

Implement four paper-inspired components.

### 5.1 Diarization-Structured Transcript Formatter

Inspired by DiarizationLM.

Convert ASR and diarization outputs into a compact structured prompt format:

```text
[00:00.00-00:03.20] SPEAKER_00 | overlap=false | confidence=0.91
Raw: We use piano note for diary station.
Retrieved terms: pyannote, speaker diarization
```

Files:

```text
backend/prompting.py
backend/alignment.py
backend/llm_correction.py
```

### 5.2 Speaker-Time Conditioned Correction

Inspired by DM-ASR.

Correct each speaker-time segment independently rather than sending the whole transcript as one block.

Files:

```text
backend/prompting.py
backend/llm_correction.py
```

### 5.3 Temporal-Anchor JSON Transcript Format

Inspired by temporal anchor ideas in recent multi-speaker ASR work.

Use this structure:

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

Files:

```text
backend/alignment.py
backend/overlap.py
backend/export.py
webapp/components/speaker_timeline.py
```

### 5.4 RAG-Based Domain Term Recovery

Inspired by retrieval-augmented ASR correction.

Use local markdown knowledge base and domain terms to retrieve correction candidates.

Files:

```text
backend/rag.py
docs/knowledge_base/domain_terms.md
experiments/evaluate_terms.py
```

---

## 6. Required Repository Structure

Create and preserve this structure:

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

Add `.gitkeep` files in otherwise-empty folders.

---

## 7. Implementation Phases

Work incrementally. Do not try to finish everything in one giant patch.

### Phase 1 — Research Documentation and Skeleton

Create:

- full project structure;
- README template;
- PROJECT_REPORT template;
- BLOG_ARTICLE template;
- literature review template;
- research questions;
- paper reading notes;
- knowledge base starter docs;
- `.env.example`;
- `.gitignore`;
- Dockerfile and docker-compose template.

Do not implement heavy model logic yet.

### Phase 2 — Audio Preprocessing and ASR Baseline

Implement:

```text
backend/preprocessing.py
backend/asr.py
scripts/run_stage.py
```

Requirements:

- load audio;
- convert to mono 16kHz;
- normalize volume;
- optional denoising;
- faster-whisper ASR if available;
- mock ASR fallback;
- JSON and Markdown export.

### Phase 3 — Diarization, Alignment, and Overlap

Implement:

```text
backend/diarization.py
backend/alignment.py
backend/overlap.py
backend/confidence.py
```

Requirements:

- pyannote support if `HF_TOKEN` is available;
- mock diarization fallback;
- word-speaker alignment by timestamp midpoint;
- overlap region detection;
- temporal-anchor JSON output.

### Phase 4 — Prompting and LLM Correction

Implement:

```text
backend/prompting.py
backend/llm_correction.py
```

Requirements:

- diarization-structured prompt;
- speaker-time conditioned correction;
- overlap-aware uncertainty constraints;
- API LLM support if key exists;
- deterministic mock/rule-based correction fallback;
- no hallucinated new facts.

### Phase 5 — RAG Domain Term Recovery

Implement:

```text
backend/rag.py
docs/knowledge_base/domain_terms.md
```

Requirements:

- load markdown knowledge base;
- TF-IDF retrieval fallback;
- return candidate domain terms;
- integrate with LLM correction;
- evaluate term recovery.

### Phase 6 — Streamlit Multi-Page App

Implement:

```text
webapp/streamlit_app.py
webapp/pages/1_Upload.py
webapp/pages/2_Pipeline.py
webapp/pages/3_Review_Transcript.py
webapp/pages/4_RAG_Assistant.py
webapp/pages/5_Metrics.py
```

Requirements:

- audio upload/playback;
- pipeline options;
- transcript review;
- speaker timeline;
- overlap warnings;
- corrected transcript;
- RAG summary/action items;
- metrics dashboard.

### Phase 7 — Experiments and Evaluation

Implement:

```text
experiments/evaluate_wer.py
experiments/evaluate_wder.py
experiments/evaluate_terms.py
experiments/evaluate_latency.py
experiments/run_ablation.py
experiments/plot_results.py
```

Requirements:

- run demo/mock ablation;
- support real results when references exist;
- save CSV files;
- generate charts;
- never fabricate metrics as real.

### Phase 8 — Final Documentation

Update:

- README.md;
- PROJECT_REPORT.md;
- BLOG_ARTICLE.md;
- docs/video_script.md;
- docs/contribution.md;
- docs/github_url.md.

---

## 8. Coding Standards

### 8.1 General

Use Python 3.10+.

Prefer:

- modular functions;
- type hints where helpful;
- dataclasses or typed dictionaries for structured outputs;
- clear error messages;
- readable docstrings;
- deterministic mock mode.

Avoid:

- huge monolithic scripts;
- hidden API keys;
- unhandled missing dependency crashes;
- hardcoded absolute paths;
- fabricated metrics;
- excessive complexity.

### 8.2 Import Strategy

Heavy dependencies such as `faster_whisper`, `pyannote.audio`, `torch`, or LLM SDKs should be imported lazily inside functions where possible.

If a dependency is missing, provide a helpful message and fallback to mock mode if allowed.

### 8.3 Configuration

Use:

```text
backend/config.py
.env
.env.example
```

Environment variables:

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

Never commit real `.env`.

---

## 9. Mock Mode Policy

Mock mode is required.

Mock mode must:

- allow Streamlit to run without GPU;
- allow CLI pipeline to run without external tokens;
- produce deterministic sample transcripts;
- produce sample diarization and overlap regions;
- produce sample corrected output;
- label outputs as mock/demo.

Do not present mock metrics as real experimental results.

---

## 10. Pipeline Output Formats

### 10.1 Raw ASR JSON

```json
[
  {
    "start": 0.0,
    "end": 3.2,
    "text": "Today we are testing speaker diarization.",
    "words": [
      {"word": "Today", "start": 0.0, "end": 0.4},
      {"word": "we", "start": 0.5, "end": 0.6}
    ]
  }
]
```

### 10.2 Diarization JSON

```json
[
  {
    "start": 0.0,
    "end": 4.1,
    "speaker": "SPEAKER_00"
  },
  {
    "start": 4.1,
    "end": 8.6,
    "speaker": "SPEAKER_01"
  }
]
```

### 10.3 Temporal-Anchor Transcript JSON

```json
[
  {
    "start": 0.0,
    "end": 3.2,
    "speaker": "SPEAKER_00",
    "raw_text": "We use piano note for diary station.",
    "corrected_text": "We use pyannote for speaker diarization.",
    "overlap": false,
    "confidence": 0.91,
    "retrieved_terms": ["pyannote.audio", "speaker diarization"]
  }
]
```

---

## 11. LLM Correction Rules

The LLM correction module must follow these rules:

1. preserve timestamps;
2. preserve speaker labels unless there is strong evidence;
3. correct punctuation and obvious ASR mistakes;
4. use retrieved domain terms;
5. mark overlap segments as uncertain if needed;
6. do not invent content not supported by the transcript;
7. do not delete uncertain content silently;
8. keep an audit trail of raw vs corrected text.

---

## 12. RAG Rules

RAG must focus on domain-term recovery, not general chatbot behavior.

Knowledge base files:

```text
docs/knowledge_base/domain_terms.md
docs/knowledge_base/asr_background.md
docs/knowledge_base/diarization_background.md
docs/knowledge_base/rag_background.md
docs/knowledge_base/project_requirement.md
```

Domain term examples:

```text
pyannote.audio
speaker diarization
overlapping speech
cross-speech
ASR
WER
DER
WDER
RAG
faster-whisper
VAD
LLM correction
temporal anchor
speaker attribution
```

Common correction examples:

```text
piano note -> pyannote
diary station -> diarization
where -> WER
the ear -> DER
rack -> RAG
```

---

## 13. Experiments and Metrics

### Required comparison groups

```text
A. Whisper only
B. Preprocessing + Whisper
C. Whisper + diarization + alignment
D. Structured LLM correction
E. Structured LLM correction + RAG glossary
F. Overlap-aware correction vs no-overlap-flag correction
```

### Required metrics

- WER;
- WDER or speaker attribution error;
- Term Error Rate;
- overlap error analysis;
- hallucinated correction count;
- latency.

### Important

If ground truth is missing, implement:

- mock/demo evaluation;
- clear placeholder comments;
- instructions for adding reference transcripts.

Do not invent final experimental claims.

---

## 14. Streamlit UI Expectations

The UI should look like a serious final project.

Required visual elements:

- audio player;
- pipeline status;
- waveform or placeholder waveform;
- speaker timeline;
- raw transcript;
- speaker-attributed transcript;
- corrected transcript;
- overlap warning badges;
- RAG retrieved terms;
- meeting summary;
- action items;
- metrics charts.

The UI should support mock mode for demo and development.

---

## 15. Documentation Expectations

### README.md

Must include:

- project overview;
- research motivation;
- related work summary;
- project architecture;
- installation;
- environment variables;
- quickstart;
- Streamlit usage;
- CLI usage;
- experiment usage;
- mock mode explanation;
- screenshots/charts placeholders;
- limitations.

### PROJECT_REPORT.md

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

### BLOG_ARTICLE.md

Must be readable and presentation-friendly.

### docs/video_script.md

Must be a 10+ minute English presentation script.

---

## 16. Git and Commit Guidance

The professor may inspect commit history.

Use meaningful commits. Avoid one giant final commit.

Suggested commit style:

```bash
git commit -m "docs: add literature review and research questions"
git commit -m "chore: initialize TalkWeaver project structure"
git commit -m "feat: implement preprocessing and ASR baseline"
git commit -m "feat: add diarization alignment and overlap detection"
git commit -m "feat: add diarization-structured LLM correction"
git commit -m "feat: add RAG-based domain term recovery"
git commit -m "feat: build multi-page Streamlit dashboard"
git commit -m "exp: add WER and term error evaluation"
git commit -m "exp: add ablation results and charts"
git commit -m "docs: add project report and video script"
```

Never commit:

- `.env`;
- API keys;
- Hugging Face tokens;
- large model files;
- large raw datasets;
- long videos.

---

## 17. Testing and Sanity Checks

At minimum, these commands should work:

```bash
python scripts/run_pipeline.py --mock
python scripts/run_stage.py --stage asr --mock
python experiments/run_ablation.py --mock
python experiments/plot_results.py
streamlit run webapp/streamlit_app.py
```

If real dependencies are installed, also support:

```bash
python scripts/run_pipeline.py --audio data/demo/demo_meeting.wav
```

The project should not crash merely because pyannote or an LLM API key is missing.

---

## 18. Deliverable Checklist

Before final submission, ensure the repository contains:

- [ ] README.md
- [ ] PROJECT_REPORT.md
- [ ] BLOG_ARTICLE.md
- [ ] requirements.txt
- [ ] .env.example
- [ ] Dockerfile
- [ ] docker-compose.yml
- [ ] Streamlit multi-page app
- [ ] backend pipeline modules
- [ ] CLI scripts
- [ ] experiment scripts
- [ ] result CSV files
- [ ] result charts
- [ ] literature review
- [ ] paper reading notes
- [ ] research questions
- [ ] video script
- [ ] contribution.md
- [ ] github_url.md
- [ ] mock mode
- [ ] real or demo pipeline output
- [ ] meaningful Git commit history

---

## 19. Final Video Narrative

The video should follow this story:

1. Real-world problem: noisy multi-speaker meetings are hard.
2. Related work: DiarizationLM, DM-ASR, TagSpeech, RAG-ASR correction.
3. Research gaps: overlap, speaker-time alignment, hallucination, domain terms.
4. Our method: structured transcript, overlap uncertainty, RAG domain recovery.
5. Demo: upload audio, run pipeline, show speaker timeline and corrections.
6. Experiments: WER, WDER, Term Error, latency, ablation.
7. Limitations and future work.

Do not present the project as merely a web app.

---

## 20. Final Reminder

The goal is not to claim state-of-the-art performance.

The goal is to build a strong final project that demonstrates:

- understanding of recent research;
- ability to identify practical gaps;
- ability to adapt paper ideas into engineering modules;
- ability to evaluate the system;
- ability to present a polished GitHub project and English demo video.