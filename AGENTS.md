# AGENTS.md — Codex Instructions for TalkWeaver

> This file is intended for Codex / AI coding agents.  
> Place this file in the repository root as `AGENTS.md`.  
> If a copy is also needed in `docs/AGENTS.md`, keep it synchronized.

---

## 0. Current Source of Truth

`PRD2.md` is the authoritative source for the final product direction and all
future implementation phases.

`PRD.md` documents the original v0 prototype. Its working pipeline, CLI,
tests, mock mode, and data contracts must be preserved, but its product
positioning and phase roadmap are historical. If `PRD.md`, this file, or older
documentation conflicts with `PRD2.md`, follow `PRD2.md`.

The final direction is:

```text
TalkWeaver: AI Meeting Detective for Chaotic Multi-Speaker Conversations
```

```text
An evidence-grounded conversation map for overlap, interruptions,
misheard terms, and speaker stances.
```

Future work must satisfy two goals:

1. build a fun, interactive, final-video-ready investigation experience;
2. keep every insight grounded in audio, temporal anchors, model artifacts,
   retrieval evidence, or human reference annotations.

Multilingual evaluation and the Level 1 whisper.cpp quantized-model mobile ASR
benchmark are mandatory final-project tracks. A full native iOS or Android app
is not mandatory.

All paper and runtime baseline integrations must follow
`docs/baseline_feasibility.md` and the reproduction policy in `PRD2.md`.
Record official source, version, license, dependencies, hardware, runtime, and
claim level. Do not run heavy training, large model downloads, or long
benchmarks without explicit user approval.

Formal evaluation follows the public-dataset-first strategy in `PRD2.md`.
Codex must not assume that self-recording is required. Data manifests and
validation tools must support both public-dataset entries and consent-safe
self-recorded entries. Never commit private or restricted audio; commit only
small synthetic examples or audio whose license and participant consent
explicitly permit redistribution.

Do not delete working v0 modules merely to fit the new interface. Extend or
adapt them incrementally.

---

## 1. Mission

You are building **TalkWeaver**, a research-inspired machine learning final project.

Final title:

```text
TalkWeaver: AI Meeting Detective for Chaotic Multi-Speaker Conversations
```

Subtitle:

```text
An evidence-grounded conversation map for overlap, interruptions,
misheard terms, and speaker stances.
```

The project must focus on:

- speaker diarization;
- overlapping speech / cross-speech;
- interruption and turn-taking evidence;
- LLM + ASR synergy;
- correction provenance and unsupported-change detection;
- speaker claims, stances, and action items with source anchors;
- recent paper-driven improvement;
- RAG-based domain term recovery as an auxiliary module;
- real manifests, human reference annotations, and controlled experiments;
- mandatory multilingual evaluation and Multilingual Demo;
- mandatory Level 1 whisper.cpp mobile ASR trade-off evaluation;
- an engaging investigation-oriented website;
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
- pet translation;
- entertainment without evidence;
- benchmark-only storytelling;
- full native mobile applications before the core research works;
- pure API usage.

The detective interface is required, but it must expose rigorous evidence.
Level 1 mobile work is required as a measured ASR trade-off experiment, not
as an early full-app rewrite. Apple WhisperKit/Core ML and Android/ONNX
Runtime remain hardware-dependent should-have or could-have extensions.

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

Inspect the local reference folder first. As of June 12, 2026, the course
anchor paper is available at `参考文献/xutong_paper.pdf`, together with the
five related papers listed in `PRD2.md`. Use the actual local sources and
never invent paper details. If files move or become unavailable, record that
status explicitly.

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

## 7. Legacy v0 Implementation Phases

The phases below document how the existing prototype was built. They are not
the active roadmap. Use the seven v1 phases in `PRD2.md` for all new planning
and implementation work.

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

The v1 UI must follow the AI Meeting Detective page plan in `PRD2.md`. The
existing Streamlit pages are a working v0 baseline and must remain usable
until their replacements are implemented incrementally.

The UI should feel fun, interactive, and impressive in the final video while
remaining a serious evidence-grounded research project. It must not look like
only a cold benchmark dashboard.

Required visual elements:

- audio player;
- conversation crime scene;
- waveform or placeholder waveform;
- speaker timeline;
- interruption map;
- raw transcript;
- speaker-attributed transcript;
- corrected transcript;
- overlap and interruption evidence;
- misheard-word rescue evidence;
- RAG candidate terms and scores;
- raw-versus-corrected audit;
- unsupported-correction warnings;
- evidence-backed speaker stance cards;
- multilingual comparison;
- mobile ASR trade-off charts;
- secondary metrics dashboard.

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

1. Open with a chaotic multi-speaker clip.
2. Use Conversation Crime Scene to ask what actually happened.
3. Investigate speakers, overlap, and interruptions through time.
4. Show a misheard term rescued by evidence-backed retrieval.
5. Audit an LLM edit in Hallucination Watchdog.
6. Compare speaker claims and stances with source anchors.
7. Map DiarizationLM, diarization-aware MS-ASR, DM-ASR, TagSpeech, and
   retrieval-based correction to concrete adaptations.
8. Present real ASR, ablation, overlap, term, multilingual, and mobile
   trade-off experiments.
9. Close with limitations and future work.

Do not present the project as merely a web app or merely a benchmark.

---

## 20. Final Reminder

The goal is not to claim state-of-the-art performance.

The goal is to build a strong final project that demonstrates:

- understanding of recent research;
- ability to identify practical gaps;
- ability to adapt paper ideas into engineering modules;
- ability to evaluate the system;
- ability to present a polished GitHub project and English demo video.
