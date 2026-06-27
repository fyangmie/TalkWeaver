# TalkWeaver

**AI Meeting Detective for Chaotic Multi-Speaker Conversations**

**Subtitle:** evidence-grounded conversation maps for overlap, interruptions,
misheard terms, and speaker stances.

TalkWeaver is a research-oriented meeting transcription project. It does not
try to claim state-of-the-art ASR. Instead, it asks a more practical question:
when ASR, diarization, overlap, retrieval, and LLM correction disagree, can the
system preserve enough evidence for a user to inspect what happened?

The repository now contains the complete project package on `main`: backend
pipeline code, Streamlit web app, public-subset experiment artifacts, controlled
safety studies, final paper sources, final PDF, presentation slides, and video
scripts.

## Core Idea

Conventional meeting pipelines often collapse uncertain speech evidence into a
single fluent transcript. TalkWeaver keeps the evidence visible:

```text
audio
  -> ASR hypothesis
  -> diarization turns
  -> word-to-speaker midpoint alignment
  -> overlap and event candidates
  -> temporal-anchor conversation map
  -> domain-term retrieval
  -> conservative correction and audit flags
  -> Streamlit review workspace
```

Each temporal anchor can retain:

- timestamp range;
- raw ASR text;
- speaker or active speaker set;
- overlap status;
- retrieved domain terms;
- corrected text when a correction is evidence-supported;
- review flags and unsupported-change audit metadata.

## Research Questions Used in the Final Paper

1. **RQ1:** How much harder is meeting speech than read speech?
2. **RQ2:** Can TalkWeaver construct auditable conversation maps from real
   meeting audio?
3. **RQ3:** Can conservative retrieval recover domain terms without degrading
   transcript quality?
4. **RQ4:** What safety boundaries limit automated correction decisions?

## What Is Included

| Area | Main paths |
| --- | --- |
| Backend pipeline | `backend/` |
| Streamlit app | `app.py`, `webapp/` |
| Experiment scripts | `experiments/`, `scripts/` |
| Result CSVs and artifacts | `experiments/results/`, `outputs/` |
| Dataset manifests and references | `data/manifests/`, `data/reference/` |
| Final paper package | `paper_v2/` |
| Final slides and scripts | `docs/TalkWeaver_Project_Presentation*.pptx`, `docs/presentation_scripts/` |
| Claim and handoff docs | `paper_v2/CLAIM_AUDIT.md`, `docs/PAPER_HANDOFF_FINAL.md`, `docs/final_claim_matrix.md` |

## Main Evidence Boundaries

TalkWeaver separates evidence types deliberately:

- **Real public audio subsets:** FLEURS, AMI, AISHELL-4, and Earnings-22.
- **Controlled text fixtures:** term rescue, overlap safety, correction gates.
- **Proxy/local runtime measurements:** whisper.cpp Level 1 local-machine rows.
- **Mock/demo mode:** deterministic pipeline sanity checks, not accuracy claims.

The public subsets are intentionally modest and should not be described as
full-corpus benchmarks. Controlled fixtures are mechanism and safety tests, not
real-audio generalization claims.

## Key Verified Results

Selected final-paper results committed in `experiments/results/`:

| Track | Verified result |
| --- | --- |
| Read speech vs meeting ASR | FLEURS zh-CN `base` CER `0.113`; AISHELL-4 formal meeting `base` CER `0.610`; AISHELL-4 60x20 `base` CER `0.537`; AMI held-out `base` WER `0.349`. |
| Larger ASR model trade-off | AISHELL-4 60x20 `small` CER `0.482` vs `base` `0.537`, with higher RTF; AMI held-out `small` WER `0.299`. |
| Diarization | AMI held-out pyannote DER `0.106`, JER `0.307`, overlap F1 `0.490`; AISHELL-4 multi-speaker subset DER `0.327`, JER `0.713`, overlap F1 `0.262`. |
| Automatic maps | AMI automatic workflow averages `4.75` anchors per clip; AISHELL-4 multi-speaker workflow averages `6.76` anchors per scored clip. |
| Earnings-22 RAG v3 | Base-model term recall improves from `0.833` to `1.000` on the 6-slice v3 blind subset, with WER unchanged at `0.212`; term F1 remains `0.833`. |
| Overlap safety | In authored high-overlap fixtures, overlap-aware rule and overlap-aware LLM variants pass `4/4`; the no-overlap-awareness rule passes `0/4` and introduces three forbidden changes. |
| EvidenceGate audit | Audit-aware grouped validation is leakage-prone; strict independent held-out performance is weak, so learned EvidenceGate is not deployed in the main workflow. |

For exact source paths and claim levels, see
[`paper_v2/CLAIM_AUDIT.md`](paper_v2/CLAIM_AUDIT.md).

## Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Optional real ASR, diarization, LLM, and plotting dependencies are documented
in [`docs/dependency_setup.md`](docs/dependency_setup.md). Install them only
when you need real model execution:

```bash
pip install -r requirements-optional.txt
```

System packages such as `ffmpeg` and `libsndfile` are listed in
[`packages.txt`](packages.txt).

## Environment Variables

Do not commit `.env` or credentials.

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

Real diarization uses `pyannote/speaker-diarization-community-1` and requires
accepted model terms plus `HF_TOKEN`. Optional LLM calls use an
OpenAI-compatible chat-completions endpoint and are never required for mock
mode.

## Quickstart

Run the deterministic mock pipeline:

```bash
python scripts/run_pipeline.py --mock
```

Run the Streamlit app:

```bash
streamlit run app.py
```

The app can run in dependency-free mock mode and is designed as a review
workspace for evidence maps, timelines, transcript comparison, term recovery,
and correction audits.

## Useful Commands

Mock evaluation and plots:

```bash
python experiments/run_ablation.py --mock
python experiments/plot_results.py
```

Formal ASR benchmark over the committed manifest paths:

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

AMI/AISHELL pyannote diarization benchmark:

```bash
python experiments/run_pyannote_diarization_benchmark.py \
  --manifest data/manifests/english_meeting_heldout_real.csv \
  --output experiments/results/pyannote_diarization_heldout_real.csv \
  --summary-output experiments/results/pyannote_diarization_heldout_summary_real.csv \
  --predictions-dir outputs/diarization/pyannote_heldout_real
```

Earnings-22 conservative RAG ablation:

```bash
python experiments/evaluate_earnings22_ablation.py \
  --asr-input experiments/results/asr_benchmark_earnings22_v3_blind_6x180.csv \
  --llm-input experiments/results/earnings22_v3_blind_rag_llm_v3.csv \
  --gate-version v3 \
  --output experiments/results/earnings22_v3_blind_ablation_v3.csv \
  --summary-output experiments/results/earnings22_v3_blind_ablation_v3_summary.csv
```

Controlled overlap safety:

```bash
python experiments/run_overlap_safety_experiment.py \
  --cases data/controlled_overlap/overlap_correction_cases.jsonl \
  --policy data/controlled_overlap/overlap_safety_policy.json \
  --output experiments/results/overlap_safety_controlled.csv
```

Local whisper.cpp Level 1 benchmark path:

```bash
python experiments/benchmark_whisper_cpp.py \
  --manifest data/manifests/formal_eval_real.csv \
  --output experiments/results/v1/whisper_cpp_mobile_level1.csv \
  --summary-output experiments/results/v1/whisper_cpp_mobile_level1_summary.csv
```

## Documentation Map

- Final paper: [`paper_v2/main.pdf`](paper_v2/main.pdf)
- Paper source: [`paper_v2/main.tex`](paper_v2/main.tex)
- Claim audit: [`paper_v2/CLAIM_AUDIT.md`](paper_v2/CLAIM_AUDIT.md)
- Final project report: [`PROJECT_REPORT.md`](PROJECT_REPORT.md)
- Presentation slides: [`docs/TalkWeaver_Project_Presentation_EN.pptx`](docs/TalkWeaver_Project_Presentation_EN.pptx)
- Video script: [`docs/video_script.md`](docs/video_script.md)
- Experiment handoff: [`docs/PAPER_HANDOFF_FINAL.md`](docs/PAPER_HANDOFF_FINAL.md)
- ASR benchmark protocol: [`docs/asr_benchmark.md`](docs/asr_benchmark.md)
- Pyannote benchmark protocol: [`docs/pyannote_diarization_benchmark.md`](docs/pyannote_diarization_benchmark.md)
- Earnings-22 v3 analysis: [`docs/earnings22_v3_blind_rag_v3_error_analysis.md`](docs/earnings22_v3_blind_rag_v3_error_analysis.md)
- EvidenceGate audit: [`docs/evidence_gate.md`](docs/evidence_gate.md)
- Frontend guide: [`docs/frontend.md`](docs/frontend.md)

## Repository Layout

```text
backend/       pipeline modules, schemas, ASR/diarization/RAG/correction logic
webapp/        Streamlit review workspace and UI components
experiments/   benchmark, ablation, safety, and plotting scripts
scripts/       pipeline runners, dataset preparation, validation utilities
data/          committed manifests, references, controlled fixtures
outputs/       committed example conversation maps and artifacts
docs/          protocols, handoff notes, reports, slides, scripts
paper_v2/      final paper source, figures, claim audit, PDF
tests/         regression and artifact validation tests
```

## Limitations

- Public-audio subsets are small and should not be reported as full-corpus
  benchmark results.
- Interruption evaluation reports candidate precision only; no exhaustive
  recall/F1 timeline annotation is committed.
- Controlled overlap and correction-gate fixtures are safety stress tests, not
  real-audio correction benchmarks.
- EvidenceGate is retained as a leakage/generalization audit and is not
  deployed as a trusted learned gate in the main workflow.
- whisper.cpp Level 1 results are local-machine measurements, not true
  phone-device deployment results.
- New private or restricted audio, API keys, model weights, and large caches
  should not be committed.

## License

See [`LICENSE`](LICENSE). Dataset use is governed by the original dataset
licenses and access terms recorded in the manifests and documentation.
