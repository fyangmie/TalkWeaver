# TalkWeaver Web MVP Deployment

This document describes how to run and deploy the TalkWeaver web MVP. The MVP is a Streamlit app packaged through a Docker Space. It shows the project story, multilingual UI, uploaded-audio playback, and precomputed EvidenceMap artifacts. It does not run heavy ASR, diarization, or LLM models online.

## Local Preview

From the repository root:

```bash
streamlit run app.py --server.address=127.0.0.1 --server.port=8501
```

Open:

```text
http://127.0.0.1:8501
```

The page should show:

- TalkWeaver project hero section;
- language switch: Chinese, English, French;
- uploaded-audio playback;
- precomputed EvidenceMap selector;
- timeline, anchors, correction audit, and paper evidence cards.
- default playable example:
  - `outputs/conversation_maps/earnings22_multi_speaker_term_rescue/earnings22_4481221_0000_180s_conversation_map.json`
  - `data/raw/public/earnings22/earnings22_4481221_0000_180s.wav`

## Docker Space Deployment

Use Hugging Face Spaces with **Docker SDK**. Do not choose the built-in Streamlit SDK for this project.

1. Log in to Hugging Face.
2. Create a new Space.
3. Select **Docker** as the SDK.
4. Push this repository to the Space repository.
5. Keep the following files/directories:
   - `Dockerfile`
   - `app.py`
   - `requirements.txt`
   - `packages.txt`
   - `webapp/`
   - `backend/`
   - `assets/`
   - selected `outputs/conversation_maps/` JSON artifacts
   - selected public demo audio used by those JSON artifacts
   - selected `experiments/results/` CSV files used by the demo
6. Do not upload:
   - `.env`
   - API keys
   - Hugging Face tokens
   - private audio
   - restricted raw datasets
   - large model weights
7. The container command is already defined in `Dockerfile`:

```bash
streamlit run app.py --server.address=0.0.0.0 --server.port=7860
```

Docker Spaces route traffic to port `7860`.

## Demo Boundary

The web MVP is intentionally stable and lightweight:

- Uploaded audio is used for playback and interaction.
- EvidenceMap display comes from existing local JSON artifacts.
- No heavyweight ASR, pyannote diarization, or LLM API call runs inside the online demo.
- The page should be described as an evidence-grounded research demo, not a production transcription service.

## If the Space Is Too Large

Keep only the artifacts needed for the demo:

- `outputs/conversation_maps/earnings22_multi_speaker_term_rescue/earnings22_4481221_0000_180s_conversation_map.json`;
- `data/raw/public/earnings22/earnings22_4481221_0000_180s.wav`;
- one or two `outputs/conversation_maps/ablation_real/full_talkweaver/*_conversation_map.json` files if you want extra meeting-overlap examples;
- the chart PNGs under `assets/result_charts/`;
- result summary CSVs used by the paper evidence cards.

Do not commit restricted raw audio unless the dataset license and redistribution terms explicitly allow it.
