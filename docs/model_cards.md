# Model Cards

Record model provenance, versions, licenses, access requirements, intended use,
and limitations before enabling real inference.

## ASR

- Candidate: faster-whisper
- Version: to be pinned
- Model size: controlled by `ASR_MODEL_SIZE`
- Intended use: timestamped meeting transcription
- Known risks: accent, noise, overlap, and domain-term errors

## Diarization

- Candidate: pyannote.audio pipeline
- Version/model identifier: to be selected
- Access: Hugging Face token may be required
- Intended use: speaker-turn estimation
- Known risks: speaker swaps, missed overlap, and domain mismatch

## LLM Correction

- Provider/model: to be selected and logged
- Intended use: constrained segment correction
- Prohibited use: adding unsupported meeting facts
- Fallback: deterministic rule-based mock correction

## Retrieval

- Phase 1: deterministic glossary hints
- Planned baseline: TF-IDF over local Markdown
- Intended use: domain-term candidates only
