# Literature Review

> Phase 1 research map. Verify bibliographic metadata, links, datasets, and
> technical claims against the original papers before final submission.

## Scope

TalkWeaver studies recent work at the intersection of:

- speaker diarization and speaker attribution;
- overlapping or cross-speech;
- ASR and LLM post-processing;
- temporal grounding of multi-speaker transcripts;
- retrieval-assisted recovery of rare domain terms.

Meeting summarization and QA are secondary outputs. They are not the central
research contribution.

## Anchor Course Paper

The required `project/xutong_paper.pdf` was not available during repository
initialization. No claims about this paper are included. See
`paper_reading_notes/00_xutong_paper.md`.

## Research Map

| Work area | Problem addressed | TalkWeaver adaptation | Verification status |
| --- | --- | --- | --- |
| DiarizationLM | LLM post-processing of ASR and diarization output | Compact speaker-time prompt with overlap, confidence, and terms | Original source required |
| Diarization-aware multi-speaker ASR | Jointly reason about speech and speaker structure | Reproducible modular ASR, diarization, and correction pipeline | Original source required |
| DM-ASR | Speaker- and time-conditioned multi-speaker queries | Correct each temporal speaker segment independently | Original source required |
| TagSpeech / temporal anchors | Ground who spoke what and when | Temporal-anchor JSON transcript | Original source required |
| Retrieval-augmented ASR correction | Recover rare entities and terminology | Local domain glossary candidates for constrained correction | Original source required |

## Practical Gaps

### ASR and Diarization Misalignment

Independent systems may disagree at speaker boundaries. TalkWeaver will test a
timestamp-based alignment baseline and record unknown or ambiguous assignments.

### Overlap Uncertainty

A fluent correction may be unsupported when two speakers overlap. TalkWeaver
will compare correction with and without explicit overlap constraints and count
unsupported changes.

### Domain-Term Recognition

Rare terms such as `pyannote.audio`, `WDER`, and `RAG` can be replaced by
common words. TalkWeaver restricts retrieval to plausible correction
candidates from a local project glossary.

### Reproducibility

Frontier multi-speaker systems may require substantial training and compute.
TalkWeaver adapts their structural ideas into a modular pipeline with mock
fallbacks and explicit ablations.

## Synthesis

The proposed contribution is not a new foundation model. It is a controlled
study of whether structured speaker-time context, overlap-aware uncertainty,
and narrow glossary retrieval improve an auditable meeting-ASR workflow.

## Citation Checklist

- [ ] Add complete citations and stable links.
- [ ] Read and summarize the provided course paper.
- [ ] Distinguish peer-reviewed papers from arXiv preprints.
- [ ] Record datasets, metrics, and model assumptions from each source.
- [ ] Connect every implemented component to a verified source or baseline.
- [ ] Avoid claiming that TalkWeaver reproduces a paper's full model.
