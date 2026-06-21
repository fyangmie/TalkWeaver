# TalkWeaver Project Report

**Title:** TalkWeaver: AI Meeting Detective for Chaotic Multi-Speaker
Conversations

**Subtitle:** An evidence-grounded conversation map for overlap,
interruptions, misheard terms, and speaker stances

> The repository contains deterministic mock demonstrations, small public-data
> real-audio results, Earnings-22 term-recovery diagnostics, and a
> mobile-style ASR proxy. Mock and proxy rows validate workflow and trade-off
> logic only; they are not full-corpus or true mobile performance claims.

## 1. Abstract

Noisy meetings combine lexical recognition, speaker attribution, temporal
grounding, and overlapping speech. A standard ASR transcript answers what was
said but may not reliably preserve who spoke, when a turn occurred, or whether
two voices were active simultaneously. TalkWeaver is a modular research
prototype that combines local audio preprocessing, faster-whisper ASR,
pyannote-compatible diarization, timestamp alignment, overlap detection, a
temporal-anchor transcript, glossary retrieval, and constrained segment-level
LLM correction. Its main contribution is an auditable interface between ASR,
speaker/time evidence, and correction: timestamps and speaker labels remain
fixed, overlap is represented explicitly, and every correction retains raw
text plus retrieval evidence. The repository now includes a 50-clip public
ASR subset, a 24-clip AMI held-out diarization run, a 60-clip AISHELL-4
Mandarin meeting benchmark subset, Earnings-22 finance-term RAG experiments,
controlled overlap-safety cases, latency and trade-off artifacts, charts, and
a Streamlit review dashboard. The results support the existence of the target
problems and the usefulness of constrained evidence, but they do not claim
state-of-the-art performance.

## 2. Introduction

Meeting speech is difficult because speakers interrupt each other, turn
boundaries are uncertain, background noise changes recognition quality, and
rare technical terms are easily replaced by common words. Independent ASR and
diarization systems also optimize different objectives. Their outputs can
disagree at boundaries even when each component appears plausible.

TalkWeaver studies a practical question: can paper-inspired structure make a
multi-speaker ASR pipeline more auditable and safer to correct? The system
does not train a new foundation model. It adapts recent ideas into an
engineering pipeline that can run with real models when available and remains
demonstrable without a GPU or external credentials.

The project is organized around four research questions:

1. Does diarization-structured prompting improve speaker consistency and
   transcript readability?
2. Does overlap-aware uncertainty reduce unsupported corrections?
3. Does local domain-term retrieval improve technical-term recovery?
4. Does mono 16 kHz normalization and optional denoising improve noisy ASR?

## 3. Related Work

### 3.1 Course Anchor Paper

The required local course thesis is available as
`参考文献/xutong_paper.pdf`. Its title page and abstract describe a
multi-speaker conversation-management system based on ASR, speaker separation,
LLM post-processing, cross-speech handling, and Streamlit integration. The
current reading note is abstract-level only; detailed experiment claims must
still be checked against the full 79-page PDF before final submission.

### 3.2 DiarizationLM

Wang et al., *DiarizationLM: Speaker Diarization Post-Processing with Large
Language Models*, appeared at Interspeech 2024. It represents ASR and
diarization output in a compact textual form and uses an LLM for diarization
post-processing and readability improvement. TalkWeaver adopts the structured
text interface, but adds explicit overlap flags, confidence, timestamps,
retrieved terms, and a correction audit trail.

Source: <https://arxiv.org/abs/2401.03506>

### 3.3 DM-ASR

Li et al., *DM-ASR: Diarization-aware Multi-speaker ASR with Large Language
Models*, is an arXiv preprint submitted in April 2026. It decomposes
multi-speaker recognition into speaker- and time-conditioned queries using
diarization as a structural prior. TalkWeaver adapts this idea as independent
correction of temporal speaker segments rather than implementing the paper's
trained speech-LLM architecture.

Source: <https://arxiv.org/abs/2604.22467>

### 3.4 TagSpeech

Huo et al., *TagSpeech: End-to-End Multi-Speaker ASR and Diarization with
Fine-Grained Temporal Grounding*, is an arXiv preprint submitted in January
2026. It uses temporal anchor grounding to synchronize semantic and speaker
streams and explicitly models who spoke what and when. TalkWeaver uses a
lightweight JSON temporal-anchor record for the same grounding objective; it
does not reproduce TagSpeech training or claim its reported accuracy.

Source: <https://arxiv.org/abs/2601.06896>

### 3.5 Retrieval-Augmented ASR Correction

Pusateri et al., *Retrieval Augmented Correction of Named Entity Speech
Recognition Errors*, is a 2024 arXiv preprint submitted to ICASSP 2025. It
retrieves relevant entities and supplies them to an adapted LLM for ASR
correction. TalkWeaver replaces the vector database with a small local TF-IDF
index and restricts retrieval to project terminology. This is an auxiliary
module, not the main research contribution.

Source: <https://arxiv.org/abs/2409.06062>

## 4. Research Gaps

The literature motivates four gaps addressed by the project:

- **Modular misalignment:** off-the-shelf ASR words and diarization turns need
  a transparent alignment rule.
- **Overlap ambiguity:** post-processing can erase uncertainty when more than
  one speaker is active.
- **Unsafe correction:** fluent LLM output can introduce unsupported facts or
  silently alter speaker structure.
- **Rare terminology:** domain terms may be acoustically confused with common
  phrases, while unrestricted retrieval can itself bias correction.

TalkWeaver responds with midpoint alignment, explicit overlap intervals,
confidence rules, immutable temporal anchors, lexical correction validation,
and narrow glossary retrieval.

## 5. Method

### 5.1 Audio Preprocessing

`backend/preprocessing.py` loads PCM WAV with the standard library and uses
soundfile or pydub/FFmpeg for additional formats. Audio is mixed to mono,
resampled to 16 kHz, peak-normalized, and optionally denoised when
`noisereduce` is installed. Mock mode records metadata without fabricating an
audio waveform.

### 5.2 ASR and Speaker Diarization

`backend/asr.py` uses faster-whisper with segment and word timestamps when the
dependency is available. Otherwise it exports a deterministic, labeled mock
transcript. `backend/diarization.py` uses pyannote only when the package,
accepted model access, and `HF_TOKEN` are available. Its fallback contains two
speakers and one deliberate overlap interval.

### 5.3 Alignment, Overlap, and Confidence

Words are assigned to turns by timestamp midpoint. One active speaker produces
a normal speaker label; multiple active speakers produce `OVERLAP` and retain
all speaker identities; no active turn produces `UNKNOWN`. Pairwise turn
intersection produces overlap intervals. Deterministic confidence rules assign
high confidence to single-speaker anchors and lower confidence to overlap or
unknown anchors.

### 5.4 Temporal-Anchor Transcript

Every aligned segment stores:

```json
{
  "start": 3.0,
  "end": 3.2,
  "speaker": "OVERLAP",
  "speakers": ["SPEAKER_00", "SPEAKER_01"],
  "raw_text": "The",
  "corrected_text": "The",
  "overlap": true,
  "confidence": 0.55,
  "retrieved_terms": []
}
```

This record is the contract between diarization, correction, export,
evaluation, and the Streamlit review interface.

### 5.5 Diarization-Structured Correction

`backend/prompting.py` formats one speaker-time segment at a time. The prompt
contains timestamps, speaker identities, raw text, overlap state, confidence,
and retrieved terms. `backend/llm_correction.py` preserves temporal anchors
and validates that corrected text does not add unsupported vocabulary or
content. Overlap segments are always marked uncertain. Without an API key, a
deterministic glossary rule system provides a reproducible fallback.

### 5.6 Auxiliary RAG Term Recovery

`backend/rag.py` loads local Markdown knowledge, creates TF-IDF vectors, and
retrieves a short candidate list for each segment. It supports corrections
such as `piano note -> pyannote`, `diary station -> diarization`, and
`rack -> RAG`. Retrieved terms are candidates, not permission to invent facts.

## 6. System Architecture

The pipeline is:

```text
audio -> preprocessing -> ASR -> diarization -> alignment
      -> overlap/confidence -> temporal anchors -> RAG terms
      -> constrained correction -> summary -> metrics -> review UI
```

The architecture figure is `assets/architecture.png`. Core modules live under
`backend/`, experiment scripts under `experiments/`, and review pages under
`webapp/`.

## 7. Experiments

### 7.1 Comparison Groups

| Group | Pipeline | Research role |
| --- | --- | --- |
| A | Whisper only | ASR baseline |
| B | + preprocessing | RQ4 |
| C | + diarization + alignment | speaker attribution baseline |
| D | + structured LLM correction | RQ1 |
| E | + RAG glossary | RQ3 |
| F | + overlap-aware correction | RQ2 |

### 7.2 Metrics

- **WER:** jiwer when installed, otherwise token-level Levenshtein distance.
- **Speaker error/WDER approximation:** duration-weighted active-speaker-set
  mismatch after temporal-overlap alignment. It is not full DER or WDER.
- **Term Error Rate:** required glossary terms absent from the hypothesis.
- **Overlap error:** temporal anchors with an incorrect overlap flag.
- **Hallucinated corrections:** corrected segments rejected by lexical
  grounding validation.
- **Latency:** measured elapsed time for eight pipeline stages.

### 7.3 Claim Levels

TalkWeaver separates four claim levels:

- **Mock/demo:** deterministic fixtures for pipeline validation only.
- **Diagnostic real:** small public subsets used to expose errors and tune
  method design.
- **Held-out real:** frozen public-data subsets used for reported results.
- **Proxy:** reproducible engineering approximations, such as the current
  mobile-style CPU int8 trade-off table.

No row is interpreted without its claim level, dataset, model, and hardware.
The final paper-safe boundaries are frozen in
`docs/final_claim_matrix.md`.

### 7.4 Mock Demonstration Protocol

The built-in reference contains the intended terms `pyannote`,
`diarization`, `RAG`, `WER`, and `DER`, plus four explicit speaker-time
anchors. The mock ASR intentionally contains acoustic-like confusions. Metrics
are calculated from those fixed inputs and written with `is_mock=true`.

### 7.5 Real Experiment Protocol

Real evaluation uses licensed public audio where possible. Current public
subsets are FLEURS for multilingual ASR, AMI for English meeting
speaker/overlap evidence, AISHELL-4 for Mandarin meeting benchmark subsets, and
Earnings-22 for finance-domain term recovery. Each result records model size,
decoding mode, hardware, and whether the reference text is used only for
scoring.

## 8. Results

### 8.1 Deterministic Demonstration Results

The following table validates metric direction and artifact plumbing only:

| Pipeline | WER | Speaker error | TER | Overlap error | Hallucinations |
| --- | ---: | ---: | ---: | ---: | ---: |
| Whisper only | 0.4444 | 1.0000 | 1.0000 | 0.2500 | 0 |
| + preprocessing | 0.4444 | 1.0000 | 1.0000 | 0.2500 | 0 |
| + diarization + alignment | 0.4444 | 0.0000 | 1.0000 | 0.0000 | 0 |
| + structured correction | 0.4444 | 0.0000 | 1.0000 | 0.0000 | 0 |
| + RAG glossary | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 |
| + overlap-aware correction | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 |

These values are expected from a deliberately constructed example: the mock
reference exactly contains the glossary-supported corrections and the mock
speaker labels exactly match the deterministic diarization turns. Therefore,
the zeros are not evidence of generalization or model quality.

Mock elapsed times are machine- and run-dependent and measure Python demo
execution, not real-time ASR or GPU inference. The current CSV is the source
of truth for those timings.

### 8.2 Real ASR Baseline

The expanded formal manifest contains 50 public clips: 10 English FLEURS, 10
French FLEURS, 10 Mandarin FLEURS, 8 AMI English meeting excerpts, and 12
AISHELL-4 Mandarin meeting excerpts.
`faster-whisper` was evaluated with `tiny` and `base` on CPU int8 with VAD
enabled. A separate 24-clip AMI held-out meeting subset was added for larger
meeting-only validation.

| Model | FLEURS EN WER | FLEURS FR WER | FLEURS Mandarin CER | AMI WER | AMI cleaned WER | AISHELL-4 CER | Mean warm RTF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tiny | 0.2104 | 0.3873 | 0.2226 | 0.4323 | 0.3775 | 0.6795 | 0.0506 |
| base | 0.1144 | 0.2271 | 0.1133 | 0.3984 | 0.3312 | 0.6100 | 0.0654 |

`base` is more accurate on the current subset, while `tiny` is faster. Meeting
speech remains much harder than read FLEURS speech, which supports the project
motivation.

On the 24-clip AMI held-out subset, the meeting-only ASR result is:

| Model | AMI held-out WER | AMI held-out cleaned WER | Mean warm RTF |
| --- | ---: | ---: | ---: |
| tiny | 0.3666 | 0.3144 | 0.0360 |
| base | 0.3493 | 0.2898 | 0.0717 |
| small | 0.2986 | 0.2336 | 0.1762 |

The `small` model improves AMI held-out WER but is roughly 2.5x slower than
`base` on this CPU int8 run, reinforcing the deployment trade-off.

For stronger Mandarin meeting evidence, the project also freezes a separate
60-clip AISHELL-4 benchmark subset: three 20-second clips from each of 20
test recordings, totaling 1200 seconds. The ASR result is:

| Model | Clips | AISHELL-4 CER | Mean warm RTF |
| --- | ---: | ---: | ---: |
| tiny | 60 | 0.6483 | 0.0636 |
| base | 60 | 0.5369 | 0.0711 |
| small | 60 | 0.4818 | 0.1367 |

The `small` model again improves accuracy, but Mandarin meeting speech remains
substantially harder than FLEURS Mandarin read speech. This is not the full
AISHELL-4 test set, but it is a much stronger Mandarin meeting benchmark than
the earlier 12-clip sanity subset.

### 8.3 Speaker-Time and Overlap Evidence

The AMI subset now has both the original 8-clip formal meeting subset and a
24-clip held-out subset balanced across `ES2002a`, `ES2002b`, `ES2002c`, and
`ES2002d`. A naive `no_diarization` baseline assigns one `UNKNOWN` speaker and
therefore has speaker-label error 1.000 and overlap F1 0.000. The
`reference_assisted` workflow uses reference turns as evidence; this validates
the evidence layer, event detection, and ConversationMap contract, but it is
not automatic diarization accuracy.

The automatic pyannote benchmark now runs successfully on the 24 held-out AMI
clips with standard `pyannote.metrics` DER/JER:

| Clips | mean DER | mean JER | DER skip-overlap | JER skip-overlap | overlap F1 | mean RTF |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 24 | 0.1060 | 0.3072 | 0.0813 | 0.2098 | 0.4902 | 0.6925 |

On the 60-clip AISHELL-4 Mandarin meeting subset, 29 clips contain multiple
reference speakers and are scored by the DER/JER script:

| Clips | mean DER | mean JER | DER skip-overlap | JER skip-overlap | overlap F1 | mean RTF |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 29 | 0.3265 | 0.7126 | 0.3262 | 0.6932 | 0.2619 | 0.5046 |

Using fixed `base` ASR predictions and automatic pyannote turns, the automatic
TalkWeaver evidence-map run produces 24 maps with mean 4.75 anchors, 4.25
speaker-labeled anchors, 1.79 overlap anchors, 2.04 events, and 2.29
needs-review flags per clip. This is the first fully automatic meeting
evidence-map result in the project.

The AISHELL-4 evidence map run uses the same automatic workflow on the 29
Mandarin clips with pyannote turns. It produces mean 6.76 anchors, 4.48
speaker-labeled anchors, 0.38 overlap anchors, 0.24 events, and 2.66
needs-review flags per clip. This closes the Mandarin meeting loop: the
AISHELL-4 subset now has ASR, diarization, and automatic TalkWeaver maps.

For interruption evidence, the rule-based floor-takeover detector generated
10 candidate windows. Human event-level review confirmed that all 10 windows
contained interruption behavior, giving candidate precision 1.000 over the
reviewed candidates. This is not recall or F1: the speaker-pair labels are
inherited from AMI timing, and non-candidate overlap windows have not been
exhaustively labeled.

### 8.4 RAG Domain-Term Recovery

Earnings-22 finance calls test whether glossary retrieval helps recover
domain terms without using reference text in prompts. On the final 12-file
blind subset, v2 evidence-gated RAG plus LLM had mixed results:

| Model | WER before | WER after | Term F1 before | Term F1 after |
| --- | ---: | ---: | ---: | ---: |
| tiny | 0.221805 | 0.221978 | 0.888889 | 0.930556 |
| base | 0.186844 | 0.187018 | 0.972222 | 0.930556 |

The `tiny` model benefits in term F1, but the `base` result gets worse because
one false-positive correction remains. This is useful evidence for the paper:
RAG can help weak ASR recover domain terms, but safe gating is part of the
method, not a minor engineering detail.

RAG v3 adds a stricter gate requiring predefined ASR error forms or numeric
unit error patterns plus explicit allowed-context evidence. A new six-file
Earnings-22 blind subset was prepared after freezing v3. The v3 result is
conservative. The formal ablation includes `asr_only`,
`glossary_candidates_only`, `llm_without_rag_conservative`,
`rag_evidence_gate_v3`, and `rag_llm_verifier_v3`:

| Model | WER before | WER after | Term recall before | Term recall after | Term F1 before | Term F1 after |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tiny | 0.2519 | 0.2519 | 0.8333 | 0.8333 | 0.8333 | 0.8333 |
| base | 0.2121 | 0.2121 | 0.8333 | 1.0000 | 0.8333 | 0.8333 |

The result does not improve WER, but it avoids the v2-style WER regression
and improves base term recall without changing transcript-level WER. This
supports a narrower claim: v3 is a safer evidence gate, not a general ASR
improvement method.

### 8.5 Overlap-Aware Correction Safety

Controlled overlap fixtures compare overlap-aware and overlap-agnostic
correction. The overlap-aware rule and LLM variants reach safety pass rate
1.000 across the controlled categories. The no-overlap-awareness rule variant
fails high-overlap cases and introduces forbidden changes. These results are
controlled text fixtures, not full meeting audio performance, but they support
RQ2's safety motivation.

### 8.6 Mobile ASR Trade-Off Proxy

`experiments/results/v1/mobile_asr.csv` contains 100 proxy rows derived from
the same faster-whisper CPU int8 benchmark. It records
`claim_level=mobile_style_proxy` and `true_mobile_device=false` for every row.
The proxy shows `tiny` with mean warm RTF 0.050645 and mixed error 0.396320,
while `base` has mean warm RTF 0.065400 and mixed error 0.301099. This
justifies the mobile trade-off track, but a true Level 1 mobile claim still
requires measured `whisper.cpp` or device-side results.

`experiments/benchmark_whisper_cpp.py` now provides the local `whisper.cpp`
Level 1 benchmark path. After installing `cmake`, building `whisper.cpp`, and
downloading git-ignored ggml tiny/base models, the benchmark produced 76
`status=ok` rows and no skipped rows over the earlier 38-clip formal subset
before AISHELL-4 was added. On AMI English meeting clips, `base` reached WER
0.351123, cleaned WER 0.280012, and RTF 0.056347; `tiny` reached WER
0.382383, cleaned WER 0.313909, and RTF 0.030548. This is a local-machine
deployment benchmark, not a phone-device measurement.

## 9. Error Analysis

The detailed audit is frozen in `docs/final_system_error_analysis.md`. The
main error patterns are:

| Area | Evidence | Failure Mode | Design Response |
| --- | --- | --- | --- |
| ASR on meetings | AMI `base` WER 0.3984, cleaned WER 0.3312 | fillers, low-energy words, insertions, and omissions | report standard and cleaned WER; keep raw ASR visible |
| Mandarin meetings | AISHELL-4 60-clip `base` CER 0.5369 and `small` CER 0.4818 vs FLEURS Mandarin CER 0.1133 | read-speech results do not predict meeting robustness | treat Mandarin meeting as a separate benchmark subset |
| Diarization/overlap | AMI held-out DER 0.1060, JER 0.3072, overlap F1 0.4902; AISHELL-4 DER 0.3265, JER 0.7126 on 29 multi-speaker clips | overlap detection remains much harder than speaker coverage | expose overlap as review evidence, not final truth |
| RAG correction | v2 improves `tiny` term F1 but hurts `base`; v3 keeps WER unchanged | retrieval can create false positives | use evidence gates, rejection paths, and audit trails |
| Interruption | 10/10 candidates verified as interruption events | recall/F1 unknown without full timeline labels | report candidate precision only |
| Mobile/runtime | 100 local CPU proxy rows; 76 local whisper.cpp rows | no phone-side latency, memory, or battery data | separate proxy, local-machine, and future phone claims |

The strongest interpretation is that TalkWeaver makes errors inspectable. It
does not eliminate ASR, diarization, retrieval, or mobile-deployment errors.

## 10. Limitations

TalkWeaver is a modular post-processing prototype, not an end-to-end trained
multi-speaker model. Real ASR and diarization require optional dependencies,
model downloads, and credentials. The simplified speaker metric is not an
industrial DER/WDER implementation. The formal real subset is still small:
FLEURS covers read speech, AMI covers English meeting recordings from one
scenario family, and AISHELL-4 Mandarin meeting results come from a fixed
60-clip subset rather than the full test partition. The automatic pyannote
track now has a real 24-clip AMI result and a 29-clip multi-speaker
AISHELL-4 subset result, and the `whisper.cpp` Level 1
local-machine benchmark now has real tiny/base rows. However, no true
phone-device benchmark has been run. TF-IDF retrieval has limited semantic and
phonetic recall. Correction validation operates on text rather than acoustic
evidence.

## 11. Future Work

Priority extensions are:

1. expand interruption labeling beyond the 10 human-confirmed event-level
   candidates if recall/F1 is required;
2. improve RAG v3 with phonetic retrieval and stricter false-positive gating;
3. expand Mandarin/code-switched meeting evaluation beyond the fixed 60-clip
   AISHELL-4 subset;
4. rerun `whisper.cpp` on a true phone or phone-like constrained environment;
5. compare local and API correction under the same hallucination policy.

## 12. Conclusion

TalkWeaver demonstrates how recent ideas in diarization-aware ASR can be
translated into a reproducible final-project system. DiarizationLM motivates
structured post-processing, DM-ASR motivates speaker-time conditioned
subtasks, TagSpeech motivates explicit temporal grounding, and
retrieval-augmented ASR correction motivates narrow domain-term candidates.
The resulting pipeline keeps speaker attribution, overlap uncertainty, raw
evidence, and corrections auditable. Current real results show that the target
failure modes exist and that constrained evidence can help, especially for
weak ASR domain-term recovery and overlap-safe correction. The project is now
closer to a credible final paper: it now has larger held-out meeting
evaluation, automatic diarization results, Mandarin meeting evidence, and
stricter RAG v3 blind validation. The main remaining gaps are exhaustive
interruption labels, deeper error analysis, and a true phone-side mobile ASR
benchmark.
