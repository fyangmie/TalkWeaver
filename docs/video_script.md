# TalkWeaver 10-12 Minute English Video Script

> Replace member names and insert real experiment footage before recording.
> Any chart labeled mock/demo must be described as a software demonstration,
> not a model-performance result.

## 0:00-1:10 - Opening Problem Demo

Hello. Our project is TalkWeaver: An Overlap-Aware Multi-Speaker ASR System
with Diarization-Structured LLM Correction. The subtitle is RAG-Enhanced
Domain Term Recovery for Noisy Meeting Speech.

Let us begin with a realistic meeting problem. One speaker says, "We use
pyannote for diarization." A second speaker interrupts and begins talking
before the first turn has completely ended. Later, the team says that it
should compare WER and DER.

A normal recognizer may output "piano note," "diary station," "where," and
"the ear." Even if a language model makes this paragraph look fluent, three
questions remain. Who said each phrase? Did the speakers overlap? Did the
model correct an actual ASR error, or did it invent content?

TalkWeaver is designed around those questions. It is not a simple Whisper web
interface. It is also not a generic RAG meeting chatbot. Its primary focus is
speaker diarization, cross-speech, temporal alignment, and the interaction
between ASR evidence and constrained language-model correction.

## 1:10-2:40 - Related Work

Our design is inspired by four research directions.

First, DiarizationLM, published at Interspeech 2024, represents ASR and
diarization outputs in a compact textual format for LLM post-processing. This
suggests that an LLM does not need to replace the speech pipeline. It can work
after existing ASR and diarization components if their structure is made
explicit.

Second, DM-ASR is a 2026 arXiv preprint about diarization-aware multi-speaker
ASR with large language models. It decomposes recognition into speaker- and
time-conditioned queries. We adapt the decomposition idea by correcting one
speaker-time segment at a time.

Third, TagSpeech is another 2026 preprint. It uses Temporal Anchor Grounding
to synchronize speaker identity, lexical content, and time. We adapt this idea
as a transparent temporal-anchor JSON format.

Fourth, retrieval-augmented ASR correction retrieves rare entities before
language-model correction. We use a smaller local TF-IDF glossary for project
terms. RAG is deliberately an auxiliary component.

The course also requires a provided paper named `xutong_paper.pdf`. That file
was not available in our repository when this documentation was prepared, so
we do not invent a summary. The final team must add and review it before
submission.

## 2:40-3:45 - Research Gaps and Questions

We identify four practical gaps.

The first gap is alignment. ASR produces words and timestamps, while
diarization produces speaker turns. Their boundaries may not agree.

The second gap is overlap. When two speakers are active, forcing one speaker
label can hide ambiguity.

The third gap is correction safety. An unconstrained LLM can change speaker
structure, remove uncertain words, or add unsupported facts.

The fourth gap is domain terminology. Technical expressions are rare in
general speech data and may sound like common phrases.

These gaps define our research questions. RQ1 tests diarization-structured
prompting. RQ2 tests overlap-aware uncertainty. RQ3 tests domain-term
retrieval. RQ4 tests local audio preprocessing.

## 3:45-5:35 - Method and System Workflow

The pipeline begins with audio preprocessing. A local recording is decoded,
mixed to mono, resampled to sixteen kilohertz, peak-normalized, and optionally
denoised. This produces a consistent ASR input.

The ASR stage uses faster-whisper when it is installed. It exports segment
timestamps and word timestamps. If the dependency is missing, the system does
not crash. It produces a deterministic output labeled as mock fallback.

The diarization stage uses pyannote when the package, model access, and Hugging
Face token are available. It produces speaker turns such as SPEAKER zero and
SPEAKER one. Mock mode contains two speakers and one deliberate overlap.

Alignment assigns each word using the midpoint of its start and end times. If
one turn contains the midpoint, that speaker is assigned. If multiple turns
contain it, the segment is labeled OVERLAP and all active speakers are
retained. If no turn contains it, the speaker is UNKNOWN.

The result is a temporal-anchor transcript. Every segment stores start time,
end time, speaker, active speakers, raw text, corrected text, overlap status,
confidence, retrieved terms, and word details.

Next, the local RAG module loads Markdown files from the knowledge base and
ranks small text chunks with TF-IDF. It may retrieve terms such as pyannote,
diarization, RAG, WER, and DER. These terms are candidates, not meeting facts.

Finally, correction runs one temporal segment at a time. The prompt includes
speaker and time structure. Timestamps and speaker labels are preserved.
Overlap segments receive conservative instructions and remain marked
uncertain. A validator rejects unsupported new words or rearrangements. When
no API key exists, deterministic glossary rules use the same structured
interface.

## 5:35-7:35 - Streamlit Demo Narration

Now we open the Streamlit application.

The Overview page displays the full project title, the research pipeline, and
the current execution mode. We can run the deterministic mock meeting without
a GPU or external credentials.

On the Audio Input page, we can upload WAV, MP3, M4A, FLAC, or OGG audio. The
file is saved locally, played in the browser, decoded for duration, sample
rate, channels, and file size, and displayed as a downsampled waveform.

On the Pipeline page, we select mock or real audio mode. We can review toggles
for preprocessing, ASR, diarization, overlap detection, RAG retrieval, LLM
correction, and summary. The integrated pipeline preserves dependencies and
shows the mode used by each component. This is important because a real audio
run may still use a labeled ASR or diarization fallback.

The Transcript Review page is the central research interface. At the top, the
speaker timeline shows each diarization turn. Overlap has a dedicated red
review lane. The raw ASR tab contains original segments and word timestamps.
The speaker-attributed tab contains every required temporal-anchor field. The
comparison tab places raw and corrected text side by side.

Notice the overlap segment. It has lower confidence, two active speakers, and
an uncertainty warning. TalkWeaver does not silently remove this warning after
correction.

The RAG page is secondary. It shows which terms were retrieved for each
segment, the local source files, an extractive summary, sourced action items,
and transcript-grounded search. It is not presented as an open-domain chatbot.

Finally, the Metrics page loads experiment CSV files and all saved charts. It
visibly warns when rows are mock or demo.

## 7:35-9:35 - Experiments and Metrics

Our ablation has six groups.

Group A is Whisper only. Group B adds preprocessing and tests RQ4. Group C
adds diarization and alignment. Group D adds structured segment-level
correction and relates to RQ1. Group E adds the RAG glossary and tests RQ3.
Group F adds explicit overlap-aware constraints and tests RQ2.

We implement six evaluation signals.

WER measures lexical errors. The script uses jiwer when installed and a
Levenshtein fallback otherwise.

Speaker error is a simplified project-level WDER approximation. Reference
temporal anchors are aligned to predicted anchors by overlap duration, then
active speaker sets are compared. We clearly state that this is not a full
industrial DER or WDER implementation.

Term Error Rate measures required domain terms missing from the hypothesis.
It also reports precision and recall.

Overlap error measures incorrect overlap flags. Hallucinated correction count
records corrections rejected by lexical grounding or manual review. Latency
measures eight pipeline stages independently.

The included CSV and five charts use the deterministic mock reference. For
example, the intended transcript exactly contains pyannote, diarization, RAG,
WER, and DER. Therefore, glossary-supported correction reaches zero demo WER
and zero demo Term Error Rate. This result is expected by construction. It
only proves that the scoring and chart pipeline works.

The mock speaker error also reaches zero after diarization and alignment
because the reference speaker anchors are the deterministic mock turns. Again,
this is not evidence of generalization.

For the real study, we must freeze a held-out manifest, annotate reference
text, speaker labels, and overlap, run all six groups on identical clips, and
report model versions, hardware, sample counts, failures, and variability.

## 9:35-10:40 - Error Analysis and Limitations

TalkWeaver exposes several remaining errors.

Word midpoints near speaker boundaries may be ambiguous. Diarization may miss
or falsely create overlap. Speaker labels can be permuted across files. TF-IDF
retrieval may miss phonetically similar terms or retrieve distractors. Text
validation reduces hallucination risk but cannot prove that a correction is
supported by the original audio.

The system is also a modular prototype, not a newly trained end-to-end speech
model. Real execution requires model downloads and credentials. The current
dataset is a deterministic demo, and the simplified speaker metric lacks
standard label mapping and collar handling.

These limitations are part of the research result. They define what the
project can and cannot claim.

## 10:40-11:35 - Future Work and Conclusion

The next priorities are a consented annotated meeting set, standard DER and
JER evaluation, a blinded overlap-aware prompt comparison, phonetic retrieval,
confidence calibration, multilingual meetings, and comparison of local and
API correction under one hallucination policy.

In conclusion, TalkWeaver demonstrates a research-driven way to connect ASR,
speaker diarization, overlap detection, and constrained language-model
correction. Its main design choices are a temporal-anchor transcript,
speaker-time conditioned correction, explicit overlap uncertainty, and narrow
domain-term retrieval.

We do not claim state-of-the-art performance. We demonstrate paper-informed
engineering, reproducible evaluation code, transparent mock behavior, and a
clear path to real experiments. Thank you.
