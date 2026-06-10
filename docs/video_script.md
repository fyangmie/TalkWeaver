# TalkWeaver Video Script

> Phase 1 narration draft for a 10-12 minute English presentation. Replace
> placeholders with verified citations, measured results, and final UI footage.

## 0:00-1:00 - Problem

Hello. Our project is TalkWeaver, an overlap-aware multi-speaker ASR system
with diarization-structured LLM correction. The subtitle is RAG-enhanced domain
term recovery for noisy meeting speech.

A normal speech recognizer tries to answer what was said. A real meeting system
must also answer who spoke, when they spoke, whether speakers overlapped, and
whether a correction is supported by the audio evidence. Noise, interruptions,
speaker changes, and rare technical terms make these questions difficult.

This project is not a simple Whisper interface and it is not a generic meeting
chatbot. Our main focus is speaker diarization, cross-speech, and the synergy
between ASR and constrained language-model correction.

## 1:00-2:30 - Related Work

We begin from the course paper and recent work on diarization-aware speech
processing. At final recording time, this section will show the verified
citations and summarize each original source.

DiarizationLM motivates a compact textual representation that combines ASR and
diarization information for language-model post-processing. Diarization-aware
multi-speaker ASR and DM-ASR motivate conditioning on speaker identity and
time rather than treating a whole meeting as one unstructured paragraph.
Temporal-anchor work such as TagSpeech motivates explicit grounding of who
spoke what and when. Retrieval-augmented ASR correction motivates retrieving
rare terms before correction.

We adapt these ideas into a lightweight engineering pipeline. We do not claim
to reproduce the complete research models.

## 2:30-3:30 - Research Gaps and Questions

We identify four practical gaps. First, independent ASR and diarization outputs
can be misaligned. Second, overlapping speech creates ambiguous evidence.
Third, an unconstrained LLM can make unsupported corrections. Fourth, ASR often
confuses rare technical terms with common words.

These gaps lead to four research questions. We test structured prompting,
overlap-aware uncertainty, glossary retrieval for domain terms, and local audio
preprocessing.

## 3:30-5:15 - Method

The pipeline starts with local audio preprocessing. Audio will be converted to
mono at 16 kilohertz, normalized, and optionally denoised. The ASR stage
produces segment and word timestamps. The diarization stage produces speaker
turns.

Next, the alignment stage assigns words or segments to speakers using temporal
evidence. The overlap stage detects simultaneous speaker turns and lowers the
confidence of affected transcript segments.

Every output is stored as a temporal-anchor record. The record contains start
and end times, speaker, raw text, corrected text, overlap status, confidence,
and retrieved domain terms. This creates an audit trail.

The retrieval module has a narrow role. It finds terms such as pyannote.audio,
speaker diarization, WER, DER, and RAG from a local project knowledge base.
These terms become correction candidates. They are not evidence for adding new
meeting content.

Finally, correction is performed independently for each speaker-time segment.
The prompt requires preservation of timestamps and speaker labels, and it asks
for conservative behavior in overlap regions.

## 5:15-7:00 - Demo

On the Upload page, we select a meeting audio file and inspect the audio
preview. During development, mock mode lets the complete interface run without
a GPU, a Hugging Face token, or an LLM API key.

On the Pipeline page, we configure preprocessing, ASR, diarization, overlap
detection, domain-term retrieval, and correction. The status view shows each
stage and its saved artifacts.

The Transcript Review page displays the speaker timeline and the raw and
corrected text side by side. Overlap segments are visibly marked for review.
For example, the mock ASR phrase "piano note" is corrected to "pyannote" only
because the glossary supplies a relevant candidate. The original text remains
visible.

The Domain-Term Recovery page shows retrieved terms and secondary meeting
understanding. This page is intentionally supporting the ASR workflow rather
than becoming the main project.

## 7:00-9:00 - Experiments

Our ablation study contains six groups. Group A is Whisper only. Group B adds
preprocessing. Group C adds diarization and alignment. Group D adds structured
LLM correction. Group E adds the RAG glossary. Group F compares overlap-aware
correction with correction that does not receive an overlap flag.

We measure WER, speaker-attribution error or WDER, Term Error Rate, overlap
errors, hallucinated corrections, and latency. We also preserve intermediate
outputs for manual error analysis.

At final recording time, this section will show measured tables and charts.
Mock rows will never be presented as experimental evidence. We will explain
the dataset, reference annotation, model versions, hardware, and any failed or
excluded samples.

## 9:00-10:15 - Limitations

TalkWeaver is a modular final-project system rather than a newly trained
speech-language foundation model. Its quality depends on ASR timestamps,
diarization accuracy, glossary relevance, and the correction model. Overlap
regions may remain difficult even when uncertainty is explicit.

The evaluation dataset may also be small. We will therefore avoid broad
claims, report controlled comparisons, and include error examples.

## 10:15-11:00 - Conclusion

TalkWeaver explores how recent research ideas can be adapted into a
reproducible multi-speaker meeting pipeline. Its central design choices are a
temporal-anchor transcript, speaker-time conditioned correction,
overlap-aware uncertainty, and narrow domain-term retrieval.

Our goal is not to claim state-of-the-art performance. Our goal is to
demonstrate research understanding, careful engineering, transparent
evaluation, and a working system for noisy multi-speaker meetings.
