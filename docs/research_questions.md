# Research Questions

## RQ1: Diarization-Structured Prompting

**Question:** Can diarization-structured prompting improve speaker-attributed
transcript readability and speaker consistency?

**Experiment mapping:** Compare group C, diarization plus alignment, with group
D, segment-level correction conditioned on speaker, timestamp, overlap, and
confidence. A stricter real study should also compare D with a whole-transcript
unstructured prompt.

**Measures:** simplified WDER or speaker-attribution error, unsupported
speaker-label changes, correction edit rate, WER, and a documented human
readability rubric.

**Interpretation rule:** improved fluency is not evidence of improved speaker
consistency. Speaker and lexical outcomes must be reported separately.

## RQ2: Overlap-Aware Uncertainty

**Question:** Can overlap-aware uncertainty control reduce hallucinated
corrections in overlapping speech regions?

**Experiment mapping:** Compare group F with an otherwise identical correction
run where overlap flags and conservative instructions are removed.

**Measures:** overlap-region WER, overlap-flag error, hallucinated correction
count, preserved uncertain spans, and manual error categories.

**Interpretation rule:** conservative correction may reduce unsupported edits
while leaving more lexical errors. Both outcomes must be reported.

## RQ3: RAG-Based Domain Term Recovery

**Question:** Can local glossary retrieval reduce ASR errors on technical
terms?

**Experiment mapping:** Compare group A, group D, and group E:

```text
Whisper only
-> structured correction without retrieved terms
-> structured correction plus RAG glossary
```

**Measures:** Term Error Rate, term precision, term recall, WER, and incorrect
glossary substitutions.

**Interpretation rule:** a term improvement is valid only when the reference
contains that term. Retrieval must not authorize unrelated content.

## RQ4: Audio Preprocessing

**Question:** Does local mono 16 kHz preprocessing improve noisy-meeting ASR?

**Experiment mapping:** Compare group A with group B on identical real audio.
Optional denoising is a separately controlled variant.

**Measures:** WER, retained audio duration, latency, failure rate, and analysis
by noise and overlap condition.

**Interpretation rule:** mock mode does not contain a real waveform, so groups
A and B are expected to match. RQ4 cannot be answered from mock results.

## Evidence Policy

The built-in ablation calculates demonstration metrics from a fixed mock
reference and labels every row `is_mock=true`. It validates the experiment
code and expected metric direction only. Research conclusions require:

- reference text;
- reference speaker and overlap labels;
- fixed model and decoding versions;
- documented hardware;
- a held-out test manifest;
- raw artifacts retained for audit.
