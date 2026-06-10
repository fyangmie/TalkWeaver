# Research Questions

## RQ1: Diarization-Structured Prompting

**Question:** Can diarization-structured prompting improve speaker-attributed
transcript readability and speaker consistency?

**Comparison:** plain transcript correction versus correction conditioned on
speaker, timestamp, confidence, overlap state, and retrieved terms.

**Measures:** speaker-attribution error or WDER proxy, unsupported speaker-label
changes, correction edit rate, and a documented human readability rubric.

**Risk:** a language model may make text more fluent without improving speaker
consistency. Readability and attribution must therefore be measured separately.

## RQ2: Overlap-Aware Uncertainty

**Question:** Can overlap-aware uncertainty control reduce hallucinated
corrections in overlapping speech regions?

**Comparison:** identical correction pipeline with and without overlap flags
and conservative instructions.

**Measures:** hallucinated correction count, overlap-region WER, preserved
uncertain spans, and manual error categories.

**Risk:** conservative correction may reduce hallucinations while leaving more
ASR errors. Both effects must be reported.

## RQ3: RAG-Based Domain Term Recovery

**Question:** Can RAG-based domain glossary retrieval reduce ASR errors on
technical terms?

**Comparison:** ASR only, structured correction, and structured correction plus
glossary retrieval.

**Measures:** Term Error Rate, term precision/recall, WER, and incorrect
glossary substitutions.

**Risk:** irrelevant candidates may bias correction. Retrieval quality must be
evaluated independently from correction quality.

## RQ4: Audio Preprocessing

**Question:** Does local audio preprocessing improve ASR robustness under noisy
meeting conditions?

**Comparison:** raw audio versus mono 16 kHz normalized audio, with optional
denoising and VAD as separately controlled variants.

**Measures:** WER, latency, audio duration retained, and failure analysis by
noise and overlap condition.

**Risk:** denoising or silence trimming may remove quiet speech and harm
diarization.

## Evidence Policy

Mock outputs validate interfaces only. Research conclusions require reference
transcripts, reference speakers, documented audio conditions, repeatable
commands, and results labeled with model and hardware versions.
