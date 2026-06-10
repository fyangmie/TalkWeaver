# ASR Background

Automatic speech recognition converts speech audio into text. TalkWeaver
requires segment timestamps and, when available, word timestamps so recognized
content can be aligned with speaker turns.

The planned baseline is faster-whisper. Evaluation must record model size,
decoding settings, language, hardware, and preprocessing. WER should be
computed from normalized reference and hypothesis text under a documented
normalization policy.

ASR errors of special interest are overlap omissions, substitutions of rare
technical terms, punctuation errors, and fluent but unsupported text.
