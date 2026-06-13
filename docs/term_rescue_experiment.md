# Controlled Term Rescue and Correction Safety

## Purpose

Phase 2E could verify ConversationMap completeness over fixed real ASR
predictions, but the 17-clip public subset does not contain annotated
TalkWeaver technical-term failures. It therefore could not measure Misheard
Word Rescue or correction safety.

Phase 2F adds a separate, controlled text experiment for:

- glossary, fuzzy, and phonetic-like candidate retrieval;
- context-aware handling of ambiguous common words;
- deterministic evidence-grounded correction;
- optional strict real LLM correction;
- unsupported-change rejection and `needs_review` behavior.

## Fixture Policy

`data/controlled_terms/term_rescue_cases.jsonl` contains 25 authored text
fixtures. They are not public audio, self-recorded audio, or measured ASR
predictions. Every result row is labeled as a controlled technical-term
fixture and remains separate from Phase 2C public-data ASR results.

The cases cover `pyannote`, `pyannote.audio`, speaker diarization, RAG, WER,
DER, temporal anchor, overlap-aware correction, faster-whisper, Whisper,
CTranslate2, Qwen, TagSpeech, and DM-ASR. Four negative controls preserve the
ordinary meanings of `rack`, `where`, `dear`, and `tag speech`.

The reference glossary is
`data/controlled_terms/reference_terms.json`. It records canonical terms,
aliases, spoken forms, likely error forms, language, category, and allowed
contexts.

## Variants

| Variant | Retrieval | Correction |
| --- | --- | --- |
| `no_retrieval` | None | None |
| `exact_glossary` | Exact canonical/error forms | None |
| `fuzzy` | Sequence similarity | None |
| `phonetic_like` | Lightweight consonant-pattern similarity | None |
| `fused` | Combined retrieval plus context gating | None |
| `fused_plus_rule_correction` | Fused | Deterministic supported substitutions |
| `fused_plus_llm_correction` | Fused | Strict configured LLM, no silent fallback |

Fused retrieval withholds ambiguous candidates when domain context is absent.
The withheld evidence is written to
`experiments/results/term_candidates_controlled.jsonl` and marks the case for
review instead of changing the text.

## Metrics

- term precision, recall, and F1 over expected canonical terms;
- false-positive and missed terms;
- WER for whitespace languages and CER for Mandarin carrier text;
- text error before and after correction;
- unsupported changes;
- review flags;
- API use, fallback use, provider, model, and prompt version.

Parent/child terms such as `pyannote` and `pyannote.audio`, or `diarization`
and `speaker diarization`, are treated as compatible retrieval hits when one
is explicitly contained in the other.

## Commands

Offline:

```bash
python experiments/run_term_rescue_experiment.py \
  --cases data/controlled_terms/term_rescue_cases.jsonl \
  --terms data/controlled_terms/reference_terms.json \
  --output experiments/results/term_rescue_controlled.csv \
  --candidates-output experiments/results/term_candidates_controlled.jsonl
```

Optional strict LLM:

```bash
python experiments/run_term_rescue_experiment.py \
  --cases data/controlled_terms/term_rescue_cases.jsonl \
  --terms data/controlled_terms/reference_terms.json \
  --output experiments/results/term_rescue_controlled.csv \
  --candidates-output experiments/results/term_candidates_controlled.jsonl \
  --include-llm-if-configured
```

Summary and charts:

```bash
python experiments/summarize_term_rescue.py \
  --input experiments/results/term_rescue_controlled.csv \
  --output experiments/results/term_rescue_summary_controlled.csv

python experiments/plot_term_rescue.py \
  --input experiments/results/term_rescue_controlled.csv \
  --output-dir assets/result_charts
```

## Results

The June 13, 2026 run evaluated 25 cases and seven variants, producing 175
rows. The optional real-LLM variant used DeepSeek `deepseek-chat`,
`talkweaver.correction.v1`, temperature `0`, with 25 API attempts and zero
fallbacks.

| Variant | Mean term F1 | Negative-control false positives | Mean text error after | Needs review |
| --- | ---: | ---: | ---: | ---: |
| `no_retrieval` | 0.1600 | 0 | 0.2880 | 0 |
| `exact_glossary` | 0.8267 | 4 | 0.2880 | 0 |
| `fuzzy` | 0.8400 | 4 | 0.2880 | 0 |
| `phonetic_like` | 0.8400 | 3 | 0.2880 | 0 |
| `fused` | 1.0000 | 0 | 0.2880 | 4 |
| `fused_plus_rule_correction` | 1.0000 | 0 | 0.0000 | 4 |
| `fused_plus_llm_correction` | 1.0000 | 0 | 0.0812 | 8 |

Mean text error before correction was `0.2880` for every variant. The rule
result reaches zero because the fixtures were deliberately authored from the
same known substitution set; this is a controlled capability check, not a
generalization claim.

## Positive and Negative Examples

Supported context:

```text
Raw:       the rack glossary retrieves technical terms
Corrected: the RAG glossary retrieves technical terms
Evidence:  retrieval augmented ASR correction
```

Negative control:

```text
Raw:       put the router on the rack near the wall
Corrected: put the router on the rack near the wall
Decision:  RAG candidate withheld; needs_review=true
Evidence:  physical equipment rack in a server room
```

The same safety behavior applies to ordinary `where`, greeting `dear`, and
non-paper `tag speech` contexts.

## Hallucination Watchdog Interpretation

Accepted corrections introduced zero unsupported tokens. Four DeepSeek
outputs were rejected because they changed content or word order beyond the
strict glossary-grounded transformation. Those rows:

- retain `raw_asr_text` as `corrected_text`;
- set `api_used=true`;
- set `fallback_used=false`;
- store a non-empty `correction_error`;
- set `needs_review=true`.

The LLM therefore improved many fixtures, but the rule variant was more
complete on this deliberately deterministic substitution set. The important
safety result is that rejected model output was not silently adopted.

## Outputs

- `experiments/results/term_rescue_controlled.csv`
- `experiments/results/term_rescue_summary_controlled.csv`
- `experiments/results/term_candidates_controlled.jsonl`
- `assets/result_charts/term_rescue_f1_by_variant.png`
- `assets/result_charts/term_rescue_false_positive_by_variant.png`
- `assets/result_charts/term_rescue_error_delta.png`

## Limitations

- The fixtures are authored text, not acoustic evidence.
- The deterministic rules share the fixture substitution vocabulary.
- French and Mandarin coverage uses carrier sentences with English technical
  terms; it is not a complete multilingual named-entity benchmark.
- API behavior can vary by provider version and time.
- Fuzzy and phonetic-like matching is intentionally lightweight.
- Real demo/mobile audio still needs technical-term reference annotations.

The next evidence step is to record or source consent-safe technical-term
audio and connect the same audit fields to the AI Meeting Detective frontend.
