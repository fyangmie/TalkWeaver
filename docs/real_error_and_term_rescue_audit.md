# Real ASR Error and Term Rescue Audit

This note defines the lightweight one-week evidence track for TalkWeaver's
RAG term-rescue claim.

## Goal

Do not assume that RAG improves every transcript. First inspect where the ASR
baseline fails, then evaluate term rescue only on rows that actually contain
glossary terms.

The intended claim is narrow:

```text
When ASR misses pre-defined technical terms, names, acronyms, or domain words,
TalkWeaver can use a bounded glossary/RAG step to recover some terms and keep
the evidence boundary explicit.
```

## Step 1: Audit Existing Real ASR Errors

```bash
python experiments/audit_asr_errors.py \
  --input experiments/results/asr_benchmark_real.csv \
  --output experiments/results/asr_error_audit_real.csv \
  --min-error-rate 0.2
```

The output labels likely failure modes:

- `professional_term_or_glossary`
- `proper_noun_or_named_entity`
- `acronym_or_short_code`
- `number_or_quantity`
- `mandarin_low_frequency_or_script`
- `meeting_disfluency_or_truncation`
- `truncation_or_omission`
- `general_asr_error`

This audit is diagnostic. It supports motivation and case selection, not a
model-performance claim by itself.

## Step 2: Evaluate Term Rescue

Use an external/pre-defined glossary when possible:

```bash
python experiments/evaluate_term_rescue_real_audit.py \
  --input experiments/results/asr_benchmark_real.csv \
  --glossary docs/knowledge_base/domain_terms.md \
  --terms-source external_or_predefined \
  --verifier rule \
  --output experiments/results/term_rescue_real_audit.csv \
  --summary-output experiments/results/term_rescue_real_audit_summary.csv
```

If the current ASR rows do not contain any reference terms from the glossary,
the script fails with a clear message. That is the correct result: the current
dataset is not enough to prove this RAG claim.

For the committed current-data upper-bound diagnostic, use the manually
curated oracle glossary derived from observed ASR failures:

```bash
python experiments/evaluate_term_rescue_real_audit.py \
  --input experiments/results/asr_benchmark_real.csv \
  --glossary data/controlled_terms/current_asr_oracle_terms.json \
  --terms-source oracle_diagnostic \
  --verifier rule \
  --output experiments/results/term_rescue_oracle_diagnostic.csv \
  --summary-output experiments/results/term_rescue_oracle_diagnostic_summary.csv
```

`oracle_diagnostic` results must be described as a custom-vocabulary diagnostic,
not as generalization evidence. These rows show whether a bounded term-rescue
mechanism can recover known difficult words such as names, acronyms, and
Chinese low-frequency terms. They do not prove that TalkWeaver automatically
discovers those terms in unseen data, and they do not claim that full-sentence
WER always improves.

## Optional DeepSeek Verifier

The verifier checks RAG candidates before they are applied. It is not an ASR
system and it does not freely rewrite the transcript. It only decides whether
one candidate replacement should be `accept`, `needs_review`, or `reject`.

Use the rule verifier for reproducible offline results:

```bash
--verifier rule
```

To enable DeepSeek, create a local `.env` file in the repository root:

```env
LLM_PROVIDER=deepseek
LLM_API_KEY=your_deepseek_api_key
LLM_MODEL=deepseek-chat
LLM_BASE_URL=https://api.deepseek.com
LLM_TEMPERATURE=0
LLM_TIMEOUT_SECONDS=30
```

Never commit `.env`.

Then run:

```bash
python experiments/evaluate_term_rescue_real_audit.py \
  --input experiments/results/asr_benchmark_real.csv \
  --glossary data/controlled_terms/current_asr_oracle_terms.json \
  --terms-source oracle_diagnostic \
  --verifier llm \
  --output experiments/results/term_rescue_oracle_diagnostic_llm.csv \
  --summary-output experiments/results/term_rescue_oracle_diagnostic_llm_summary.csv
```

`llm_with_rule_fallback` keeps the experiment runnable when the API is missing
or rejects a response. Use `--verifier llm` only when the experiment should fail
strictly if DeepSeek is unavailable. Keep rule-verifier and LLM-verifier
outputs separate. In the current oracle diagnostic, the rule verifier has
higher recall, while the LLM verifier is more conservative and rejects some
plausible repairs when the ASR sentence is too corrupted.

The LLM verifier receives the row language as an explicit JSON field, not only
as prose context. This matters for Mandarin and French cases because the model
must consider Chinese homophones/script variants and French phonetic or accent
confusions.

## If Existing Data Is Insufficient

Use a small public domain-specific subset instead of building a large new
dataset. Good candidates are earnings-call or finance ASR corpora because they
naturally contain company names, acronyms, numbers, and financial terms.

Commit only:

- manifests;
- references when the license permits;
- aggregate CSVs;
- charts;
- documentation.

Do not commit raw audio, private data, model weights, or per-clip prediction
caches.

## Meeting Evidence Boundary

AMI meeting rows can show that meeting ASR has omissions, disfluency issues,
and truncation. Do not claim that TalkWeaver has already reduced meeting WER.
The current safer claim is:

```text
TalkWeaver marks high-risk meeting segments and keeps unsupported correction
proposals visible for review.
```
