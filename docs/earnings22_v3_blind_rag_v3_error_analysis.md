# Earnings-22 Finance RAG + LLM Correction

This diagnostic experiment uses a predefined finance glossary and a conservative LLM verifier. The reference transcript is used only for scoring, not as prompt input.

## Summary

| model_name | num_rows | mean_wer_before | mean_wer_after | mean_wer_delta | mean_term_recall_before | mean_term_recall_after | applied_correction_count | rejected_correction_count | needs_review_count | no_op_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | 6 | 0.212099 | 0.212099 | 0.000000 | 0.833333 | 1.000000 | 2 | 0 | 2 | 0 |
| tiny | 6 | 0.251941 | 0.251941 | 0.000000 | 0.833333 | 0.833333 | 0 | 1 | 1 | 0 |

## Data Scope

- Evaluated rows: 12 ASR rows across 6 audio slice(s).
- This is a diagnostic multi-file subset, not a final held-out benchmark.
- The RAG glossary is treated as external/context knowledge; reference transcripts are used only for scoring.
- ASR-specific error forms should be validated on additional held-out Earnings-22 files before making a final generalization claim.

## Correction Examples

| clip_id | model_name | wer_before | wer_after | term_recall_before | term_recall_after | applied | rejected_count | no_op_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| earnings22_4473238_0000_180s | tiny | 0.305810 | 0.305810 | 0.000000 | 0.000000 |  | 1 | 0 |
| earnings22_4475486_0000_180s | tiny | 0.305195 | 0.305195 | 1.000000 | 1.000000 |  | 0 | 0 |
| earnings22_4479524_0000_180s | tiny | 0.306931 | 0.306931 | 1.000000 | 1.000000 |  | 0 | 0 |
| earnings22_4469088_0000_180s | tiny | 0.215385 | 0.215385 | 1.000000 | 1.000000 |  | 0 | 0 |
| earnings22_4470684_0000_180s | tiny | 0.189922 | 0.189922 | 1.000000 | 1.000000 |  | 0 | 0 |
| earnings22_4473837_0000_180s | tiny | 0.188406 | 0.188406 | 1.000000 | 1.000000 |  | 0 | 0 |
| earnings22_4473238_0000_180s | base | 0.204893 | 0.204893 | 0.000000 | 1.000000 | non-gap -> non-GAAP (non-GAAP) | 0 | 0 |
| earnings22_4475486_0000_180s | base | 0.269481 | 0.269481 | 1.000000 | 1.000000 |  | 0 | 0 |
| earnings22_4479524_0000_180s | base | 0.272277 | 0.272277 | 1.000000 | 1.000000 | UPS -> UEPS (UEPS) | 0 | 0 |
| earnings22_4469088_0000_180s | base | 0.184615 | 0.184615 | 1.000000 | 1.000000 |  | 0 | 0 |
| earnings22_4470684_0000_180s | base | 0.186047 | 0.186047 | 1.000000 | 1.000000 |  | 0 | 0 |
| earnings22_4473837_0000_180s | base | 0.155280 | 0.155280 | 1.000000 | 1.000000 |  | 0 | 0 |

## Interpretation

- The intended win condition is better finance-term recall without unsupported rewrites.
- Numeric-unit corrections must preserve the original number and require dividend/share context.
- Equivalent wording such as cents per share is recorded as no_op rather than applied as a rescue.
- Common tokens such as U.S. are rejected as entity rewrites unless represented by a non-ambiguous source form.
- Style-only differences and filler cleanup are not counted as the core RAG contribution.
