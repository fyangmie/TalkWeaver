# Earnings-22 Finance RAG + LLM Correction

This diagnostic experiment uses a predefined finance glossary and a conservative LLM verifier. The reference transcript is used only for scoring, not as prompt input.

## Summary

| model_name | num_rows | mean_wer_before | mean_wer_after | mean_wer_delta | mean_term_recall_before | mean_term_recall_after | applied_correction_count | rejected_correction_count | needs_review_count | no_op_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | 12 | 0.325876 | 0.325876 | 0.000000 | 0.833333 | 1.000000 | 2 | 3 | 1 | 1 |
| tiny | 12 | 0.372716 | 0.372716 | 0.000000 | 0.750000 | 0.916667 | 2 | 4 | 25 | 3 |

## Data Scope

- Evaluated rows: 24 ASR rows across 12 audio slice(s).
- This is a diagnostic multi-file subset, not a final held-out benchmark.
- The RAG glossary is treated as external/context knowledge; reference transcripts are used only for scoring.
- ASR-specific error forms should be validated on additional held-out Earnings-22 files before making a final generalization claim.

## Correction Examples

| clip_id | model_name | wer_before | wer_after | term_recall_before | term_recall_after | applied | rejected_count | no_op_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| earnings22_4474955_0000_180s | tiny | 0.293399 | 0.293399 | 0.000000 | 0.000000 |  | 1 | 1 |
| earnings22_4483046_0000_180s | tiny | 0.412371 | 0.412371 | 1.000000 | 1.000000 |  | 0 | 0 |
| earnings22_4468919_0000_180s | tiny | 0.259424 | 0.259424 | 0.000000 | 1.000000 | non-gap -> non-GAAP (non-GAAP) | 0 | 0 |
| earnings22_4475604_0000_180s | tiny | 0.318766 | 0.318766 | 1.000000 | 1.000000 |  | 0 | 0 |
| earnings22_4471586_0000_180s | tiny | 0.391982 | 0.391982 | 1.000000 | 1.000000 |  | 0 | 0 |
| earnings22_4482110_0000_180s | tiny | 0.252101 | 0.252101 | 0.000000 | 1.000000 | non-gap -> non-GAAP (non-GAAP) | 0 | 0 |
| earnings22_4446796_0000_180s | tiny | 0.231920 | 0.231920 | 1.000000 | 1.000000 |  | 1 | 0 |
| earnings22_4483623_0000_180s | tiny | 0.114286 | 0.114286 | 1.000000 | 1.000000 |  | 0 | 1 |
| earnings22_4470290_0000_180s | tiny | 1.377593 | 1.377593 | 1.000000 | 1.000000 |  | 1 | 0 |
| earnings22_4469528_0000_180s | tiny | 0.189873 | 0.189873 | 1.000000 | 1.000000 |  | 0 | 0 |
| earnings22_4470570_0000_180s | tiny | 0.411364 | 0.411364 | 1.000000 | 1.000000 |  | 1 | 1 |
| earnings22_4329526_0000_180s | tiny | 0.219512 | 0.219512 | 1.000000 | 1.000000 |  | 0 | 0 |
| earnings22_4474955_0000_180s | base | 0.259169 | 0.259169 | 0.000000 | 1.000000 | non-gap -> non-GAAP (non-GAAP) | 0 | 0 |
| earnings22_4483046_0000_180s | base | 0.342268 | 0.342268 | 1.000000 | 1.000000 |  | 0 | 0 |
| earnings22_4468919_0000_180s | base | 0.150776 | 0.150776 | 0.000000 | 1.000000 | non-gap -> non-GAAP (non-GAAP) | 0 | 0 |
| earnings22_4475604_0000_180s | base | 0.311054 | 0.311054 | 1.000000 | 1.000000 |  | 0 | 0 |
| earnings22_4471586_0000_180s | base | 0.320713 | 0.320713 | 1.000000 | 1.000000 |  | 0 | 0 |
| earnings22_4482110_0000_180s | base | 0.260504 | 0.260504 | 1.000000 | 1.000000 |  | 0 | 0 |
| earnings22_4446796_0000_180s | base | 0.236908 | 0.236908 | 1.000000 | 1.000000 |  | 1 | 0 |
| earnings22_4483623_0000_180s | base | 0.101099 | 0.101099 | 1.000000 | 1.000000 |  | 0 | 1 |
| earnings22_4470290_0000_180s | base | 1.215768 | 1.215768 | 1.000000 | 1.000000 |  | 1 | 0 |
| earnings22_4469528_0000_180s | base | 0.179325 | 0.179325 | 1.000000 | 1.000000 |  | 0 | 0 |
| earnings22_4470570_0000_180s | base | 0.350000 | 0.350000 | 1.000000 | 1.000000 |  | 1 | 0 |
| earnings22_4329526_0000_180s | base | 0.182927 | 0.182927 | 1.000000 | 1.000000 |  | 0 | 0 |

## Interpretation

- The intended win condition is better finance-term recall without unsupported rewrites.
- Numeric-unit corrections must preserve the original number and require dividend/share context.
- Equivalent wording such as cents per share is recorded as no_op rather than applied as a rescue.
- Common tokens such as U.S. are rejected as entity rewrites unless represented by a non-ambiguous source form.
- Style-only differences and filler cleanup are not counted as the core RAG contribution.
