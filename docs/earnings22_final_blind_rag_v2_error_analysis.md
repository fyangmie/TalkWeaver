# Earnings-22 Finance RAG + LLM Correction

This diagnostic experiment uses a predefined finance glossary and a conservative LLM verifier. The reference transcript is used only for scoring, not as prompt input.

## Summary

| model_name | num_rows | mean_wer_before | mean_wer_after | mean_wer_delta | mean_term_recall_before | mean_term_recall_after | applied_correction_count | rejected_correction_count | needs_review_count | no_op_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | 12 | 0.186844 | 0.187018 | 0.000174 | 0.958333 | 0.958333 | 1 | 0 | 4 | 1 |
| tiny | 12 | 0.221805 | 0.221978 | 0.000174 | 0.875000 | 0.958333 | 2 | 2 | 0 | 0 |

## Data Scope

- Evaluated rows: 24 ASR rows across 12 audio slice(s).
- This is a diagnostic multi-file subset, not a final held-out benchmark.
- The RAG glossary is treated as external/context knowledge; reference transcripts are used only for scoring.
- ASR-specific error forms should be validated on additional held-out Earnings-22 files before making a final generalization claim.

## Correction Examples

| clip_id | model_name | wer_before | wer_after | term_recall_before | term_recall_after | applied | rejected_count | no_op_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| earnings22_4481952_0000_180s | tiny | 0.100823 | 0.100823 | 1.000000 | 1.000000 |  | 0 | 0 |
| earnings22_4482383_0000_180s | tiny | 0.143141 | 0.143141 | 1.000000 | 1.000000 |  | 1 | 0 |
| earnings22_4449269_0000_180s | tiny | 0.105850 | 0.105850 | 1.000000 | 1.000000 |  | 0 | 0 |
| earnings22_4450779_0000_180s | tiny | 0.287846 | 0.287846 | 0.500000 | 0.500000 |  | 0 | 0 |
| earnings22_4467079_0000_180s | tiny | 0.207101 | 0.207101 | 1.000000 | 1.000000 |  | 0 | 0 |
| earnings22_4481601_0000_180s | tiny | 0.362222 | 0.362222 | 1.000000 | 1.000000 |  | 0 | 0 |
| earnings22_4483296_0000_180s | tiny | 0.152216 | 0.152216 | 0.000000 | 1.000000 | non-gap -> non-GAAP (non-GAAP) | 0 | 0 |
| earnings22_4483668_0000_180s | tiny | 0.220721 | 0.220721 | 1.000000 | 1.000000 |  | 0 | 0 |
| earnings22_4483678_0000_180s | tiny | 0.318750 | 0.320833 | 1.000000 | 1.000000 | non-gap -> non-GAAP (non-GAAP) | 0 | 0 |
| earnings22_4483680_0000_180s | tiny | 0.202020 | 0.202020 | 1.000000 | 1.000000 |  | 0 | 0 |
| earnings22_4423872_0000_180s | tiny | 0.179688 | 0.179688 | 1.000000 | 1.000000 |  | 0 | 0 |
| earnings22_4472895_0000_180s | tiny | 0.381279 | 0.381279 | 1.000000 | 1.000000 |  | 1 | 0 |
| earnings22_4481952_0000_180s | base | 0.067901 | 0.067901 | 1.000000 | 1.000000 |  | 0 | 0 |
| earnings22_4482383_0000_180s | base | 0.143141 | 0.143141 | 1.000000 | 1.000000 |  | 0 | 0 |
| earnings22_4449269_0000_180s | base | 0.091922 | 0.091922 | 1.000000 | 1.000000 |  | 0 | 0 |
| earnings22_4450779_0000_180s | base | 0.238806 | 0.238806 | 0.500000 | 0.500000 |  | 0 | 0 |
| earnings22_4467079_0000_180s | base | 0.210059 | 0.210059 | 1.000000 | 1.000000 |  | 0 | 0 |
| earnings22_4481601_0000_180s | base | 0.257778 | 0.257778 | 1.000000 | 1.000000 |  | 0 | 0 |
| earnings22_4483296_0000_180s | base | 0.134875 | 0.134875 | 1.000000 | 1.000000 |  | 0 | 0 |
| earnings22_4483668_0000_180s | base | 0.198198 | 0.198198 | 1.000000 | 1.000000 |  | 0 | 0 |
| earnings22_4483678_0000_180s | base | 0.270833 | 0.272917 | 1.000000 | 1.000000 | non-gap -> non-GAAP (non-GAAP) | 0 | 1 |
| earnings22_4483680_0000_180s | base | 0.184343 | 0.184343 | 1.000000 | 1.000000 |  | 0 | 0 |
| earnings22_4423872_0000_180s | base | 0.140625 | 0.140625 | 1.000000 | 1.000000 |  | 0 | 0 |
| earnings22_4472895_0000_180s | base | 0.303653 | 0.303653 | 1.000000 | 1.000000 |  | 0 | 0 |

## Interpretation

- The intended win condition is better finance-term recall without unsupported rewrites.
- Numeric-unit corrections must preserve the original number and require dividend/share context.
- Equivalent wording such as cents per share is recorded as no_op rather than applied as a rescue.
- Common tokens such as U.S. are rejected as entity rewrites unless represented by a non-ambiguous source form.
- Style-only differences and filler cleanup are not counted as the core RAG contribution.
