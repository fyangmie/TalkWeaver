# Earnings-22 Finance RAG + LLM Correction

This diagnostic experiment uses a predefined finance glossary and a conservative LLM verifier. The reference transcript is used only for scoring, not as prompt input.

## Summary

| model_name | num_rows | mean_wer_before | mean_wer_after | mean_wer_delta | mean_term_recall_before | mean_term_recall_after | applied_correction_count | rejected_correction_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | 1 | 0.227642 | 0.222222 | -0.005420 | 0.923077 | 1.000000 | 1 | 0 |
| tiny | 1 | 0.249322 | 0.238482 | -0.010840 | 0.769231 | 1.000000 | 4 | 0 |

## Correction Examples

| clip_id | model_name | wer_before | wer_after | term_recall_before | term_recall_after | applied | rejected_count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| earnings22_4453225_intro_180s | tiny | 0.249322 | 0.238482 | 0.769231 | 1.000000 | 262 cents to share -> 262 cents a share (cents a share); regional brand -> Regional Brands (Regional Brands); several focus brands -> Sterile Focus Brands (Sterile Focus Brands); continuing operation -> continuing operations (continuing operations) | 0 |
| earnings22_4453225_intro_180s | base | 0.227642 | 0.222222 | 0.923077 | 1.000000 | 262 seems to share -> 262 cents a share (cents a share) | 0 |

## Interpretation

- The intended win condition is better finance-term recall without unsupported rewrites.
- Numeric-unit corrections must preserve the original number and require dividend/share context.
- Style-only differences and filler cleanup are not counted as the core RAG contribution.
