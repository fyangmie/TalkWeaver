# Earnings-22 Finance RAG + LLM Correction

This diagnostic experiment uses a predefined finance glossary and a conservative LLM verifier. The reference transcript is used only for scoring, not as prompt input.

## Summary

| model_name | num_rows | mean_wer_before | mean_wer_after | mean_wer_delta | mean_term_recall_before | mean_term_recall_after | applied_correction_count | rejected_correction_count | needs_review_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | 4 | 0.193489 | 0.189808 | -0.003681 | 0.593750 | 0.875000 | 7 | 0 | 0 |
| tiny | 4 | 0.218628 | 0.214269 | -0.004358 | 0.604167 | 0.875000 | 7 | 0 | 0 |

## Data Scope

- Evaluated rows: 8 ASR rows across 4 audio slice(s).
- This is a diagnostic multi-file subset, not a final held-out benchmark.
- The RAG glossary is treated as external/context knowledge; reference transcripts are used only for scoring.
- ASR-specific error forms should be validated on additional held-out Earnings-22 files before making a final generalization claim.

## Correction Examples

| clip_id | model_name | wer_before | wer_after | term_recall_before | term_recall_after | applied | rejected_count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| earnings22_4453225_0000_180s | tiny | 0.230352 | 0.222222 | 0.750000 | 1.000000 | regional brand -> Regional Brands (Regional Brands); several focus brands -> Sterile Focus Brands (Sterile Focus Brands); continuing operation -> continuing operations (continuing operations) | 0 |
| earnings22_4467434_0000_180s | tiny | 0.288618 | 0.288618 | 0.500000 | 0.500000 |  | 0 |
| earnings22_4481221_0000_180s | tiny | 0.230233 | 0.220930 | 0.166667 | 1.000000 | non-capped -> non-GAAP (non-GAAP); City Technology's -> Sify Technologies (Sify Technologies); Peacock Knowledge Limited -> Sify Technologies Limited (Sify Technologies Limited); sifetechnologies.com -> sifytechnologies.com (sifytechnologies.com) | 0 |
| earnings22_4462231_0000_180s | tiny | 0.125307 | 0.125307 | 1.000000 | 1.000000 |  | 0 |
| earnings22_4453225_0000_180s | base | 0.224932 | 0.219512 | 0.875000 | 1.000000 | 262 seems to share -> 262 cents a share (cents a share) | 0 |
| earnings22_4467434_0000_180s | base | 0.186992 | 0.186992 | 0.500000 | 0.500000 |  | 0 |
| earnings22_4481221_0000_180s | base | 0.202326 | 0.193023 | 0.000000 | 1.000000 | non-gap -> non-GAAP (non-GAAP); IFR -> IFRS (IFRS); SIFI -> Sify (Sify); CIFI Technologies -> Sify Technologies (Sify Technologies); P. Technology Limited -> Sify Technologies Limited (Sify Technologies Limited); sefitechnologies.com -> sifytechnologies.com (sifytechnologies.com) | 0 |
| earnings22_4462231_0000_180s | base | 0.159705 | 0.159705 | 1.000000 | 1.000000 |  | 0 |

## Interpretation

- The intended win condition is better finance-term recall without unsupported rewrites.
- Numeric-unit corrections must preserve the original number and require dividend/share context.
- Style-only differences and filler cleanup are not counted as the core RAG contribution.
