# Earnings-22 Held-Out Error Analysis

This report summarizes the correction decisions on the frozen held-out subset. Reference text is used only for scoring and analysis.

## Decision Counts

| name | count |
| --- | --- |
| applied | 3 |
| needs_review | 4 |
| no_op | 1 |
| rejected | 2 |
| unchanged | 19 |

## Applied Correction Categories

| name | count |
| --- | --- |
| financial_metric | 3 |

## Examples

| clip_id | model_name | source_text | replacement_text | canonical_term | category | decision | wer_delta | term_recall_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| earnings22_4482383_0000_180s | tiny | U.S. | UEPS | UEPS | company_name | rejected | 0.000000 | 0.000000 |
| earnings22_4483296_0000_180s | tiny | non-gap | non-GAAP | non-GAAP | financial_metric | applied | 0.000000 | 1.000000 |
| earnings22_4483678_0000_180s | tiny | non-gap | non-GAAP | non-GAAP | financial_metric | applied | 0.002083 | 0.000000 |
| earnings22_4472895_0000_180s | tiny | U.S. | UEPS | UEPS | company_name | rejected | 0.000000 | 0.000000 |
| earnings22_4483678_0000_180s | base | non-gap | non-GAAP | non-GAAP | financial_metric | applied | 0.002083 | 0.000000 |
| earnings22_4483678_0000_180s | base | cupping notes |  | accompanying notes | needs_review | needs_review | 0.002083 | 0.000000 |
| earnings22_4483678_0000_180s | base | Cedar and Enter |  | SEDAR and SEDAR+ | needs_review | needs_review | 0.002083 | 0.000000 |
| earnings22_4483678_0000_180s | base | hand winds |  | headwinds | needs_review | needs_review | 0.002083 | 0.000000 |
| earnings22_4483678_0000_180s | base | mark out |  | market outlook | needs_review | needs_review | 0.002083 | 0.000000 |
| earnings22_4483678_0000_180s | base | cents a share |  | cents a share | number_unit | no_op | 0.002083 | 0.000000 |

## Interpretation Notes

- `unchanged` rows are not failures by default; they usually mean no safe glossary-grounded candidate was available or the baseline already had the terms.
- `no_op` rows are equivalent wording or already acceptable forms, not term-rescue wins.
- Held-out errors that look fixable but are missing from the frozen glossary should be recorded for a future dev iteration, not patched into this held-out run.
