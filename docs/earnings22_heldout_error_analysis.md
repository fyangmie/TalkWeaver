# Earnings-22 Held-Out Error Analysis

This report summarizes the correction decisions on the frozen held-out subset. Reference text is used only for scoring and analysis.

## Decision Counts

| name | count |
| --- | --- |
| applied | 13 |
| needs_review | 29 |
| unchanged | 11 |

## Applied Correction Categories

| name | count |
| --- | --- |
| company_name | 7 |
| financial_metric | 4 |
| number_unit | 2 |

## Examples

| clip_id | model_name | source_text | replacement_text | canonical_term | category | decision | wer_delta | term_recall_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| earnings22_4474955_0000_180s | tiny | U.S. | UEPS | UEPS | company_name | applied | 0.000000 | 0.000000 |
| earnings22_4474955_0000_180s | tiny | Chidian |  | Sify Technologies | needs_review | needs_review | 0.000000 | 0.000000 |
| earnings22_4474955_0000_180s | tiny | Purigen |  | Sify Technologies | needs_review | needs_review | 0.000000 | 0.000000 |
| earnings22_4474955_0000_180s | tiny | Minlow |  | Min-Luo | needs_review | needs_review | 0.000000 | 0.000000 |
| earnings22_4474955_0000_180s | tiny | Chidiu |  | Sify Technologies | needs_review | needs_review | 0.000000 | 0.000000 |
| earnings22_4468919_0000_180s | tiny | non-gap | non-GAAP | non-GAAP | financial_metric | applied | 0.000000 | 1.000000 |
| earnings22_4468919_0000_180s | tiny | Wilma Negra |  | Monegras | needs_review | needs_review | 0.000000 | 1.000000 |
| earnings22_4468919_0000_180s | tiny | Monegras |  | Monegras | needs_review | needs_review | 0.000000 | 1.000000 |
| earnings22_4468919_0000_180s | tiny | Diego Hello |  | Diego Hello | needs_review | needs_review | 0.000000 | 1.000000 |
| earnings22_4468919_0000_180s | tiny | Marko Heradine |  | Marko Heradine | needs_review | needs_review | 0.000000 | 1.000000 |
| earnings22_4468919_0000_180s | tiny | solely the man in high historic figures |  | solid demand in high historic figures | needs_review | needs_review | 0.000000 | 1.000000 |
| earnings22_4468919_0000_180s | tiny | superset |  | supply set | needs_review | needs_review | 0.000000 | 1.000000 |
| earnings22_4482110_0000_180s | tiny | non-gap | non-GAAP | non-GAAP | financial_metric | applied | 0.000000 | 1.000000 |
| earnings22_4446796_0000_180s | tiny | U.S. | UEPS | UEPS | company_name | applied | 0.000000 | 0.000000 |
| earnings22_4483623_0000_180s | tiny | 42 cents per share | 42 cents a share | cents a share | number_unit | applied | 0.002198 | 0.000000 |
| earnings22_4470290_0000_180s | tiny | U.S. | UEPS | UEPS | company_name | applied | 0.000000 | 0.000000 |
| earnings22_4470570_0000_180s | tiny | U.S. | UEPS | UEPS | company_name | applied | 0.000000 | 0.000000 |
| earnings22_4474955_0000_180s | base | non-gap | non-GAAP | non-GAAP | financial_metric | applied | 0.000000 | 1.000000 |
| earnings22_4474955_0000_180s | base | U.S. |  | UEPS | needs_review | needs_review | 0.000000 | 1.000000 |
| earnings22_4468919_0000_180s | base | non-gap | non-GAAP | non-GAAP | financial_metric | applied | 0.000000 | 1.000000 |

## Interpretation Notes

- `unchanged` rows are not failures by default; they usually mean no safe glossary-grounded candidate was available or the baseline already had the terms.
- Held-out errors that look fixable but are missing from the frozen glossary should be recorded for a future dev iteration, not patched into this held-out run.
