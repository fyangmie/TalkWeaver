# Earnings-22 Held-Out Error Analysis

This report summarizes the correction decisions on the frozen held-out subset. Reference text is used only for scoring and analysis.

## Decision Counts

| name | count |
| --- | --- |
| applied | 4 |
| needs_review | 26 |
| no_op | 4 |
| rejected | 7 |
| unchanged | 11 |

## Applied Correction Categories

| name | count |
| --- | --- |
| financial_metric | 4 |

## Examples

| clip_id | model_name | source_text | replacement_text | canonical_term | category | decision | wer_delta | term_recall_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| earnings22_4474955_0000_180s | tiny | U.S. | UEPS | UEPS | company_name | rejected | 0.000000 | 0.000000 |
| earnings22_4474955_0000_180s | tiny | Chidian |  | Sify Technologies | needs_review | needs_review | 0.000000 | 0.000000 |
| earnings22_4474955_0000_180s | tiny | Purigen |  | Sify Technologies | needs_review | needs_review | 0.000000 | 0.000000 |
| earnings22_4474955_0000_180s | tiny | Minlow |  | Min-Luo | needs_review | needs_review | 0.000000 | 0.000000 |
| earnings22_4474955_0000_180s | tiny | Chidiu |  | Sify Technologies | needs_review | needs_review | 0.000000 | 0.000000 |
| earnings22_4474955_0000_180s | tiny | Human Rights Commission |  | Human Resources | needs_review | needs_review | 0.000000 | 0.000000 |
| earnings22_4474955_0000_180s | tiny | Graduate Security Slication Reform Act |  | Private Securities Litigation Reform Act | needs_review | needs_review | 0.000000 | 0.000000 |
| earnings22_4474955_0000_180s | tiny | Cynthia |  | SEC filing | needs_review | needs_review | 0.000000 | 0.000000 |
| earnings22_4474955_0000_180s | tiny | on-orchist gap |  | non-GAAP | needs_review | needs_review | 0.000000 | 0.000000 |
| earnings22_4474955_0000_180s | tiny | on-orchist-anung gap |  | non-GAAP | needs_review | needs_review | 0.000000 | 0.000000 |
| earnings22_4474955_0000_180s | tiny | good-dent |  | good | needs_review | needs_review | 0.000000 | 0.000000 |
| earnings22_4474955_0000_180s | tiny | cash credit |  | cash credit | needs_review | needs_review | 0.000000 | 0.000000 |
| earnings22_4474955_0000_180s | tiny | loan food |  | loan book | needs_review | needs_review | 0.000000 | 0.000000 |
| earnings22_4474955_0000_180s | tiny | 3.4 billion I am being |  | 3.4 billion RMB | needs_review | needs_review | 0.000000 | 0.000000 |
| earnings22_4474955_0000_180s | tiny | U.S. Security and Exchange Commission |  | U.S. Securities and Exchange Commission | no_op | no_op | 0.000000 | 0.000000 |
| earnings22_4468919_0000_180s | tiny | non-gap | non-GAAP | non-GAAP | financial_metric | applied | 0.000000 | 1.000000 |
| earnings22_4468919_0000_180s | tiny | Wilma Negra |  | unknown | needs_review | needs_review | 0.000000 | 1.000000 |
| earnings22_4468919_0000_180s | tiny | Monegras |  | unknown | needs_review | needs_review | 0.000000 | 1.000000 |
| earnings22_4468919_0000_180s | tiny | Diego Hello |  | unknown | needs_review | needs_review | 0.000000 | 1.000000 |
| earnings22_4468919_0000_180s | tiny | Marko Heradine |  | unknown | needs_review | needs_review | 0.000000 | 1.000000 |

## Interpretation Notes

- `unchanged` rows are not failures by default; they usually mean no safe glossary-grounded candidate was available or the baseline already had the terms.
- `no_op` rows are equivalent wording or already acceptable forms, not term-rescue wins.
- Held-out errors that look fixable but are missing from the frozen glossary should be recorded for a future dev iteration, not patched into this held-out run.
