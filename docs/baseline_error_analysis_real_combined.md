# Baseline ASR Error Pattern Analysis

This report separates strict WER from actionable error types. It is diagnostic and should not be presented as a final benchmark claim.

## Model Summary

| dataset_name | language | model_name | rows | mean_wer | mean_cleaned_wer | strict_cleaned_gap | top_error_categories |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AMI Meeting Corpus | en | base | 4 | 0.315848 | 0.221154 | 0.094694 | [["disfluency_or_style_error", 14], ["omission_or_insertion", 6], ["semantic_word_error", 4], ["function_word_error", 4]] |
| AMI Meeting Corpus | en | tiny | 4 | 0.344866 | 0.250000 | 0.094866 | [["disfluency_or_style_error", 16], ["semantic_word_error", 10], ["omission_or_insertion", 6], ["function_word_error", 4]] |
| Earnings-22 | en | base | 1 | 0.227642 | 0.227642 | 0.000000 | [["disfluency_or_style_error", 28], ["function_word_error", 13], ["semantic_word_error", 7], ["domain_term_error", 2], ["omission_or_insertion", 2]] |
| Earnings-22 | en | tiny | 1 | 0.249322 | 0.249322 | 0.000000 | [["disfluency_or_style_error", 27], ["semantic_word_error", 12], ["function_word_error", 11], ["domain_term_error", 4], ["number_unit_error", 2]] |
| Google FLEURS | en | base | 5 | 0.154242 | 0.154242 | 0.000000 | [["semantic_word_error", 7], ["function_word_error", 1]] |
| Google FLEURS | en | tiny | 5 | 0.296212 | 0.296212 | 0.000000 | [["semantic_word_error", 11], ["omission_or_insertion", 1]] |
| Google FLEURS | fr | base | 5 | 0.273839 | 0.273839 | 0.000000 | [["semantic_word_error", 18]] |
| Google FLEURS | fr | tiny | 5 | 0.438703 | 0.438703 | 0.000000 | [["semantic_word_error", 24], ["omission_or_insertion", 1]] |
| Google FLEURS | zh-CN | base | 5 | 0.089651 | 0.089651 | 0.000000 | [["semantic_word_error", 5]] |
| Google FLEURS | zh-CN | tiny | 5 | 0.276133 | 0.276133 | 0.000000 | [["semantic_word_error", 5]] |

## Error Categories

| dataset_name | language | model_name | error_category | count | share | example_reference | example_hypothesis | recommendation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AMI Meeting Corpus | en | base | disfluency_or_style_error | 14 | 0.500000 | um |  | Report cleaned WER separately; do not optimize RAG for this category. |
| AMI Meeting Corpus | en | base | function_word_error | 4 | 0.142857 | our |  | Use stronger ASR/context; glossary RAG should usually avoid this. |
| AMI Meeting Corpus | en | base | omission_or_insertion | 6 | 0.214286 |  | great | Inspect audio and segment boundaries; consider VAD/chunking changes. |
| AMI Meeting Corpus | en | base | semantic_word_error | 4 | 0.142857 | this is | it's | Collect more examples, then decide whether ASR API or targeted RAG helps. |
| AMI Meeting Corpus | en | tiny | disfluency_or_style_error | 16 | 0.444444 | um |  | Report cleaned WER separately; do not optimize RAG for this category. |
| AMI Meeting Corpus | en | tiny | function_word_error | 4 | 0.111111 | our |  | Use stronger ASR/context; glossary RAG should usually avoid this. |
| AMI Meeting Corpus | en | tiny | omission_or_insertion | 6 | 0.166667 | kind of |  | Inspect audio and segment boundaries; consider VAD/chunking changes. |
| AMI Meeting Corpus | en | tiny | semantic_word_error | 10 | 0.277778 | this is | it's | Collect more examples, then decide whether ASR API or targeted RAG helps. |
| Earnings-22 | en | base | date_time_error | 1 | 0.018519 | end of | in the | Use stronger ASR or audio-backed verification; RAG alone is weak here. |
| Earnings-22 | en | base | disfluency_or_style_error | 28 | 0.518519 | gonna | going to | Report cleaned WER separately; do not optimize RAG for this category. |
| Earnings-22 | en | base | domain_term_error | 2 | 0.037037 | double-digit | double digit | Use a predeclared finance glossary plus conservative LLM verifier. |
| Earnings-22 | en | base | function_word_error | 13 | 0.240741 |  | the | Use stronger ASR/context; glossary RAG should usually avoid this. |
| Earnings-22 | en | base | number_unit_error | 1 | 0.018519 | cents a | seems to | Add numeric-unit candidate rules and verify against local context/audio. |
| Earnings-22 | en | base | omission_or_insertion | 2 | 0.037037 | here |  | Inspect audio and segment boundaries; consider VAD/chunking changes. |
| Earnings-22 | en | base | semantic_word_error | 7 | 0.129630 | notice at | notes of | Collect more examples, then decide whether ASR API or targeted RAG helps. |
| Earnings-22 | en | tiny | date_time_error | 1 | 0.016949 | end of 30 | in the 3d | Use stronger ASR or audio-backed verification; RAG alone is weak here. |
| Earnings-22 | en | tiny | disfluency_or_style_error | 27 | 0.457627 | gonna | going to | Report cleaned WER separately; do not optimize RAG for this category. |
| Earnings-22 | en | tiny | domain_term_error | 4 | 0.067797 | operations | operation | Use a predeclared finance glossary plus conservative LLM verifier. |
| Earnings-22 | en | tiny | function_word_error | 11 | 0.186441 |  | the | Use stronger ASR/context; glossary RAG should usually avoid this. |
| Earnings-22 | en | tiny | number_unit_error | 2 | 0.033898 | at | has | Add numeric-unit candidate rules and verify against local context/audio. |
| Earnings-22 | en | tiny | omission_or_insertion | 2 | 0.033898 | here |  | Inspect audio and segment boundaries; consider VAD/chunking changes. |
| Earnings-22 | en | tiny | semantic_word_error | 12 | 0.203390 | notice | notes | Collect more examples, then decide whether ASR API or targeted RAG helps. |
| Google FLEURS | en | base | function_word_error | 1 | 0.125000 | in | and | Use stronger ASR/context; glossary RAG should usually avoid this. |
| Google FLEURS | en | base | semantic_word_error | 7 | 0.875000 | javanese | japanese | Collect more examples, then decide whether ASR API or targeted RAG helps. |
| Google FLEURS | en | tiny | omission_or_insertion | 1 | 0.083333 | singing |  | Inspect audio and segment boundaries; consider VAD/chunking changes. |
| Google FLEURS | en | tiny | semantic_word_error | 11 | 0.916667 | you are | you're | Collect more examples, then decide whether ASR API or targeted RAG helps. |
| Google FLEURS | fr | base | semantic_word_error | 18 | 1.000000 | telles | dès | Collect more examples, then decide whether ASR API or targeted RAG helps. |
| Google FLEURS | fr | tiny | omission_or_insertion | 1 | 0.040000 | le |  | Inspect audio and segment boundaries; consider VAD/chunking changes. |
| Google FLEURS | fr | tiny | semantic_word_error | 24 | 0.960000 | telles | dès | Collect more examples, then decide whether ASR API or targeted RAG helps. |
| Google FLEURS | zh-CN | base | semantic_word_error | 5 | 1.000000 | 报告警告称没有人能保证目前在伊拉克采取的任何行动能够阻止宗派战争不断增长的暴力或走向混乱 | 报告就告称没有人能保证目前在伊拉克采取的任何行动能够阻止宗派战争不断增长的暴力或走向混乱 | Collect more examples, then decide whether ASR API or targeted RAG helps. |
| Google FLEURS | zh-CN | tiny | semantic_word_error | 5 | 1.000000 | 报告警告称没有人能保证目前在伊拉克采取的任何行动能够阻止宗派战争不断增长的暴力或走向混乱 | 报告警告称没有人能保证目前在伊拉克采取的任何行动能够阻止宗派战争不断增长的暴力或走下混乱 | Collect more examples, then decide whether ASR API or targeted RAG helps. |

## Representative Errors

| clip_id | model_name | error_category | reference_span | hypothesis_span | impact |
| --- | --- | --- | --- | --- | --- |
| fleurs_en_1548 | tiny | semantic_word_error | you are | you're | meaning_may_change |
| fleurs_en_1620 | tiny | semantic_word_error | archipelago javanese | archaeopilago japanese | meaning_may_change |
| fleurs_en_1620 | tiny | semantic_word_error | features an | speeches and | meaning_may_change |
| fleurs_en_1620 | tiny | semantic_word_error | javanese favor | japanese favorite | meaning_may_change |
| fleurs_en_1620 | tiny | semantic_word_error | peanuts chillies | peanut chilies | meaning_may_change |
| fleurs_en_1620 | tiny | semantic_word_error | javanese | japanese | meaning_may_change |
| fleurs_en_1510 | tiny | semantic_word_error | u.n | un | meaning_may_change |
| fleurs_en_1578 | tiny | semantic_word_error | lakkha singh took | lock a sink to | meaning_may_change |
| fleurs_en_1578 | tiny | semantic_word_error | lead | lid and sink | meaning_may_change |
| fleurs_en_1578 | tiny | omission_or_insertion | singing |  | meaning_may_change |
| fleurs_en_1578 | tiny | semantic_word_error | bhajans | mushrooms | meaning_may_change |
| fleurs_en_1652 | tiny | semantic_word_error | moldova | wildova | meaning_may_change |

## Interpretation

- RAG is most defensible for `domain_term_error` and some `number_unit_error` cases.
- A text-only LLM can judge whether a correction is safe, but it cannot hear the audio.
- `disfluency_or_style_error` should be handled with cleaned WER, not with aggressive correction.
- Persistent date, number, and short-word errors are candidates for a stronger ASR baseline or audio-backed API comparison.
