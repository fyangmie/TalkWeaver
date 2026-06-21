# Earnings-22 Held-Out Ablation

This report compares raw ASR, deterministic RAG candidate validation, and RAG + LLM verification on the frozen held-out subset.

| variant | model_name | num_rows | mean_wer | mean_term_recall | mean_term_f1 | candidate_count | applied_count | api_used_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| asr_only | base | 12 | 0.325876 | 0.833333 | 0.833333 | 7 | 0 | 0 |
| asr_only | tiny | 12 | 0.372716 | 0.750000 | 0.750000 | 7 | 0 | 0 |
| rag_llm_verifier | base | 12 | 0.326059 | 1.000000 | 0.805556 | 7 | 6 | 6 |
| rag_llm_verifier | tiny | 12 | 0.372899 | 0.916667 | 0.722222 | 7 | 7 | 7 |
| rag_validator_only | base | 12 | 0.326059 | 1.000000 | 0.788889 | 7 | 7 | 0 |
| rag_validator_only | tiny | 12 | 0.372899 | 0.916667 | 0.722222 | 7 | 7 | 0 |
