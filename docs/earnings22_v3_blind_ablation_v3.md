# Earnings-22 Held-Out Ablation

This report compares raw ASR, deterministic RAG candidate validation, and RAG + LLM verification on the frozen held-out subset.
The deterministic correction variants use evidence gate v3: common-token entity rewrites are rejected and equivalent wording is counted as no_op.

| variant | model_name | num_rows | mean_wer | mean_term_recall | mean_term_f1 | candidate_count | applied_count | no_op_count | api_used_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| asr_only | base | 6 | 0.212099 | 0.833333 | 0.833333 | 4 | 0 | 0 | 0 |
| asr_only | tiny | 6 | 0.251941 | 0.833333 | 0.833333 | 2 | 0 | 0 | 0 |
| glossary_candidates_only | base | 6 | 0.212099 | 0.833333 | 0.833333 | 4 | 0 | 0 | 0 |
| glossary_candidates_only | tiny | 6 | 0.251941 | 0.833333 | 0.833333 | 2 | 0 | 0 | 0 |
| llm_without_rag_conservative | base | 6 | 0.212099 | 0.833333 | 0.833333 | 0 | 0 | 0 | 0 |
| llm_without_rag_conservative | tiny | 6 | 0.251941 | 0.833333 | 0.833333 | 0 | 0 | 0 | 0 |
| rag_evidence_gate_v3 | base | 6 | 0.212099 | 1.000000 | 0.833333 | 4 | 2 | 0 | 0 |
| rag_evidence_gate_v3 | tiny | 6 | 0.251941 | 0.833333 | 0.833333 | 2 | 0 | 0 | 0 |
| rag_llm_verifier_v3 | base | 6 | 0.212099 | 1.000000 | 0.833333 | 4 | 2 | 0 | 3 |
| rag_llm_verifier_v3 | tiny | 6 | 0.251941 | 0.833333 | 0.833333 | 2 | 0 | 0 | 2 |
