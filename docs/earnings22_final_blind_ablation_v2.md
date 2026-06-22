# Earnings-22 Held-Out Ablation

This report compares raw ASR, deterministic RAG candidate validation, and RAG + LLM verification on the frozen held-out subset.
The deterministic correction variants use the v2 evidence gate: common-token entity rewrites are rejected and equivalent wording is counted as no_op.

| variant | model_name | num_rows | mean_wer | mean_term_recall | mean_term_f1 | candidate_count | applied_count | no_op_count | api_used_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| asr_only | base | 12 | 0.186844 | 0.958333 | 0.972222 | 1 | 0 | 0 | 0 |
| asr_only | tiny | 12 | 0.221805 | 0.875000 | 0.888889 | 4 | 0 | 0 | 0 |
| glossary_candidates_only | base | 12 | 0.186844 | 0.958333 | 0.972222 | 1 | 0 | 0 | 0 |
| glossary_candidates_only | tiny | 12 | 0.221805 | 0.875000 | 0.888889 | 4 | 0 | 0 | 0 |
| llm_without_rag_conservative | base | 12 | 0.186844 | 0.958333 | 0.972222 | 0 | 0 | 0 | 0 |
| llm_without_rag_conservative | tiny | 12 | 0.221805 | 0.875000 | 0.888889 | 0 | 0 | 0 | 0 |
| rag_evidence_gate_v2 | base | 12 | 0.187018 | 0.958333 | 0.930556 | 1 | 1 | 0 | 0 |
| rag_evidence_gate_v2 | tiny | 12 | 0.221978 | 0.958333 | 0.930556 | 4 | 2 | 0 | 0 |
| rag_llm_verifier_v2 | base | 12 | 0.187018 | 0.958333 | 0.930556 | 1 | 1 | 1 | 1 |
| rag_llm_verifier_v2 | tiny | 12 | 0.221978 | 0.958333 | 0.930556 | 4 | 2 | 0 | 4 |
