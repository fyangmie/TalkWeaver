# Earnings-22 Held-Out Ablation

This report compares raw ASR, deterministic RAG candidate validation, and RAG + LLM verification on the frozen held-out subset.
The deterministic correction variants use the v2 evidence gate: common-token entity rewrites are rejected and equivalent wording is counted as no_op.

| variant | model_name | num_rows | mean_wer | mean_term_recall | mean_term_f1 | candidate_count | applied_count | no_op_count | api_used_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| asr_only | base | 12 | 0.325876 | 0.833333 | 0.833333 | 7 | 0 | 0 | 0 |
| asr_only | tiny | 12 | 0.372716 | 0.750000 | 0.750000 | 7 | 0 | 0 | 0 |
| glossary_candidates_only | base | 12 | 0.325876 | 0.833333 | 0.833333 | 7 | 0 | 0 | 0 |
| glossary_candidates_only | tiny | 12 | 0.372716 | 0.750000 | 0.750000 | 7 | 0 | 0 | 0 |
| llm_without_rag_conservative | base | 12 | 0.325876 | 0.833333 | 0.833333 | 0 | 0 | 0 | 0 |
| llm_without_rag_conservative | tiny | 12 | 0.372716 | 0.750000 | 0.750000 | 0 | 0 | 0 | 0 |
| rag_evidence_gate_v2 | base | 12 | 0.325876 | 1.000000 | 1.000000 | 7 | 2 | 1 | 0 |
| rag_evidence_gate_v2 | tiny | 12 | 0.372716 | 0.916667 | 0.916667 | 7 | 2 | 1 | 0 |
| rag_llm_verifier_v2 | base | 12 | 0.325876 | 1.000000 | 1.000000 | 7 | 2 | 1 | 6 |
| rag_llm_verifier_v2 | tiny | 12 | 0.372716 | 0.916667 | 0.916667 | 7 | 2 | 3 | 7 |
