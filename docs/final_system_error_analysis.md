# Final System Error Analysis

更新时间：2026-06-21

这份文档把最终论文要讲的错误类型固定下来。它不是新增 claim，而是把现有 CSV 中的结果整理成论文可解释的 failure modes。

## ASR Error Patterns

Primary artifacts:

- `experiments/results/asr_benchmark_summary_real.csv`
- `experiments/results/asr_error_patterns_real_combined_summary.csv`
- `experiments/results/asr_error_patterns_real_combined_model_summary.csv`

Observed patterns:

| Dataset / Track | Main Error Type | Evidence | Interpretation |
| --- | --- | --- | --- |
| AMI meeting | disfluency/style, omission/insertion, boundary-sensitive words | AMI `base` standard WER 0.3984, cleaned WER 0.3312; held-out `base` cleaned WER 0.2898 | Meeting references include fillers and low-energy speech; cleaned WER must be reported beside standard WER |
| AISHELL-4 meeting | high Mandarin CER | 60-clip AISHELL-4 subset: `base` CER 0.5369 and `small` CER 0.4818 vs 0.1133 on FLEURS Mandarin | Read-speech multilingual ASR does not predict meeting robustness |
| FLEURS read speech | semantic substitutions | FLEURS errors are mostly lexical/semantic substitutions in the audit summaries | Stronger ASR helps; meeting-specific logic is less relevant here |
| Earnings-22 | finance term and number/unit errors | tiny has more domain-term and number-unit errors than base in the combined audit | RAG should target finance terms and units, not general grammar |

Representative examples from the audit:

- AMI disfluency/style: reference `um`, hypothesis empty; use cleaned WER as diagnostic.
- AMI semantic/function-word error: short words such as `our` can disappear or change; RAG should not try to fix this.
- Earnings-22 term error: `double-digit` vs `double digit`; safe glossary handling can help.
- Earnings-22 number/unit error: context around `cents` or units needs stricter validation.

## Diarization and Overlap Errors

Primary artifacts:

- `experiments/results/pyannote_diarization_heldout_summary_real.csv`
- `experiments/results/pyannote_diarization_aishell4_60x20_summary_real.csv`
- `experiments/results/automatic_pyannote_workflow_aishell4_60x20_summary_real.csv`
- `experiments/results/speaker_overlap_baseline_real.csv`

Observed patterns:

| Track | Current Result | Main Risk | Paper Interpretation |
| --- | ---: | --- | --- |
| AMI held-out pyannote | DER 0.1060, JER 0.3072, overlap F1 0.4902 | overlap detection is much harder than speaker coverage | Good enough for evidence maps, not perfect diarization |
| AISHELL-4 pyannote subset | DER 0.3265, JER 0.7126, overlap F1 0.2619 on 29 multi-speaker scored clips | Mandarin meeting diarization and overlap detection remain difficult | Useful fixed-subset result, not full AISHELL-4 |
| AISHELL-4 evidence map | 29 maps, mean 6.76 anchors, 2.66 needs-review flags | Mandarin ASR and diarization uncertainty propagates into the map | Review flags are part of the method, not a failure to hide |
| Reference-assisted workflow | AMI overlap F1 about 0.980 | oracle speaker-time evidence inflates performance | Use only to validate plumbing and upper-bound logic |
| FLEURS single-speaker | pyannote may over-segment | single-speaker read speech is not a diarization target | Do not use FLEURS to claim meeting diarization quality |

The downstream implication is direct: if diarization splits one speaker into two or misses an overlap region, the ConversationMap may show wrong speaker attribution or miss review flags. This is why TalkWeaver records evidence and uncertainty rather than presenting corrected text as final truth.

## RAG and LLM Correction Errors

Primary artifacts:

- `experiments/results/earnings22_final_blind_ablation_v2_summary.csv`
- `experiments/results/earnings22_v3_blind_ablation_v3_summary.csv`
- `experiments/results/earnings22_v3_blind_rag_llm_v3_summary.csv`

Observed patterns:

| Variant | Result | Error Lesson |
| --- | --- | --- |
| RAG v2 on `tiny` | term F1 improves 0.8889 -> 0.9306, WER slightly worsens | RAG can recover terms for weak ASR but may not improve WER |
| RAG v2 on `base` | term F1 drops 0.9722 -> 0.9306 | strong ASR can be harmed by false-positive correction |
| RAG v3 on `base` | term recall improves 0.8333 -> 1.0000, WER unchanged | stricter evidence gates can improve term coverage without transcript drift |
| RAG v3 on `tiny` | no accepted corrections, one rejected/needs-review path | conservative gating prefers false negatives over unsafe edits |
| `glossary_candidates_only` | same WER and term scores as ASR-only | retrieval alone is evidence, not correction | Need a gated correction step to change output safely |
| `llm_without_rag_conservative` | same WER and term scores as ASR-only | no retrieved candidate means no supported term edit | LLM should not invent domain terms without evidence |

The paper claim should be: RAG is not a general meeting summarizer and not a guaranteed WER improver. It is a narrow, auditable domain-term recovery module whose safety depends on predefined term sources, local context, and rejection paths.

## Interruption Labeling Errors

Primary artifacts:

- `data/reference/public/english_meeting_heldout/interruption_label_candidates.csv`
- `experiments/results/interruption_label_summary_heldout_real.csv`

Current status:

- 10 generated candidate windows were reviewed.
- 10/10 were labeled as real interruption events.
- Candidate precision is 1.000.
- Recall and F1 are blank because non-candidate windows were not exhaustively labeled.
- Human review confirmed event existence, not independent speaker identity by voice.

Paper implication: the interruption detector is promising as a candidate generator, but not yet a complete interruption recognizer.

## Mobile and Runtime Errors

Primary artifacts:

- `experiments/results/v1/mobile_asr.csv`
- `experiments/results/v1/whisper_cpp_mobile_level1_summary.csv`
- `docs/mobile_asr_tradeoff.md`

Current status:

- `mobile_asr.csv` has 100 local CPU proxy rows with `claim_level=mobile_style_proxy`.
- whisper.cpp has 76 local-machine rows on the earlier 38-clip formal subset.
- No phone-device measurement has been run.
- whisper.cpp multilingual rows are weak under the current command path, especially French and Mandarin FLEURS.

Paper implication: model-size trade-off is real, but true phone latency, memory, battery, and packaged-runtime behavior remain future work.

## Final Error-Analysis Conclusion

The strongest final story is not "TalkWeaver fixes all errors." The defensible story is:

1. ASR errors are different across read speech, meetings, Mandarin meetings, and finance calls.
2. Diarization and overlap errors are visible and measurable, but still propagate into downstream maps.
3. RAG can help domain terms only when constrained; otherwise it creates false positives.
4. Human-reviewed interruption candidates are useful, but recall needs full timeline labels.
5. Mobile results are useful engineering diagnostics but not final phone deployment evidence.
