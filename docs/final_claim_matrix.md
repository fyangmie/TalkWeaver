# Final Claim Matrix

更新时间：2026-06-21

这份表是最终论文和视频的 claim 边界。写报告时优先引用这里的表述，避免把 small-subset、proxy 或 diagnostic 结果说成 full benchmark。

## Claim-Level Definitions

| Level | Meaning | Can Say | Cannot Say |
| --- | --- | --- | --- |
| Mock/demo | Deterministic fixture validates pipeline wiring | The pipeline, UI, and metrics run end to end | Any real model performance |
| Diagnostic real | Public data used to expose failure modes or tune design | The failure mode exists on these clips | Full-corpus performance or statistical significance |
| Held-out real | Frozen public subset after method choices are fixed | Small-subset result on this frozen set | State-of-the-art or broad generalization |
| Proxy | Local engineering approximation | Speed/accuracy trade-off direction on this machine | True phone or production deployment result |

## Final Result Matrix

| Track | Artifact | Claim Level | Final Claim | Required Caveat |
| --- | --- | --- | --- | --- |
| Formal ASR baseline | `experiments/results/asr_benchmark_summary_real.csv` | Diagnostic real | `base` is more accurate than `tiny`; meeting speech is harder than read FLEURS speech | 50 clips only; not full FLEURS, AMI, or AISHELL-4 |
| AMI held-out ASR | `experiments/results/asr_benchmark_english_meeting_heldout_summary_real.csv` | Held-out real | `small` improves AMI WER but is slower than `base`/`tiny` | 24 clips from four AMI recordings |
| Mandarin meeting ASR | `experiments/results/asr_benchmark_aishell4_60x20_summary_real.csv` | Held-out real | AISHELL-4 meeting speech is substantially harder than FLEURS Mandarin read speech; `small` has the best CER among tiny/base/small | 60 AISHELL-4 clips from 20 recordings; not full AISHELL-4 |
| Automatic diarization | `experiments/results/pyannote_diarization_heldout_summary_real.csv` | Held-out real | pyannote gives measurable DER/JER and overlap F1 on AMI held-out | 24 AMI clips; not full AMI corpus |
| Mandarin diarization | `experiments/results/pyannote_diarization_aishell4_60x20_summary_real.csv` | Held-out real | pyannote gives measurable Mandarin meeting DER/JER, but overlap F1 remains low | 29 multi-speaker scored clips from the 60-clip AISHELL-4 subset; not full AISHELL-4 |
| Workflow ablation | `experiments/results/workflow_ablation_summary_real.csv` | Diagnostic real | TalkWeaver adds speaker/time anchors, review flags, overlap evidence, and correction audits without forcing edits | Reference-assisted speaker-time is oracle evidence |
| Automatic evidence map | `experiments/results/automatic_pyannote_workflow_heldout_summary_real.csv` | Held-out real | Fully automatic AMI maps now contain speaker-labeled anchors, overlap anchors, events, and review flags | Evidence-map quality depends on pyannote output |
| Mandarin evidence map | `experiments/results/automatic_pyannote_workflow_aishell4_60x20_summary_real.csv` | Held-out real | AISHELL-4 evidence map rows show the same automatic ConversationMap workflow works on Mandarin meeting clips | 29 clips with pyannote turns from the 60-clip subset; not full AISHELL-4 |
| RAG v2 | `experiments/results/earnings22_final_blind_ablation_v2_summary.csv` | Held-out real | RAG can improve weak-ASR term F1, but can harm strong-ASR term F1 | Shows why safety gating is needed |
| RAG v3 | `experiments/results/earnings22_v3_blind_ablation_v3_summary.csv` | Held-out real | v3 is conservative: no WER change, base term recall improves from 0.8333 to 1.0000 across `asr_only`, `glossary_candidates_only`, no-RAG conservative, gate-only, and RAG+LLM variants | Claim is safe term recovery, not WER improvement |
| Interruption labels | `experiments/results/interruption_label_summary_heldout_real.csv` | Human-reviewed diagnostic | 10/10 reviewed candidates contain interruption behavior | Candidate precision only; no recall/F1 |
| Mobile proxy | `experiments/results/v1/mobile_asr.csv` | Proxy | local CPU int8 shows model-size trade-off | Not phone, not whisper.cpp |
| whisper.cpp Level 1 | `experiments/results/v1/whisper_cpp_mobile_level1_summary.csv` | Local-machine benchmark | local whisper.cpp tiny/base runs completed on earlier 38-clip subset | Not true phone; not rerun on AISHELL-4 |

## Paper-Safe One-Sentence Claims

- TalkWeaver turns ASR and diarization outputs into an auditable conversation map with temporal anchors, overlap flags, retrieved terms, and correction provenance.
- The real public-data ASR results show that meeting speech is harder than read speech, especially on AMI and AISHELL-4 meeting excerpts.
- pyannote diarization can be evaluated with standard DER/JER on the AMI held-out subset, and its errors directly affect downstream evidence maps.
- The 60 AISHELL-4 fixed subset strengthens the Mandarin meeting evidence, but still does not justify full-corpus AISHELL-4 claims.
- AISHELL-4 evidence map rows show TalkWeaver's main automatic pipeline can run on Mandarin meeting clips, not only English AMI.
- RAG is useful only as a constrained domain-term recovery module; unrestricted retrieval or LLM correction can introduce false positives.
- Overlap-aware correction is a safety mechanism: uncertain overlapping regions should be flagged and audited rather than silently rewritten.
- The current mobile results are local CPU/proxy and local-machine whisper.cpp evidence, not true phone-device measurements.

## Claims To Avoid

- Do not claim state-of-the-art ASR, diarization, or RAG correction.
- Do not claim full FLEURS, full AMI, full AISHELL-4, or full Earnings-22 benchmark results.
- Do not claim interruption recall or F1 from the 10 reviewed candidates.
- Do not claim full AISHELL-4 or Mandarin interruption performance from the current AISHELL-4 subset.
- Do not describe mobile proxy rows as phone-device or production deployment results.
