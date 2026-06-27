# TalkWeaver Paper v2 Claim Audit

This audit lists the quantitative claims reported in `paper_v2/main.tex` and `paper_v2/appendix.tex`. The paper draft does not report mock/demo metrics as scientific results.

## Claim-Level Legend

| Level | Meaning |
| --- | --- |
| Real-public diagnostic | Public audio subset used to expose failure modes; not full-corpus benchmarking. |
| Held-out real | Frozen public subset after method choices; still small-subset evidence. |
| Controlled | Authored text fixture or deterministic stress test; not real-audio generalization. |
| Controlled held-out | Authored held-out proposal set used for leakage/generalization audit. |
| Controlled/pilot | Mixed controlled and pilot labels; not a real-audio claim. |
| Proxy/local-machine | Runtime or deployment proxy measured locally; not true phone deployment. |

## Abstract and Core Claims

| Claim in draft | Value(s) | Source path | Level | Notes |
| --- | ---: | --- | --- | --- |
| Conservative RAG improves base-model Earnings-22 term recall without WER change on v3 blind subset | term recall `0.833333 -> 1.000000`; WER `0.212099 -> 0.212099` | `experiments/results/earnings22_v3_blind_ablation_v3_summary.csv` | Held-out real | 6-file Earnings-22 subset; claim is term recall only, not WER improvement. |
| EvidenceGate learned gates remain unsafe on independent held-out evidence | best strict macro F1 `0.325056`; needs-review recall `0.033333`; unsafe accept `0.166667` | `experiments/results/evidence_gate/evidence_gate_validation_metrics.csv`; `docs/evidence_gate.md` | Controlled held-out | Used as negative result and limitation. |

## Dataset and Artifact Size Claims

| Claim in draft | Value(s) | Source path | Level | Notes |
| --- | ---: | --- | --- | --- |
| Formal manifest size | `50` clips | `data/manifests/formal_eval_real.csv`; `docs/PAPER_HANDOFF_FINAL.md` | Real-public diagnostic | Combined FLEURS/AMI/AISHELL-4 manifest. |
| FLEURS formal subset size | `30` clips inside formal manifest | `docs/PAPER_HANDOFF_FINAL.md`; `experiments/results/asr_benchmark_summary_real.csv` | Real-public diagnostic | 10 en, 10 fr, 10 zh-CN rows reflected in ASR summary. |
| AMI formal subset size | `8` clips | `experiments/results/asr_benchmark_summary_real.csv` | Real-public diagnostic | AMI formal row. |
| AMI held-out size | `24` clips | `data/manifests/english_meeting_heldout_real.csv`; `experiments/results/asr_benchmark_english_meeting_heldout_summary_real.csv` | Held-out real | Four AMI recordings according to final claim matrix. |
| AISHELL-4 60x20 size | `60` clips; `29` diarization/map scored clips | `data/manifests/aishell4_benchmark_60x20.csv`; `experiments/results/pyannote_diarization_aishell4_60x20_summary_real.csv`; `experiments/results/automatic_pyannote_workflow_aishell4_60x20_summary_real.csv` | Held-out real | Not full AISHELL-4. |
| Earnings-22 v2/v3 subset sizes | `12` files v2; `6` files v3 | `data/manifests/earnings22_final_blind_12x180.csv`; `data/manifests/earnings22_v3_blind_6x180.csv` | Held-out real | Domain-term recovery only. |
| Controlled fixture sizes | `175` term rows; `80` overlap rows | `experiments/results/term_rescue_summary_controlled.csv`; `experiments/results/overlap_safety_summary_controlled.csv` | Controlled | Text fixtures, not audio metrics. |
| EvidenceGate independent held-out size | `90` proposals | `docs/evidence_gate.md`; `experiments/results/evidence_gate/evidence_gate_validation_metrics.csv` | Controlled held-out | Authored correction proposals. |
| whisper.cpp Level 1 rows | `76` rows | `experiments/results/v1/whisper_cpp_mobile_level1_summary.csv` | Proxy/local-machine | Not true phone device. |

## RQ1 ASR Claims

| Claim in draft | Value(s) | Source path | Level | Notes |
| --- | ---: | --- | --- | --- |
| FLEURS zh-CN base CER | `0.113336`; RTF `0.060634`; `10` clips | `experiments/results/asr_benchmark_summary_real.csv` | Real-public diagnostic | Read-speech contrast. |
| AISHELL-4 formal base CER | `0.609966`; RTF `0.049883`; `12` clips | `experiments/results/asr_benchmark_summary_real.csv` | Real-public diagnostic | Mandarin meeting sanity subset. |
| AISHELL-4 60x20 base CER | `0.536940`; RTF `0.071076`; `60` clips | `experiments/results/asr_benchmark_aishell4_60x20_summary_real.csv` | Held-out real | 60 clips, not full AISHELL-4. |
| AISHELL-4 60x20 small CER | `0.481842`; RTF `0.136653`; `60` clips | `experiments/results/asr_benchmark_aishell4_60x20_summary_real.csv` | Held-out real | Best CER among tiny/base/small in this subset. |
| AMI formal base WER | `0.398364`; cleaned `0.331216`; RTF `0.054548`; `8` clips | `experiments/results/asr_benchmark_summary_real.csv` | Real-public diagnostic | Meeting-speech contrast. |
| AMI held-out base WER | `0.349338`; cleaned `0.289807`; RTF `0.071662`; `24` clips | `experiments/results/asr_benchmark_english_meeting_heldout_summary_real.csv` | Held-out real | Held-out AMI subset. |
| AMI held-out small WER | `0.298610`; cleaned `0.233636`; RTF `0.176200`; `24` clips | `experiments/results/asr_benchmark_english_meeting_heldout_small_summary_real.csv` | Held-out real | More accurate but slower than base/tiny. |

## RQ1/RQ2 Diarization and Evidence-Map Claims

| Claim in draft | Value(s) | Source path | Level | Notes |
| --- | ---: | --- | --- | --- |
| AMI held-out pyannote DER/JER/overlap F1/RTF | DER `0.106035`; JER `0.307202`; overlap F1 `0.490214`; RTF `0.692477`; `24` clips | `experiments/results/pyannote_diarization_heldout_summary_real.csv` | Held-out real | Standard DER/JER result, not full AMI. |
| AISHELL-4 pyannote DER/JER/overlap F1/RTF | DER `0.326501`; JER `0.712577`; overlap F1 `0.261905`; RTF `0.504550`; `29` clips | `experiments/results/pyannote_diarization_aishell4_60x20_summary_real.csv` | Held-out real | 29 scored multi-speaker clips from 60x20 subset. |
| AMI automatic evidence map means | pyannote turns `4.666667`; anchors `4.75`; speaker anchors `4.25`; overlap anchors `1.791667`; events `2.041667`; needs review `2.291667`; ASR error `0.349338`; `24` clips | `experiments/results/automatic_pyannote_workflow_heldout_summary_real.csv` | Held-out real | Fully automatic map from fixed ASR and pyannote turns. |
| AISHELL-4 automatic evidence map means | pyannote turns `5.655172`; anchors `6.758621`; speaker anchors `4.482759`; overlap anchors `0.379310`; events `0.241379`; needs review `2.655172`; ASR error `0.586883`; `29` clips | `experiments/results/automatic_pyannote_workflow_aishell4_60x20_summary_real.csv` | Held-out real | 29 clips with pyannote turns; not full AISHELL-4. |
| Human-reviewed interruption candidates | total `10`; reviewed `10`; interruptions `10`; precision `1.0`; recall/F1 unavailable | `experiments/results/interruption_label_summary_heldout_real.csv` | Human-reviewed diagnostic | Candidate precision only; no recall or F1. |

## RQ3 RAG Term-Recovery Claims

| Claim in draft | Value(s) | Source path | Level | Notes |
| --- | ---: | --- | --- | --- |
| Earnings-22 v3 base ASR-only result | rows `6`; WER `0.212099`; term recall `0.833333`; term F1 `0.833333` | `experiments/results/earnings22_v3_blind_ablation_v3_summary.csv` | Held-out real | Baseline for v3. |
| Earnings-22 v3 base RAG EvidenceGate result | rows `6`; WER `0.212099`; term recall `1.000000`; term F1 `0.833333`; applied `2`; rejected `1`; needs review `1` | `experiments/results/earnings22_v3_blind_ablation_v3_summary.csv` | Held-out real | Conservative term-recall gain, WER unchanged. |
| Earnings-22 v3 base RAG+LLM verifier result | rows `6`; WER `0.212099`; term recall `1.000000`; term F1 `0.833333`; applied `2`; needs review `2`; API rows `3` | `experiments/results/earnings22_v3_blind_ablation_v3_summary.csv` | Held-out real | Same aggregate WER and recall as gate-only v3. |
| Earnings-22 v3 tiny unchanged result | rows `6`; WER `0.251941`; term recall `0.833333`; term F1 `0.833333` for ASR-only and RAG variants | `experiments/results/earnings22_v3_blind_ablation_v3_summary.csv` | Held-out real | No v3 gain for tiny. |
| Earnings-22 v2 tiny ASR-only result | rows `12`; WER `0.221805`; term recall `0.875000`; term F1 `0.888889` | `experiments/results/earnings22_final_blind_ablation_v2_summary.csv` | Held-out real | Baseline for v2 tiny. |
| Earnings-22 v2 tiny RAG+LLM result | rows `12`; WER `0.221978`; term recall `0.958333`; term F1 `0.930556`; applied `2`; rejected `2`; API rows `4` | `experiments/results/earnings22_final_blind_ablation_v2_summary.csv` | Held-out real | Helps tiny term F1 but slightly worsens WER. |
| Earnings-22 v2 base ASR-only result | rows `12`; WER `0.186844`; term recall `0.958333`; term F1 `0.972222` | `experiments/results/earnings22_final_blind_ablation_v2_summary.csv` | Held-out real | Stronger baseline. |
| Earnings-22 v2 base RAG+LLM negative result | rows `12`; WER `0.187018`; term recall `0.958333`; term F1 `0.930556`; applied `1`; needs review `4`; no-op `1`; API rows `1` | `experiments/results/earnings22_final_blind_ablation_v2_summary.csv` | Held-out real | Lower base term F1; motivates safety gate. |

## RQ4 Safety, Leakage, and Controlled Claims

| Claim in draft | Value(s) | Source path | Level | Notes |
| --- | ---: | --- | --- | --- |
| Controlled overlap-aware high-overlap rule safety | cases `4`; safety pass `1.000`; forbidden changes `0` | `experiments/results/overlap_safety_summary_controlled.csv` | Controlled | Text fixture only. |
| Controlled overlap-aware high-overlap LLM safety | cases `4`; safety pass `1.000`; forbidden changes `0` | `experiments/results/overlap_safety_summary_controlled.csv` | Controlled | Text fixture only. |
| Controlled no-overlap rule high-overlap failure | cases `4`; safety pass `0.000`; forbidden changes `3`; overcorrection rate `0.5` | `experiments/results/overlap_safety_summary_controlled.csv` | Controlled | Supports overlap-aware gating motivation. |
| Controlled no-overlap LLM high-overlap result | cases `4`; safety pass `0.500`; forbidden changes `0` | `experiments/results/overlap_safety_summary_controlled.csv` | Controlled | Text fixture only. |
| EvidenceGate feature leakage audit | total fields `41`; allowed `19`; risky reference-derived `4`; direct label proxy `14`; final audit outcome `4` | `docs/evidence_gate.md`; `experiments/results/evidence_gate/evidence_gate_feature_leakage_audit.csv` | Controlled audit | Reported as leakage risk. |
| EvidenceGate audit-aware grouped result | macro F1 `1.000`; false accept `0.000`; unsafe accept `0.000`; review recall `1.000`; reject recall `1.000` | `experiments/results/evidence_gate/evidence_gate_validation_metrics.csv`; `docs/evidence_gate.md` | Controlled policy-distillation sanity check | Not real generalization. |
| EvidenceGate evidence-only grouped result | macro F1 `0.975629`; false accept `0.000`; unsafe accept `0.000`; review recall `0.961538`; reject recall `1.000` | `experiments/results/evidence_gate/evidence_gate_validation_metrics.csv` | Controlled validation | Still optimistic grouped validation. |
| EvidenceGate risk-only grouped result | macro F1 `0.924396`; false accept `0.133333`; unsafe accept `0.000`; review recall `0.769231`; reject recall `0.947368` | `experiments/results/evidence_gate/evidence_gate_validation_metrics.csv` | Controlled validation | Strict feature set but same controlled source structure. |
| EvidenceGate evidence-only independent held-out RF | macro F1 `0.325056`; false accept `0.216667`; unsafe accept `0.166667`; review recall `0.033333`; reject recall `0.833333` | `experiments/results/evidence_gate/evidence_gate_validation_metrics.csv`; `docs/evidence_gate.md` | Controlled held-out | Best strict macro F1 but weak generalization. |
| EvidenceGate evidence-only independent held-out LR | macro F1 `0.316340`; false accept `0.100000`; unsafe accept `0.066667`; review recall `0.000000`; reject recall `0.933333` | `experiments/results/evidence_gate/evidence_gate_validation_metrics.csv`; `docs/evidence_gate.md` | Controlled held-out | Safer unsafe-accept rate but no needs-review recall. |
| Binary safe-to-apply always-apply baseline | examples `327`; macro F1 `0.196560`; unsafe apply `1.000`; coverage `1.000` | `experiments/results/binary_safe_apply/binary_safe_apply_summary.csv` | Controlled/pilot | Stress-test baseline. |
| Binary safe-to-apply retrieval-only row | examples `327`; macro F1 `0.815751`; unsafe apply `0.198381`; coverage `0.385321` | `experiments/results/binary_safe_apply/binary_safe_apply_summary.csv` | Controlled/pilot | Not real-audio generalization. |
| Binary safe-to-apply overlap-unaware row | examples `327`; macro F1 `0.884304`; unsafe apply `0.076923`; coverage `0.272171` | `experiments/results/binary_safe_apply/binary_safe_apply_summary.csv` | Controlled/pilot | Slightly higher macro F1 than Binary EccoGate, worse unsafe apply. |
| Binary EccoGate row | examples `327`; macro F1 `0.883167`; unsafe apply `0.052632`; false block `0.187500`; coverage `0.238532` | `experiments/results/binary_safe_apply/binary_safe_apply_summary.csv` | Controlled/pilot | Safer than LLM self-judge in this benchmark. |
| LLM self-judge with evidence row | examples `327`; macro F1 `0.596069`; unsafe apply `0.323887`; false block `0.425000`; coverage `0.385321` | `experiments/results/binary_safe_apply/binary_safe_apply_summary.csv` | Controlled/pilot | Shows evidence prompt alone is not enough. |

## Appendix Controlled Term-Rescue Rows

| Claim in draft | Value(s) | Source path | Level | Notes |
| --- | ---: | --- | --- | --- |
| No-retrieval English easy term F1 | cases `7`; term F1 `0.000`; error after `0.217687`; review `0` | `experiments/results/term_rescue_summary_controlled.csv` | Controlled | Text fixture. |
| Fused English hard term F1 | cases `5`; term F1 `1.000`; error after `0.351429`; review `0` | `experiments/results/term_rescue_summary_controlled.csv` | Controlled | Text fixture. |
| Fused+rule English hard term F1 | cases `5`; term F1 `1.000`; error after `0.000`; review `0` | `experiments/results/term_rescue_summary_controlled.csv` | Controlled | Text fixture. |
| Fused+LLM English hard term F1 | cases `5`; term F1 `1.000`; error after `0.222857`; review `2` | `experiments/results/term_rescue_summary_controlled.csv` | Controlled | Text fixture. |
| Fused+rule negative-control review behavior | cases `4`; term F1 `1.000`; error after `0.000`; review `4` | `experiments/results/term_rescue_summary_controlled.csv` | Controlled | Safety behavior, not public-audio claim. |

## Appendix Mobile/Runtime Claims

| Claim in draft | Value(s) | Source path | Level | Notes |
| --- | ---: | --- | --- | --- |
| whisper.cpp AMI tiny row | rows `8`; error `0.382383`; cleaned `0.313909`; RTF `0.030548` | `experiments/results/v1/whisper_cpp_mobile_level1_summary.csv` | Proxy/local-machine | Not phone-device measurement. |
| whisper.cpp AMI base row | rows `8`; error `0.351123`; cleaned `0.280012`; RTF `0.056347` | `experiments/results/v1/whisper_cpp_mobile_level1_summary.csv` | Proxy/local-machine | Not phone-device measurement. |
| whisper.cpp FLEURS en tiny row | rows `10`; error `0.185512`; cleaned `0.185512`; RTF `0.071982` | `experiments/results/v1/whisper_cpp_mobile_level1_summary.csv` | Proxy/local-machine | Not phone-device measurement. |
| whisper.cpp FLEURS en base row | rows `10`; error `0.133832`; cleaned `0.133832`; RTF `0.139889` | `experiments/results/v1/whisper_cpp_mobile_level1_summary.csv` | Proxy/local-machine | Not phone-device measurement. |

## Non-Quantitative and TODO Evidence

| Item | Status | Why it matters |
| --- | --- | --- |
| Dataset citations for FLEURS, AMI, AISHELL-4, Earnings-22 | TODO citation keys in `paper_v2/references.bib` | Full bibliographic details were not verified from local materials. |
| pyannote and Whisper/faster-whisper citations | TODO citation keys in `paper_v2/references.bib` | Result files identify models/tools, but complete bibliographic entries need verification. |
| Full-corpus benchmark claims | Not made | Existing evidence is small-subset or held-out subset only. |
| State-of-the-art claims | Not made | No SOTA comparison table exists in repository evidence. |
| Interruption recall/F1 | Not made | `experiments/results/interruption_label_summary_heldout_real.csv` leaves recall and F1 blank. |
| True phone-device mobile benchmark | Not made | Existing mobile evidence is proxy/local-machine. |
