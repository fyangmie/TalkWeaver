# TalkWeaver Paper v2 Figure/Table Manifest

This manifest lists every figure and table used in `paper_v2/main.tex` and `paper_v2/appendix.tex`. Captions are factual and point to existing repository artifacts. Placement is intentionally split between main paper and appendix so that the main paper supports RQ1-RQ4 without overclaiming secondary or controlled results.

## Main-Paper Figures

| ID | Placement | Source file | RQ | Factual caption / role |
| --- | --- | --- | --- | --- |
| Fig. 1 `fig:architecture` | Main | `assets/architecture.png` copied to `paper_v2/figures/architecture.png` | Method / RQ2 | TalkWeaver architecture and evidence flow. Used as a system diagram, not as a quantitative claim. |
| Fig. 2 `fig:asr_error` | Main | `assets/result_charts/asr_error_by_dataset.png` copied to `paper_v2/figures/asr_error_by_dataset.png` | RQ1 | ASR error by dataset from existing public-subset CSVs. Supports meeting-vs-read-speech difficulty, not full-corpus benchmarking. |
| Fig. 3 `fig:workflow_completeness` | Main | `assets/result_charts/workflow_ablation_completeness.png` copied to `paper_v2/figures/workflow_ablation_completeness.png` | RQ2 | Evidence-map structure and workflow-completeness visualization. Some underlying workflow variants are reference-assisted, so this is not automatic accuracy. |

## Main-Paper Tables

| ID | Placement | Source file(s) | RQ | Factual caption / role |
| --- | --- | --- | --- | --- |
| Table 1 `tab:data` | Main | `docs/PAPER_HANDOFF_FINAL.md`, `docs/final_claim_matrix.md`, dataset manifests under `data/manifests/`, result CSVs listed in table | All | Dataset/artifact boundary table distinguishing real public, controlled, held-out, proxy, and mock/demo categories. |
| Table 2 `tab:asr` | Main | `experiments/results/asr_benchmark_summary_real.csv`, `experiments/results/asr_benchmark_english_meeting_heldout_summary_real.csv`, `experiments/results/asr_benchmark_english_meeting_heldout_small_summary_real.csv`, `experiments/results/asr_benchmark_aishell4_60x20_summary_real.csv` | RQ1 | ASR error and RTF rows showing read-speech vs meeting-speech difficulty and AMI/AISHELL model-size trade-offs. |
| Table 3 `tab:diarization` | Main | `experiments/results/pyannote_diarization_heldout_summary_real.csv`, `experiments/results/pyannote_diarization_aishell4_60x20_summary_real.csv` | RQ1/RQ2 | DER, JER, overlap F1, and RTF on AMI held-out and AISHELL-4 60x20 scored clips. |
| Table 4 `tab:map` | Main | `experiments/results/automatic_pyannote_workflow_heldout_summary_real.csv`, `experiments/results/automatic_pyannote_workflow_aishell4_60x20_summary_real.csv` | RQ2 | Automatic conversation-map statistics: anchors, speaker-labeled anchors, overlap anchors, events, and needs-review flags. |
| Table 5 `tab:rag` | Main | `experiments/results/earnings22_v3_blind_ablation_v3_summary.csv`, `experiments/results/earnings22_final_blind_ablation_v2_summary.csv` | RQ3 | Earnings-22 RAG v3 positive term-recall result and RAG v2 negative/false-positive-risk result. |
| Table 6 `tab:safety` | Main | `experiments/results/overlap_safety_summary_controlled.csv`, `docs/evidence_gate.md`, `experiments/results/evidence_gate/evidence_gate_validation_metrics.csv`, `experiments/results/binary_safe_apply/binary_safe_apply_summary.csv` | RQ4 | Controlled safety, leakage audit, independent held-out, and binary safe-to-apply summary. |

## Appendix Figures

| ID | Placement | Source file | RQ | Factual caption / role |
| --- | --- | --- | --- | --- |
| Fig. A1 `fig:app_term_f1` | Appendix | `assets/result_charts/term_rescue_f1_by_variant.png` copied to `paper_v2/figures/term_rescue_f1_by_variant.png` | RQ3/RQ4 | Controlled text-fixture term-recovery F1 by variant. Mechanism testing only. |
| Fig. A2 `fig:app_overlap_safety` | Appendix | `assets/result_charts/overlap_safety_pass_rate.png` copied to `paper_v2/figures/overlap_safety_pass_rate.png` | RQ4 | Controlled overlap-safety pass rate. Not real-audio overlap performance. |
| Fig. A3 `fig:app_evidence_leakage` | Appendix | `assets/result_charts/evidence_gate_feature_leakage_audit.png` copied to `paper_v2/figures/evidence_gate_feature_leakage_audit.png` | RQ4 | EvidenceGate feature leakage audit: 41 features/fields categorized by leakage risk. |
| Fig. A4 `fig:app_evidence_confusion` | Appendix | `assets/result_charts/evidence_gate_heldout_confusion_matrix.png` copied to `paper_v2/figures/evidence_gate_heldout_confusion_matrix.png` | RQ4 | Independent held-out EvidenceGate confusion matrix showing weak needs-review generalization. |
| Fig. A5 `fig:app_binary_f1` | Appendix | `assets/result_charts/binary_safe_apply_macro_f1.png` copied to `paper_v2/figures/binary_safe_apply_macro_f1.png` | RQ4 | Binary safe-to-apply macro F1 across controlled/pilot policies. |
| Fig. A6 `fig:app_rtf` | Appendix | `assets/result_charts/asr_rtf_by_model.png` copied to `paper_v2/figures/asr_rtf_by_model.png` | RQ1 / deployment | ASR real-time factor chart for local runtime trade-off discussion. |

## Appendix Tables

| ID | Placement | Source file(s) | RQ | Factual caption / role |
| --- | --- | --- | --- | --- |
| Table A1 `tab:app_term_rows` | Appendix | `experiments/results/term_rescue_summary_controlled.csv` | RQ3/RQ4 | Selected controlled term-rescue rows. Includes negative-control review behavior. |
| Table A2 `tab:app_overlap_rows` | Appendix | `experiments/results/overlap_safety_summary_controlled.csv` | RQ4 | Selected high-overlap controlled safety rows. |
| Table A3 `tab:app_evidence_gate` | Appendix | `experiments/results/evidence_gate/evidence_gate_validation_metrics.csv`, `docs/evidence_gate.md` | RQ4 | EvidenceGate grouped-test and independent-held-out rows. |
| Table A4 `tab:app_binary` | Appendix | `experiments/results/binary_safe_apply/binary_safe_apply_summary.csv` | RQ4 | Selected binary safe-to-apply benchmark rows. |
| Table A5 `tab:app_interruption` | Appendix | `experiments/results/interruption_label_summary_heldout_real.csv` | RQ2/RQ4 | Human-reviewed interruption candidate precision only; recall/F1 unavailable. |
| Table A6 `tab:app_mobile` | Appendix | `experiments/results/v1/whisper_cpp_mobile_level1_summary.csv` | Deployment limitation | Local-machine whisper.cpp Level 1 selected rows; not a true phone-device benchmark. |

## Not Used as Main Figures

The repository contains additional charts in `assets/result_charts/` such as hallucination, WDER, latency, pilot, and category-failure plots. They are not included in the v2 main paper because their underlying evidence is either mock/demo, older diagnostic, controlled/pilot only, or less directly tied to RQ1-RQ4. They can be added later only with a corresponding claim-audit entry.
