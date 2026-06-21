# TalkWeaver 最终论文实验交接报告

更新时间：2026-06-21
分支现场：`feature/r2-heldout-dataset`
使用原则：本报告只整理仓库中已有代码、docs、CSV、图表、git log 和验证脚本信息，不新增实验结果，不重跑大实验，不修改模型逻辑。

## 1. 最终科研故事主线

### 1.1 项目真正研究的问题

TalkWeaver 研究的不是“如何做一个会议纪要系统”，而是：

> 在混乱多人会议中，ASR、说话人归属、重叠说话、打断、专有词误听和 LLM 修正都可能出错。系统能否把这些不确定因素组织成一个可审计的 conversation evidence map，而不是给用户一个看似流畅但无法验证的最终文本？

论文应该围绕 **evidence-grounded conversation map** 写，而不是围绕普通 summary/chatbot 写。

### 1.2 研究动机

仓库中的 real results 支持以下动机：

- **读句子 ASR 不能代表会议 ASR**：formal ASR 中 FLEURS Mandarin `base` CER 为 `0.113336`，而 AISHELL-4 12-clip meeting `base` CER 为 `0.609966`；在 AISHELL-4 60-clip 子集上 `base` CER 仍为 `0.536940`。
- **多人会议的说话人和重叠证据很关键**：AMI held-out pyannote DER 为 `0.106035`，但 overlap F1 只有 `0.490214`；AISHELL-4 Mandarin 子集 DER 为 `0.326501`，JER 为 `0.712577`，overlap F1 为 `0.261905`。
- **RAG 可以帮专有词，但也可能伤害强 ASR**：Earnings-22 RAG v2 对 `tiny` term F1 有提升，但对 `base` term F1 变差；v3 改为更保守的安全门控。
- **LLM 或规则修正不能静默应用**：controlled overlap safety、binary safe-to-apply、EvidenceGate 审计都显示，缺少证据的修正有 unsafe accept 风险。

### 1.3 核心 research questions

以 `docs/research_questions.md` 为准，论文可以收束成四个 RQ：

| RQ | 简写 | 当前证据强度 | 论文写法 |
| --- | --- | --- | --- |
| RQ1 | diarization-structured prompting / speaker-time map 是否提升 speaker-attributed transcript 可读性和一致性 | 中等。workflow 和 evidence-map 真实跑通，但不是人工 readability study | 写成系统设计和 evidence-map 可审计性，不要写成用户研究结论 |
| RQ2 | overlap-aware uncertainty 是否减少 unsupported correction | 中等偏强，但主要来自 controlled fixtures | 写成 safety mechanism，在真实 AMI/AISHELL 中通过 overlap flags 和 needs-review 暴露风险 |
| RQ3 | RAG glossary 是否减少 domain term ASR error | 中等。Earnings-22 v2/v3 有 real blind subset | 写成 constrained term recovery，不写成 general WER improvement |
| RQ4 | preprocessing 是否提升 noisy ASR | 弱。未找到正式 real A/B preprocessing result | 写成 pipeline 支持和 future work，不作为主结论 |

### 1.4 TalkWeaver 产品/UI 在论文中的角色

TalkWeaver UI 是 **research artifact / evidence viewer**：

- 展示 temporal anchors、speaker labels、overlap flags、retrieved terms、correction audit、needs-review。
- 用于解释模型输出和错误传播。
- 不是论文主要贡献里的“自动会议总结器”。
- 不应把 Streamlit dashboard 写成核心算法创新；它是把研究结果可视化、可审计化的界面。

推荐一句话定位：

> TalkWeaver is an evidence-grounded meeting detective interface that makes ASR, diarization, overlap, retrieval, and correction uncertainty inspectable.

## 2. 实验类型边界

| 类型 | 代表实验 | 可作为主结论吗 | 正确写法 | 不能写成 |
| --- | --- | --- | --- | --- |
| 主实验 | ASR real baseline、AMI/AISHELL pyannote DER/JER、automatic evidence map、Earnings-22 RAG v3 | 可以，但要加 small-subset caveat | public-data small subset results | full benchmark / SOTA |
| 真实公开数据验证 | FLEURS、AMI、AISHELL-4、Earnings-22 | 可以支持问题存在和方法可运行 | real public-data evidence | 大规模泛化 |
| controlled / curated / stress-test | term rescue fixtures、overlap safety fixtures、selective correction pilot、binary safe-to-apply、EvidenceGate | 只能支持机制和风险分析 | controlled safety validation | 真实世界性能 |
| mock/demo | `experiments/run_ablation.py --mock`、demo pipeline | 不能作为性能结论 | pipeline sanity check | real model result |
| 负结果 / 风险 | RAG v2 hurts `base` term F1；EvidenceGate independent heldout poor；LLM self-judge unsafe；mobile not true phone | 必须写 | limitations and honest analysis | 隐藏或包装成成功 |

## 3. 数据集总表

| 数据集 / 子集 | 来源 | 样本数 | 语言 | 真实音频 | Controlled | 用途 | 局限性 | 路径 |
| --- | --- | ---: | --- | --- | --- | --- | --- | --- |
| Formal real manifest | FLEURS + AMI + AISHELL-4 | 50 clips | en/fr/zh-CN | 是 | 否 | 总体 ASR baseline 和 workflow ablation | small subset，不是 full corpus | `data/manifests/formal_eval_real.csv` |
| FLEURS fallback | Google FLEURS | 30 clips | en/fr/zh-CN | 是 | 否 | multilingual read-speech ASR | read speech，不代表 meeting | `data/manifests/common_voice_multilingual_real.csv` |
| AMI formal meeting | AMI Meeting Corpus | 8 clips | en | 是 | 否 | English meeting overlap/speaker evidence | 一个 meeting 系列小切片 | `data/manifests/english_meeting_real.csv` |
| AMI held-out | AMI Meeting Corpus | 24 clips | en | 是 | 否 | held-out ASR、pyannote DER/JER、automatic evidence map | 4 个 AMI recordings，小规模 | `data/manifests/english_meeting_heldout_real.csv` |
| AISHELL-4 formal sanity | AISHELL-4 | 12 clips | zh-CN | 是 | 否 | formal 50 manifest 里的 Mandarin meeting sanity | 一个小子集，不能作为主 AISHELL claim | `data/manifests/mandarin_meeting_real.csv` |
| AISHELL-4 60x20 | AISHELL-4 test split | 60 clips | zh-CN | 是 | 否 | Mandarin meeting ASR、DER/JER、evidence map | 不是 full AISHELL-4；29 clips 有 multi-speaker pyannote score | `data/manifests/aishell4_benchmark_60x20.csv` |
| Earnings-22 final blind v2 | Earnings-22 | 12 files | en | 是 | 否 | finance term RAG v2 | finance-call domain，非 meeting overlap | `data/manifests/earnings22_final_blind_12x180.csv` |
| Earnings-22 v3 blind | Earnings-22 | 6 files | en | 是 | 否 | RAG v3 safety gate ablation | 小 blind subset，不证明 general WER gain | `data/manifests/earnings22_v3_blind_6x180.csv` |
| Term rescue controlled | Authored fixtures | 175 rows | en/fr/zh-CN | 否 | 是 | RAG/glossary and correction safety stress test | text fixture，不是音频结果 | `experiments/results/term_rescue_controlled.csv` |
| Overlap safety controlled | Authored fixtures | 80 rows | en | 否 | 是 | overlap-aware correction safety | text fixture，不是真实 overlap ASR | `experiments/results/overlap_safety_controlled.csv` |
| Selective correction pilot | Authored proposals | 72 rows | en/fr/zh-CN | 否 | 是 | early accept/reject/needs_review pilot | auto-labeled，非 human gold | `data/pilot/selective_correction_pilot.csv` |
| Binary safe-to-apply | Controlled + pilot proposals | 327 rows | en/fr/zh-CN | 否 | 是 | binary safety gate stress-test | mixed label sources，非 real generalization | `data/pilot/binary_safe_apply_benchmark.csv` |
| Binary independent heldout | Authored heldout proposals | 153 rows | 多语言 | 否 | 是 | safety heldout / leakage risk check | curated text proposals | `data/pilot/binary_safe_apply_independent_heldout.csv` |
| Binary R2 real-derived | Real-derived proposals | 34 rows | en/fr/zh-CN | 来自真实 ASR 文本 | 半 controlled | correction safety realism probe | reference-derived labels，不是独立人工标注 | `data/pilot/binary_safe_apply_r2_real_derived.csv` |
| EvidenceGate controlled dataset | Term + overlap safety augmented | 525 rows in docs | 多语言 | 否 | 是 | trained policy / leakage audit | policy-distillation，independent heldout poor | `docs/evidence_gate.md`, `experiments/results/evidence_gate/` |

## 4. 完整 pipeline 交接表

| Pipeline step | 主要代码 | 输入 | 输出 | 是否真实运行 | 是否主实验 |
| --- | --- | --- | --- | --- | --- |
| Audio preprocessing | `backend/preprocessing.py`, `scripts/run_stage.py` | raw audio path | mono/normalized audio, stage artifact | 支持真实运行 | 不是主实验结论；RQ4 未确认 |
| ASR | `backend/asr.py`, `experiments/run_asr_benchmark.py`, `experiments/summarize_asr_results.py` | audio manifest | ASR segments, hypothesis, WER/CER, RTF | 是，FLEURS/AMI/AISHELL/Earnings-22 | 是 |
| Prediction loading | `experiments/prediction_loader.py` | ASR prediction JSON | stable `PredictionRecord` | 是 | 工程支撑 |
| Diarization | `backend/diarization.py`, `experiments/run_pyannote_diarization_benchmark.py` | audio manifest, HF token/model | pyannote turns, DER/JER, overlap F1 | 是，AMI/AISHELL | 是 |
| Speaker-time alignment | `backend/alignment.py`, `backend/temporal_anchor.py`, `backend/reference_evidence.py` | ASR words/segments + diarization/reference turns | temporal anchors | 是 | 是 |
| Overlap/events | `backend/overlap.py`, `backend/events.py`, `scripts/generate_interruption_label_candidates.py` | speaker intervals / anchors | overlap flags, interruption candidates | 是，小规模 human review | 是，但 interruption only precision |
| RAG/term rescue | `backend/rag.py`, `backend/term_rescue.py`, `experiments/evaluate_rag_llm_correction.py` | ASR text + glossary | retrieved terms, candidate corrections | 是，Earnings-22 and fixtures | 是，作为辅助贡献 |
| Constrained/LLM correction | `backend/llm_correction.py`, `backend/constrained_correction.py`, `experiments/evaluate_earnings22_ablation.py` | raw text + candidates + evidence | corrected text, applied/rejected/needs_review/no_op | 是，DeepSeek rows in some experiments | 是，但 claim 是 safe term recovery |
| Audit/gate | `backend/evidence_gate.py`, `backend/eccogate.py`, `backend/eccogate_binary.py`, `experiments/evaluate_evidence_gate.py`, `experiments/run_binary_safe_apply_experiment.py` | proposed correction + evidence features | accept/reject/needs_review or safe_to_apply | 是，controlled | 不是主 real claim；是 safety analysis |
| ConversationMap | `backend/conversation_map.py`, `backend/workflow_variants.py`, `experiments/run_automatic_pyannote_workflow.py` | ASR + diarization + RAG/correction artifacts | evidence map rows and JSON maps | 是，AMI/AISHELL | 是 |
| UI / Detective app | `webapp/streamlit_app.py`, `webapp/app.py`, `webapp/detective_ui.py`, `webapp/components/*` | pipeline outputs / maps / metrics | review dashboard | 可运行，主要展示 | 论文 artifact，不是算法性能 |

## 5. 已完成实验总表

| 实验 | 目的 | 数据 | 方法 / variants | 指标 | 关键结果 | 路径 | 可以写进论文的结论 | 不能夸大 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Formal ASR baseline | 建立 real ASR baseline | 50 real clips | faster-whisper `tiny/base`, CPU int8 | WER/CER, RTF | AMI `base` WER `0.398364`, cleaned `0.331216`; FLEURS zh `base` CER `0.113336`; AISHELL-4 12-clip `base` CER `0.609966` | `experiments/results/asr_benchmark_summary_real.csv`, charts `asr_*` | meeting speech harder than read speech | 不是 full FLEURS/AMI/AISHELL |
| AMI held-out ASR | meeting-only held-out ASR | 24 AMI clips | `tiny/base/small` | WER, cleaned WER, RTF | `small` WER `0.298610`, cleaned `0.233636`, RTF `0.176200`; `base` cleaned `0.289807` | `experiments/results/asr_benchmark_english_meeting_heldout*_summary_real.csv` | larger model helps but slower | 仍是 24 clips |
| AISHELL-4 60x20 ASR | Mandarin meeting ASR | 60 AISHELL-4 clips | `tiny/base/small` | CER, RTF | `small` CER `0.481842`, `base` `0.536940`, `tiny` `0.648291` | `experiments/results/asr_benchmark_aishell4_60x20_summary_real.csv` | Mandarin meetings are hard; small is best among tested | 不是 full AISHELL-4 |
| AMI pyannote diarization | automatic diarization scoring | 24 AMI clips | pyannote community pipeline | DER/JER, overlap F1, RTF | DER `0.106035`, JER `0.307202`, overlap F1 `0.490214` | `experiments/results/pyannote_diarization_heldout_summary_real.csv` | pyannote usable but overlap remains hard | 不代表 full AMI |
| AISHELL-4 pyannote diarization | Mandarin diarization scoring | 29 multi-speaker AISHELL clips | pyannote | DER/JER, overlap F1, RTF | DER `0.326501`, JER `0.712577`, overlap F1 `0.261905` | `experiments/results/pyannote_diarization_aishell4_60x20_summary_real.csv` | Mandarin meeting diarization is harder | 不是 full AISHELL-4; no Mandarin interruption score |
| AMI automatic evidence map | full TalkWeaver automatic map | 24 AMI clips | base ASR + pyannote + ConversationMap | anchors, speaker anchors, events, needs_review | mean anchors `4.75`, overlap anchors `1.791667`, events `2.041667`, needs_review `2.291667` | `experiments/results/automatic_pyannote_workflow_heldout_summary_real.csv` | main pipeline works end-to-end on English meetings | quality depends on pyannote |
| AISHELL-4 evidence map | Mandarin full map closure | 29 AISHELL clips with turns | base ASR + pyannote + ConversationMap | anchors, events, needs_review | mean anchors `6.758621`, speaker anchors `4.482759`, needs_review `2.655172` | `experiments/results/automatic_pyannote_workflow_aishell4_60x20_summary_real.csv` | TalkWeaver runs on Mandarin meeting clips too | no full Mandarin workflow benchmark |
| Reference-assisted workflow ablation | show pipeline components and oracle speaker-time effect | 50 formal clips | asr_only, temporal_anchor_only, reference_speaker_time, overlap_aware, term_rescue, constrained, full_talkweaver | anchors, events, review flags, error rate | AMI full_talkweaver: anchors `4.5`, events `3.75`, needs_review `2.125`; note says reference speaker-time is oracle-assisted | `experiments/results/workflow_ablation_summary_real.csv`, `assets/result_charts/workflow_ablation_*.png` | evidence layer adds structure and audits | oracle/reference-assisted, not automatic accuracy |
| Earnings-22 RAG v2 | test RAG term correction before stricter gate | 12 Earnings-22 files | asr_only, glossary_only, no-RAG conservative, RAG gate v2, RAG+LLM v2 | WER, term recall/F1 | `tiny` term F1 `0.888889 -> 0.930556`; `base` term F1 `0.972222 -> 0.930556`; WER slightly worsens | `experiments/results/earnings22_final_blind_ablation_v2_summary.csv` | RAG can help weak ASR but can hurt strong ASR | 不是稳定 WER improvement |
| Earnings-22 RAG v3 | stricter safe term recovery | 6 Earnings-22 files | asr_only, `glossary_candidates_only`, no-RAG conservative, gate v3, RAG+LLM v3 | WER, term recall/F1, applied/rejected | `base` term recall `0.833333 -> 1.000000`, WER unchanged `0.212099`; `tiny` unchanged | `experiments/results/earnings22_v3_blind_ablation_v3_summary.csv`, `docs/earnings22_v3_blind_ablation_v3.md` | safe term recovery, no transcript drift | 不是 general ASR correction |
| Interruption candidate review | validate candidate generator precision | 10 AMI candidate windows | human event-level review | candidate precision | 10/10 interruption, precision `1.0`; recall/F1 blank | `experiments/results/interruption_label_summary_heldout_real.csv`, `data/reference/public/english_meeting_heldout/interruption_label_candidates.csv` | candidate generator promising | no recall/F1, no independent speaker identity |
| Controlled term rescue | stress-test glossary and correction | 175 text rows | no_retrieval, exact, fuzzy, phonetic, fused, rule/LLM correction | term precision/recall/F1, error before/after | rule/LLM variants often reach high term F1; negative controls reveal false-positive risk | `experiments/results/term_rescue_summary_controlled.csv`, `assets/result_charts/term_rescue_*.png` | mechanism works in fixtures | not real audio generalization |
| Controlled overlap safety | stress-test overlap-aware correction | 80 text rows | overlap-aware vs no-overlap awareness, rule/LLM | safety pass, forbidden changes, review flags | overlap-aware rule/LLM safety pass `1.0`; no-overlap rule fails high-overlap cases | `experiments/results/overlap_safety_summary_controlled.csv`, `assets/result_charts/overlap_*.png` | overlap flags reduce unsafe correction in fixtures | not real overlap performance |
| EvidenceGate audit | trained safety gate and leakage audit | controlled/augmented + heldout | audit-aware, evidence-only, risk-only models | macro F1, false/unsafe accept, review recall | audit-aware grouped test F1 `1.0` is leakage-prone; independent heldout best macro F1 only `0.325`; needs_review recall `0.000` to `0.033` | `docs/evidence_gate.md`, `experiments/results/evidence_gate/*`, `assets/result_charts/evidence_gate_*.png` | leakage audit is honest negative result; gate not ready | do not claim deployable ML gate |
| Selective correction pilot | early accept/reject/review feasibility | 72 authored proposals | always accept, always review, EccoGate | macro F1, unsafe accept, coverage | EccoGate macro F1 `0.818736`, unsafe accept `0.0`, coverage `0.708333` | `experiments/results/pilot/selective_correction_pilot_summary.csv`, `docs/pilot_selective_correction.md` | transparent abstention policy promising | auto-labeled pilot, not paper-level |
| Binary safe-to-apply | binary correction safety stress-test | 327 proposals | retrieval-only, overlap-unaware, Binary EccoGate, LLM judges | macro F1, unsafe apply, false block | Binary EccoGate macro F1 `0.883167`, unsafe apply `0.052632`; LLM with evidence macro F1 `0.596069`, unsafe apply `0.323887` | `experiments/results/binary_safe_apply/binary_safe_apply_summary.csv`, `docs/binary_safe_apply_correction.md`, `assets/result_charts/binary_*.png` | explicit evidence beats LLM self-judge for safety on controlled benchmark | mixed controlled/pilot labels |
| Mobile proxy | local speed/accuracy proxy | 100 rows | faster-whisper CPU int8 derived | RTF, error rate | `claim_level=mobile_style_proxy`; no true mobile | `experiments/results/v1/mobile_asr.csv`, `docs/mobile_asr_tradeoff.md` | local trade-off evidence | not phone, not whisper.cpp |
| whisper.cpp Level 1 | local-machine deployment benchmark | 76 rows | whisper.cpp tiny/base | WER/CER, RTF | AMI base WER `0.351123`, cleaned `0.280012`, RTF `0.056347`; French/Mandarin rows weak | `experiments/results/v1/whisper_cpp_mobile_level1_summary.csv`, `docs/whisper_cpp_level1.md` | local-machine Level 1 path completed | not true phone; not rerun on AISHELL-4 |
| ASR error pattern audit | explain ASR failures | selected real ASR rows | category classifier/audit | category counts/share | AMI errors dominated by disfluency/style; Earnings-22 includes domain term and number/unit errors | `experiments/results/asr_error_patterns_real_combined_summary.csv`, `docs/final_system_error_analysis.md` | supports targeted RAG and cleaned WER reporting | audit subset, not exhaustive |

## 6. 最终可主张的论文结论

可以主张：

1. TalkWeaver 构建了一个 evidence-grounded conversation map，把 ASR、说话人、时间、overlap、retrieved terms、correction audit 和 needs-review 统一到可审计输出中。
2. 真实公开数据 small-subset 显示，meeting speech 明显比 read speech 更难：AMI `base` WER `0.398364`，AISHELL-4 60x20 `base` CER `0.536940`，而 FLEURS zh `base` CER `0.113336`。
3. pyannote 在 AMI held-out 上可给出可用但不完美的 diarization：DER `0.106035`，JER `0.307202`，overlap F1 `0.490214`。
4. AISHELL-4 Mandarin meeting 的 diarization 更难：DER `0.326501`，JER `0.712577`，overlap F1 `0.261905`。
5. TalkWeaver 的 automatic evidence map 已在 AMI 24 clips 和 AISHELL-4 29 clips 上跑通，分别产生 speaker-labeled anchors、overlap anchors、events 和 needs-review flags。
6. RAG v3 是 conservative term recovery：Earnings-22 `base` term recall 从 `0.833333` 到 `1.000000`，WER 保持 `0.212099` 不变。
7. RAG 不是无条件有益：v2 让 `base` term F1 从 `0.972222` 降到 `0.930556`，说明强 ASR 会被 false-positive correction 伤害。
8. Controlled safety experiments 支持 overlap-aware correction 和 evidence-gated correction 的安全动机，但只在 fixtures/stress tests 上成立。
9. EvidenceGate 的泄漏审计是重要负结果：audit-aware `1.0` 是 policy-distillation sanity check；independent heldout 最好 macro F1 只有 `0.325`，说明训练 gate 不能作为主方法声称。
10. Mobile/whisper.cpp 结果只能说明 local-machine/proxy trade-off，不能说明手机真机部署。

## 7. 绝对不能主张的结论

不能主张：

- TalkWeaver 达到 state-of-the-art ASR、diarization、RAG correction 或 meeting summarization。
- 已完成 full FLEURS、full AMI、full AISHELL-4、full Earnings-22 benchmark。
- RAG v3 提升整体 WER。现有 v3 结论是 WER unchanged + base term recall improved。
- RAG correction 可以自动、安全地改所有错误。
- Interruption detector 有 recall/F1。当前只有 10 reviewed candidates 的 candidate precision。
- EvidenceGate 已经能真实泛化。independent heldout 是负结果。
- controlled term/overlap safety 结果可以代表真实音频性能。
- mobile proxy 或 whisper.cpp local-machine 结果是 phone-device measurement。
- UI 本身是算法贡献。UI 是 evidence review artifact。
- RQ4 preprocessing 已经被 real A/B 证明。当前未确认。

## 8. 如何串成诚实且有研究价值的故事

### 8.1 FLEURS / AMI 真实结果说明什么

- FLEURS 提供 read-speech multilingual baseline：`base` 比 `tiny` 更准，中文 FLEURS `base` CER `0.113336`。
- AMI 说明 meeting speech 受 disfluency、low-energy speech、overlap、speaker boundary 影响：formal AMI `base` WER `0.398364`，cleaned `0.331216`；held-out `base` WER `0.349338`，cleaned `0.289807`。
- 论文写法：read speech ASR cannot predict meeting robustness；需要 meeting-specific evidence map。

### 8.2 Controlled / curated safety 实验说明什么

- Controlled term rescue 说明 glossary retrieval 和 constrained correction 在设计好的专有词错误上有效，但 negative controls 暴露 false positive 风险。
- Controlled overlap safety 说明 overlap-aware rule/LLM 在 fixtures 中能避免 high-overlap forbidden changes。
- Binary safe-to-apply 说明 explicit evidence policy 比 LLM self-judge 更安全，但以 coverage 和 false block 为代价。
- 论文写法：这些实验支持 safety design rationale，不支持 real-world泛化。

### 8.3 EvidenceGate 泄漏审计说明什么

- 初始 audit-aware 模型 grouped test macro F1 `1.0`，但 docs 明确标注为 policy-distillation sanity check。
- 泄漏审计发现 41 个 features/fields，其中 14 direct label proxy，4 final audit outcome，4 risky reference-derived。
- strict grouped-test 仍好看，但 independent heldout 降到 macro F1 `0.325` 左右，并且 needs_review recall 几乎失败。
- 论文写法：这是一个负结果，证明只靠 controlled policy distillation 不够；需要外部真实标注或 human-in-the-loop。

### 8.4 正结果和负结果

正结果：

- real ASR/diarization/evidence-map pipeline 跑通。
- AMI/AISHELL 明确暴露 meeting difficulty。
- RAG v3 在 `base` 上提高 term recall 且不改变 WER。
- controlled overlap-aware safety 通过 fixtures。
- human-reviewed interruption candidates 10/10 为真实 interruption event。

负结果：

- RAG v2 会伤害 `base` term F1。
- RAG v3 不降低 WER，只是保守 term recovery。
- EvidenceGate independent heldout 表现差。
- LLM self-judge with evidence 仍有 unsafe apply `0.323887`。
- whisper.cpp French/Mandarin rows weak，不适合作主质量 claim。
- Interruption 没有 recall/F1。

### 8.5 最诚实的论文故事

推荐主线：

> We do not claim to solve ASR. We show that chaotic meetings require evidence-grounded post-processing. TalkWeaver converts real ASR and diarization outputs into an auditable map, exposes uncertainty, and constrains RAG/LLM correction so that domain-term recovery is possible without hiding false positives.

## 9. 论文建议结构

### Title

推荐：

> TalkWeaver: Evidence-Grounded Conversation Maps for Chaotic Multi-Speaker Meetings

备选副标题：

> Auditing ASR, Diarization, Overlap, and RAG-Based Term Recovery

### Abstract 要点

应包含：

- problem：multi-speaker meeting transcripts lack reliable evidence for who/when/overlap/correction.
- method：ASR + diarization + temporal anchors + overlap flags + RAG glossary + constrained correction + audit.
- results：AMI/AISHELL public-data small subsets, Earnings-22 RAG v3, controlled safety.
- caveat：not SOTA, not full corpus, not phone-device.

引用：

- `docs/final_claim_matrix.md`
- `experiments/results/asr_benchmark_summary_real.csv`
- `experiments/results/pyannote_diarization_heldout_summary_real.csv`
- `experiments/results/earnings22_v3_blind_ablation_v3_summary.csv`

### Introduction 逻辑

1. Meeting transcript failure is not only lexical WER.
2. Need evidence: speaker, time, overlap, term source, correction provenance.
3. LLM correction is risky without evidence.
4. TalkWeaver proposes auditable conversation maps.

引用：

- `README.md`
- `docs/research_questions.md`
- `docs/final_system_error_analysis.md`

### Method

按 pipeline 写：

- preprocessing/ASR
- diarization and temporal anchoring
- overlap/event evidence
- RAG term retrieval
- constrained/LLM correction and audit
- ConversationMap UI

引用：

- `backend/asr.py`
- `backend/diarization.py`
- `backend/alignment.py`
- `backend/overlap.py`
- `backend/rag.py`
- `backend/llm_correction.py`
- `backend/conversation_map.py`
- `webapp/detective_ui.py`

### Experiments

分四组：

1. real public-data ASR/diarization/evidence maps；
2. RAG term recovery on Earnings-22；
3. controlled safety and correction gates；
4. mobile/local deployment diagnostics。

引用：

- `docs/asr_benchmark.md`
- `docs/pyannote_diarization_benchmark.md`
- `docs/earnings22_v3_blind_ablation_v3.md`
- `docs/overlap_safety_experiment.md`
- `docs/evidence_gate.md`
- `docs/mobile_asr_tradeoff.md`
- `docs/whisper_cpp_level1.md`

### Results

建议表格：

- ASR summary by dataset/model。
- Diarization summary by dataset。
- Evidence-map summary。
- RAG v2/v3 ablation。
- Controlled safety summary。
- EvidenceGate leakage/heldout summary。

引用：

- `experiments/results/*summary*.csv`
- `assets/result_charts/*.png`

### Limitations

必须写：

- all public-data results are small subsets.
- RQ4 preprocessing not confirmed.
- no full interruption recall/F1.
- EvidenceGate not ready.
- mobile not true phone.
- controlled fixtures do not prove real generalization.
- external LLM privacy/cost risks.

引用：

- `docs/final_claim_matrix.md`
- `docs/final_system_error_analysis.md`
- `docs/evidence_gate.md`

### Conclusion

强调：

- contribution is auditable evidence map + constrained correction philosophy.
- positive result is not WER SOTA, but safe and inspectable handling of chaotic meeting evidence.

## 10. 推荐保留的 5-8 张论文图表

优先保留：

1. `assets/architecture.png` - 系统架构图。
2. `assets/result_charts/asr_error_by_dataset.png` - ASR 不同数据集难度对比。
3. `assets/result_charts/asr_error_by_language.png` - multilingual/read vs meeting context。
4. `assets/result_charts/asr_rtf_by_model.png` - accuracy/speed trade-off。
5. `assets/result_charts/workflow_ablation_completeness.png` - evidence-map 结构增加。
6. `assets/result_charts/workflow_ablation_review_flags.png` - needs-review/audit 机制。
7. `assets/result_charts/term_rescue_f1_by_variant.png` 或 `assets/result_charts/term_rescue_error_delta.png` - controlled RAG mechanism。
8. `assets/result_charts/evidence_gate_feature_leakage_audit.png` 或 `assets/result_charts/evidence_gate_heldout_confusion_matrix.png` - honest leakage/negative result。

谨慎使用：

- `assets/result_charts/binary_safe_apply_macro_f1.png`：可以作为 appendix safety analysis。
- `assets/result_charts/overlap_safety_pass_rate.png`：必须标明 controlled fixtures。
- whisper.cpp/mobile 图若没有清晰图表，建议用表格而不是主图。

## 11. 推荐保留的核心表格

| 表格 | 内容 | 来源 |
| --- | --- | --- |
| Dataset table | all real/controlled/mock/proxy datasets | 本报告第 3 节 |
| ASR baseline table | FLEURS/AMI/AISHELL WER/CER/RTF | `experiments/results/asr_benchmark_summary_real.csv`, `asr_benchmark_aishell4_60x20_summary_real.csv` |
| Diarization table | AMI/AISHELL DER/JER/overlap F1 | `pyannote_diarization_*_summary_real.csv` |
| Evidence-map table | AMI/AISHELL anchors/events/review flags | `automatic_pyannote_workflow_*_summary_real.csv` |
| RAG ablation table | v2 negative and v3 conservative result | `earnings22_final_blind_ablation_v2_summary.csv`, `earnings22_v3_blind_ablation_v3_summary.csv` |
| Safety controlled table | overlap-aware vs no-overlap-aware | `overlap_safety_summary_controlled.csv` |
| EvidenceGate audit table | audit-aware vs strict vs independent heldout | `docs/evidence_gate.md`, `evidence_gate_validation_metrics.csv` |
| Claim boundary table | can/cannot say | `docs/final_claim_matrix.md` |

## 12. 当前还缺什么

不是必须补实验，但写论文前必须承认：

- 没有 full-corpus benchmark。
- 没有 true phone benchmark。
- 没有 full interruption timeline annotation，所以没有 recall/F1。
- RQ4 preprocessing 没有正式 real A/B 支撑。
- 没有人类可读性评分或用户研究。
- EvidenceGate independent heldout 是负结果，不能作为主方法部署。
- RAG v3 没有降低 WER，只改善一部分 term recall。

## 13. 论文写作前必须核对 checklist

- [ ] 论文所有数字都能追到 `experiments/results/*.csv` 或 `docs/*.md`。
- [ ] 每个 result 都标注 real / controlled / mock / proxy。
- [ ] ASR 表里区分 WER、CER、cleaned WER、RTF。
- [ ] AMI/AISHELL 结果写 small subset，不写 full corpus。
- [ ] RAG v2 负结果必须保留。
- [ ] RAG v3 写 safe term recovery，不写 WER improvement。
- [ ] Interruption 只写 candidate precision，不写 recall/F1。
- [ ] EvidenceGate 写 leakage audit 和 independent heldout failure。
- [ ] Mobile 写 local CPU proxy / local-machine whisper.cpp，不写 phone-device。
- [ ] UI 写 evidence viewer / research artifact，不写普通 meeting summarizer。
- [ ] `.env`、raw audio、ASR prediction JSON、large archives 不进入 Git。
- [ ] 合并前运行 `python experiments/validate_final_research_artifacts.py`，并记录是否通过。

## 14. 最终一句话交接

主实验是 real public-data ASR + diarization + automatic ConversationMap + Earnings-22 RAG v3。最强结论是：TalkWeaver 能把混乱会议中的 ASR、speaker-time、overlap、retrieval 和 correction uncertainty 组织成可审计证据地图，并且 RAG v3 能在不改变 WER 的前提下提高部分 domain-term recall。最大限制是：所有真实结果仍是 small subset，EvidenceGate 和 safety gate 主要是 controlled/curated 证据，mobile 不是真机，interruption 没有 recall/F1。
