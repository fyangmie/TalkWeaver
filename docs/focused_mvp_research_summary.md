# TalkWeaver Focused MVP 实现与科研故事总结

更新日期：2026-06-17

## 1. 一句话总结

本次工作把 TalkWeaver 从一个已有的语音处理 pipeline，推进成一个可以展示、解释和审计的 Focused MVP：系统不只输出一段转写文本，而是输出带时间、说话人、重叠发言、术语救援和 LLM 改写审计的 evidence-grounded conversation map。

核心目标不是单纯做一个网页，而是让评审能看清楚：

- 每句话来自哪个时间段；
- 哪些术语被系统救回；
- 哪些 LLM 改写有证据支持；
- 哪些改写证据不足，需要人工复核或拒绝。

## 2. Baseline 原来有什么

原仓库已经有比较完整的 TalkWeaver 原型，包括：

- mock pipeline；
- ASR、diarization、alignment、overlap detection 等模块；
- term rescue / glossary retrieval；
- constrained correction；
- workflow ablation、term rescue、overlap safety 等实验脚本和结果文件；
- Streamlit 前端和若干 visualization 页面；
- pytest 测试。

但是 baseline 更像一个后端工程原型：能跑 pipeline、能生成 JSON/CSV/图表，但不够直接展示 TalkWeaver 的核心价值，也不够清楚地表达“证据约束的转写改写”这个研究点。

## 3. 本次具体做了什么

### 3.1 新增 Focused MVP demo 数据

新增文件：

```text
data/focused_mvp/focused_chaotic_demo_conversation_map.json
```

该 demo 是一段合成的混乱多人对话，包含：

- 时间锚点；
- speaker 标注；
- overlap 发言；
- 术语误识别；
- 术语救援结果；
- LLM correction audit；
- 一个 unsupported LLM change 示例。

重要边界：这份数据明确标注为 synthetic demo，不声称来自真实音频，也不作为真实性能实验结果。

### 3.2 新增四个聚焦页面

默认 Streamlit 入口被收敛成四个核心页面：

1. Evidence Timeline
   - 按时间线展示每个 utterance；
   - 展示 speaker、时间戳、overlap、raw text、corrected text 和 correction status。

2. Term Rescue
   - 展示 raw phrase 到 corrected term 的恢复过程；
   - 展示 term rescue 的来源、状态和证据。

3. Correction Audit
   - 展示 LLM 改写是否被证据支持；
   - 区分 supported、weakly_supported、needs_review、unsupported。

4. Evidence Dashboard
   - 汇总当前 conversation map 的证据覆盖、术语救援、审计状态和风险点。

相关新增/修改文件：

```text
webapp/app.py
webapp/views/focused.py
webapp/data_loader.py
webapp/detective_ui.py
webapp/components/project_layout.py
```

### 3.3 扩展后端审计字段

为了让结果可解释，而不是只有“改完后的文本”，本次扩展了后端 schema 和相关逻辑。

主要新增字段：

- correction_status；
- term rescue status；
- term rescue source；
- raw_phrase；
- corrected_term。

相关修改文件：

```text
backend/schemas.py
backend/constrained_correction.py
backend/term_rescue.py
```

correction status 目前分为：

- supported：改写有证据支持；
- weakly_supported：有部分证据，但仍偏弱；
- needs_review：需要人工复核；
- unsupported：没有足够证据，应视为风险改写。

### 3.4 补充测试和文档

新增/修改：

```text
tests/test_frontend_data_layer.py
README.md
```

新增测试确认 committed focused demo 可以被前端数据层自动发现。

## 4. 科研故事

### 4.1 问题背景

传统 ASR 系统通常输出一段 transcript。后续如果用 LLM 润色，文本可能更流畅，但会带来新的风险：

- LLM 可能补充原音频里不存在的内容；
- 专业术语、人名、缩写容易被误识别；
- 多人会议中的 speaker attribution 容易错；
- overlap 发言场景下，系统可能把 A 说的话归给 B；
- 最终用户很难知道一句话为什么被改、证据来自哪里。

因此，问题不只是“转写准不准”，还包括“改写能不能被证据约束”和“结果能不能被审计”。

### 4.2 核心研究问题

可以将 TalkWeaver 的研究问题表述为：

> Can an evidence-grounded transcript pipeline improve domain-term recovery and reduce unsafe LLM corrections in noisy multi-speaker conversations?

中文表述：

> 在嘈杂、多说话人、术语密集的对话场景中，一个证据约束的转写 pipeline，能否提升术语恢复能力，并降低 LLM 后处理带来的不安全改写？

### 4.3 方法主张

TalkWeaver 的方法不是单纯追求更低 WER，而是构建 conversation map：

- temporal anchors：把文本绑定到时间片段；
- speaker evidence：记录说话人和重叠发言；
- term rescue：用 glossary/RAG 恢复专业术语；
- constrained correction：限制 LLM 不得自由脑补；
- correction audit：记录每次改写是否有证据支持。

因此，核心贡献可以写成：

1. 一个 evidence-grounded conversation map 表示；
2. 一个术语救援和证据约束改写 pipeline；
3. 一个 correction audit 机制，用于发现 unsupported LLM edits；
4. 一个面向人工复核的证据展示界面。

## 5. 跑了什么实验

目前实验分为三类：mock 消融、controlled fixture 实验、小规模真实/公开数据实验。

### 5.1 Mock 消融实验

本次实际跑通了 mock ablation：

```bash
python experiments/run_ablation.py --mock
```

结果文件：

```text
experiments/results/ablation_results.csv
experiments/results/term_error_results.csv
experiments/results/latency_results.csv
```

消融结果如下：

| Pipeline | WER | Speaker/WDER error | Term error | Overlap error | Hallucinated corrections |
| --- | ---: | ---: | ---: | ---: | ---: |
| Whisper only | 0.4444 | 1.0 | 1.0 | 0.25 | 0 |
| + preprocessing | 0.4444 | 1.0 | 1.0 | 0.25 | 0 |
| + diarization + alignment | 0.4444 | 0.0 | 1.0 | 0.0 | 0 |
| + structured LLM correction | 0.4444 | 0.0 | 1.0 | 0.0 | 0 |
| + RAG glossary | 0.0 | 0.0 | 0.0 | 0.0 | 0 |
| + overlap-aware correction | 0.0 | 0.0 | 0.0 | 0.0 | 0 |

解释：

- diarization + alignment 在 mock 中消除了 speaker/WDER error 和 overlap error；
- structured LLM correction 单独没有救回术语；
- RAG glossary 在 mock 中把术语错误降到 0；
- overlap-aware correction 保持安全约束。

注意：这是 deterministic mock/demo metric，不能作为真实模型性能 claim。

### 5.2 Mock 术语恢复实验

结果文件：

```text
experiments/results/term_error_results.csv
```

| Pipeline | Term error | Precision | Recall |
| --- | ---: | ---: | ---: |
| Whisper only | 1.0 | 0.0 | 0.0 |
| Structured LLM correction | 1.0 | 0.0 | 0.0 |
| Structured LLM correction + RAG glossary | 0.0 | 1.0 | 1.0 |

解释：

- 没有 glossary/RAG 时，mock 中的专业术语全部缺失；
- 加入 RAG glossary 后，pyannote、diarization、WER、DER、RAG 等术语被恢复；
- 这支持“术语救援需要外部知识约束，而不能只靠 LLM 自由改写”的论点。

### 5.3 Controlled term rescue 实验

结果文件：

```text
experiments/results/term_rescue_summary_controlled.csv
```

核心结论：

- no_retrieval 基本救不回术语；
- exact_glossary 在 easy/medium 场景效果较好，hard 英文 recall 为 0.9；
- fused 在英文 easy/medium/hard controlled fixtures 上 precision、recall、F1 均达到 1.0；
- fused_plus_rule_correction 在英文 easy/medium/hard 场景中将 text error after 降到 0.0；
- negative control 会被标为 needs_review，说明系统不会盲目替换术语。

该实验支持：

> 检索增强的术语救援比无检索 baseline 更可靠，尤其在专业术语和近音误识别场景下。

限制：

- controlled fixtures 是人为构造文本，不是真实音频；
- 可以证明模块行为，但不能直接证明真实会议中的整体性能。

### 5.4 Controlled overlap safety 实验

结果文件：

```text
experiments/results/overlap_safety_summary_controlled.csv
```

核心结论：

- overlap_aware_rule 和 overlap_aware_llm 在 overlap/high uncertainty 场景下 safety pass rate 为 1.0；
- no_overlap_awareness_rule 在 overlap/high uncertainty 场景下 safety pass rate 为 0.0；
- no-overlap-awareness 在重叠发言场景中更容易出现 forbidden change 或 speaker attribution 风险。

该实验支持：

> overlap-aware correction 的价值主要不是让文本更流畅，而是降低重叠发言场景中的错误改写和错误归因风险。

### 5.5 小规模真实/公开数据结果

结果文件：

```text
experiments/results/workflow_ablation_summary_real.csv
```

包含数据集：

- AMI Meeting Corpus；
- Google FLEURS en；
- Google FLEURS fr；
- Google FLEURS zh-CN。

ASR-only 错误率：

| Dataset | Language | Clips | Mean ASR error |
| --- | --- | ---: | ---: |
| AMI Meeting Corpus | en | 2 | 0.370536 |
| Google FLEURS | en | 5 | 0.154242 |
| Google FLEURS | fr | 5 | 0.273839 |
| Google FLEURS | zh-CN | 5 | 0.089651 |

该部分说明 pipeline 可以在真实/公开数据小样本上生成 workflow ablation 结果。

限制：

- 数据规模很小；
- speaker-time rows 有 oracle-assisted 成分；
- full_talkweaver 的 corrected error rate 没有显示出明显优于 ASR-only；
- 因此不能强 claim 真实 ASR 准确率大幅提升。

## 6. 本次验证结果

本次实现后运行了以下验证：

```bash
python -m pytest -q
python scripts/run_pipeline.py --mock
python experiments/run_ablation.py --mock
MPLCONFIGDIR=/tmp/matplotlib-talkweaver python experiments/plot_results.py
```

结果：

- pytest：157 passed；
- mock pipeline：通过；
- mock ablation：通过；
- result charts 生成：通过。

注意：运行 plot_results.py 会重新生成 assets/result_charts 下的图表文件；运行 run_ablation.py 会更新 experiments/results 下的 mock CSV。

## 7. 当前结果能支撑什么结论

可以支撑的结论：

1. TalkWeaver 可以把 transcript 表示成 evidence-grounded conversation map。
2. glossary/RAG 对术语恢复有明显作用。
3. overlap-aware correction 对重叠发言场景的安全性有价值。
4. correction audit 可以暴露 unsupported LLM edits。
5. Focused MVP 可以把证据链展示给人工复核者。

不能支撑的结论：

1. 不能说 TalkWeaver 在真实会议数据上显著降低 WER。
2. 不能说系统已经超过强 ASR/diarization baseline。
3. 不能说目前结果已经经过大规模真实数据验证。
4. 不能把 mock demo 数字写成真实性能提升。

## 8. 能否支撑会议论文

当前状态不足以支撑一篇强会议主会论文。

更现实的定位：

- 课程项目报告：可以；
- 系统 demo：可以；
- workshop paper：有机会；
- demo paper：有机会；
- 强会议 full paper：目前不够。

主要原因：

- 真实数据规模太小；
- controlled fixtures 和 mock 实验占比过高；
- 缺少充分的真实音频端到端评测；
- 缺少人工标注的 unsupported correction ground truth；
- 缺少统计显著性分析；
- 与强 baseline 的对比还不够系统。

## 9. 如果要推进到论文级，还需要补什么

建议补一个 focused_mvp_eval 实验包。

### 9.1 数据

至少准备几十到上百条 clip，覆盖：

- 多人会议；
- overlap 发言；
- 专业术语密集场景；
- 中英文或多语言场景；
- 噪声和口语化表达。

### 9.2 Baseline

建议比较：

- ASR-only；
- ASR + plain LLM correction；
- ASR + glossary/RAG；
- ASR + diarization/alignment；
- TalkWeaver full pipeline。

### 9.3 指标

建议报告：

- WER/CER；
- DER/WDER；
- term precision/recall/F1；
- unsupported correction rate；
- needs-review detection accuracy；
- evidence coverage；
- speaker attribution error；
- human review time 或人工信任评分。

### 9.4 人工标注

需要人工标注：

- 正确 transcript；
- speaker/time boundaries；
- domain terms；
- overlap 区间；
- 哪些 LLM correction 是 supported / unsupported。

### 9.5 论文主张

如果补强实验后，论文主张可以是：

> TalkWeaver does not merely improve transcripts; it makes transcript correction auditable by grounding edits in temporal, speaker, overlap, and glossary evidence.

中文：

> TalkWeaver 的贡献不是简单让转写文本更流畅，而是让转写改写过程可证据化、可审计、可人工复核。

## 10. 当前项目状态

本地 Focused MVP 已经可以运行：

```bash
streamlit run webapp/app.py
```

本次实现后的核心文件：

```text
data/focused_mvp/focused_chaotic_demo_conversation_map.json
webapp/views/focused.py
webapp/app.py
webapp/data_loader.py
backend/schemas.py
backend/constrained_correction.py
backend/term_rescue.py
tests/test_frontend_data_layer.py
README.md
```

当前最适合对外讲法：

> We built a focused prototype of TalkWeaver that turns noisy multi-speaker transcripts into auditable conversation maps. The system highlights temporal evidence, rescued domain terms, overlap-sensitive corrections, and unsupported LLM edits. Existing mock and controlled experiments support the value of glossary-based term rescue and overlap-aware safety, while larger real-audio evaluation remains future work.
