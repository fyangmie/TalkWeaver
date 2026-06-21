# TalkWeaver 科研交接文档

更新时间：2026-06-21

这份文档是给后续继续做项目、写论文、录视频的人看的。它回答四个问题：

1. 我们现在到底做了什么。
2. 这些东西在科研主线里分别有什么用。
3. 目前离“完整论文成品”还差什么。
4. 接下来应该按什么顺序补，怎么验证，哪些结果可以说，哪些结果不能说。

## 2026-06-21 最新实施状态

本节覆盖下方早期记录中已经过时的部分。

已经补上的真实结果：

- 新增最终 claim/错误分析交付文档：
  - `docs/final_claim_matrix.md`
  - `docs/final_system_error_analysis.md`
  - `experiments/validate_final_research_artifacts.py`
  - 作用：固定哪些结果可以说、哪些不能说，并自动检查关键 CSV 行数和旧表述残留。
- AMI held-out 已从 6 条扩到 24 条，覆盖 `ES2002a/b/c/d`，每个 meeting 6 条。
- pyannote 自动 diarization 已跑通，不再 skipped。24 条 AMI held-out 结果：
  - mean DER = 0.106035
  - mean JER = 0.307202
  - DER skip-overlap = 0.081325
  - JER skip-overlap = 0.209765
  - overlap F1 = 0.490214
  - mean RTF = 0.692477
- AMI held-out ASR 已跑 tiny/base/small：
  - tiny WER = 0.366577，cleaned WER = 0.314403，RTF = 0.035976
  - base WER = 0.349338，cleaned WER = 0.289807，RTF = 0.071662
  - small WER = 0.298610，cleaned WER = 0.233636，RTF = 0.176200
- 新增自动 TalkWeaver evidence-map 结果：
  - `experiments/results/automatic_pyannote_workflow_heldout_real.csv`
  - 24 条 base ASR + pyannote maps
  - 平均 4.75 anchors、4.25 speaker-labeled anchors、1.79 overlap anchors、2.04 events、2.29 needs-review flags
  - `experiments/results/automatic_pyannote_workflow_aishell4_60x20_real.csv`
  - 29 条 AISHELL-4 base ASR + pyannote maps
  - 平均 6.76 anchors、4.48 speaker-labeled anchors、0.38 overlap anchors、0.24 events、2.66 needs-review flags
- RAG v3 已从 1 文件 smoke 扩到 6 文件 blind：
  - `data/manifests/earnings22_v3_blind_6x180.csv`
  - ablation 包含 `asr_only`、`glossary_candidates_only`、`llm_without_rag_conservative`、`rag_evidence_gate_v3`、`rag_llm_verifier_v3`
  - base WER 0.212099 -> 0.212099
  - base term recall 0.833333 -> 1.000000
  - tiny WER 0.251941 -> 0.251941
  - tiny term recall 0.833333 -> 0.833333
  - 结论是 v3 更像安全门控和 term-recall 诊断，不是 WER 提升方法
- interruption candidates 已从 2 条扩到 10 条，并在 2026-06-21 完成人工事件级审核：
  - 10/10 候选窗口包含真实 interruption event
  - event-level candidate precision = 1.0
  - speaker pair 仍来自 AMI reference timing，没有由人工通过音色独立确认
  - 仍不能报告 recall/F1，因为缺完整时间轴标签或非候选负样本
- Mandarin meeting 已从小规模 sanity check 升级为 AISHELL-4 固定 benchmark 子集：
  - `data/manifests/mandarin_meeting_real.csv`
  - 12 条 20s 真实 AISHELL-4 普通话会议片段，总计 240s
  - 正式 50 条 ASR 复跑中：tiny CER=0.679514，base CER=0.609966
  - AISHELL-only small 补充跑分：small CER=0.464762
  - pyannote 在 2 条多说话人片段上可评分：DER=0.237577，JER=0.698066
  - `data/manifests/aishell4_benchmark_60x20.csv`
  - 完整 5.24GB AISHELL-4 `test.tar.gz` 已本地下载、校验并解压，原始包和 FLAC/TextGrid/RTTM 均被 Git 忽略
  - 固定 benchmark 子集：20 个 test recordings，每个录音 3 条 20s clips，共 60 条、1200s
  - 其中 29 条是多说话人可评分片段，10 条带 reference overlap
  - ASR 60 条结果：tiny CER=0.648291，base CER=0.536940，small CER=0.481842
  - pyannote 29 条多说话人结果：DER=0.326501，JER=0.712577，overlap F1=0.261905
- whisper.cpp Level 1 已在 2026-06-21 跑通：
  - `experiments/results/v1/whisper_cpp_mobile_level1.csv`
  - 76 rows，tiny/base 各 38 条，skipped=0
  - AMI en base WER=0.351123，cleaned WER=0.280012，RTF=0.056347
  - AMI en tiny WER=0.382383，cleaned WER=0.313909，RTF=0.030548
  - 这是 local-machine benchmark，不是手机真机 benchmark

当前最准确的论文状态：

```text
ASR baseline + pyannote diarization + automatic evidence map + RAG v3 blind
已经有真实小规模结果；event-level human interruption labels 已有 10 条，
true whisper.cpp Level 1 和 AISHELL-4 Mandarin meeting 固定子集结果已补上；
RAG ablation 与 AISHELL-4 evidence-map 闭环已补上；手机真机 benchmark
仍是最后缺口。
```

---

## 1. 一句话版本

TalkWeaver 现在不是一个简单的 Whisper 转文字项目，而是一个面向混乱多人会议的“证据地图”系统：

```text
音频
-> ASR 识别文字
-> 说话人/时间/重叠区域对齐
-> 生成带证据的时间锚点 transcript
-> 用 RAG 找可能被误听的专业词
-> 用受约束的 LLM 做非常保守的修正
-> 记录每一次修正证据，避免 LLM 乱改
-> 输出给网页、实验表格、论文分析
```

现在已经完成了大部分工程骨架，也跑了一些真实小规模结果。最重要的结论是：

- baseline ASR 在干净单人语音上还可以，但在多人会议上明显更差。
- RAG 修专有词这个方向是有信号的，但不能直接放开让 LLM 改，否则会出现假阳性。
- v2 RAG 在 `tiny` 模型上提升了 term F1，但在 `base` 模型上伤害了 term F1，所以论文里必须强调“安全门控”和“防乱改”。
- v3 安全门控已经实现，并扩到 6 个 Earnings-22 blind 文件；当前信号是 term recall 提升，而不是 WER 提升。
- 自动 pyannote DER/JER、自动 evidence map、人工 event-level interruption 标签、真实 whisper.cpp local-machine benchmark 和 AISHELL-4 60-clip 普通话会议子集都已经有真实结果。

---

## 2. 最终科研主线

项目最终题目来自 `PRD2.md` 和 `AGENTS.md`：

```text
TalkWeaver: AI Meeting Detective for Chaotic Multi-Speaker Conversations
```

核心不是“做一个会议总结器”，也不是“调用一个 API 识别语音”。核心是：

```text
把混乱多人对话变成一个可审计的证据地图。
```

这里的“证据”包括：

- 原始音频时间；
- ASR 原始识别文本；
- 说话人时间段；
- 重叠说话区域；
- interruption 候选；
- RAG 检索到的专业词；
- LLM 修改前后对照；
- 哪些修改被接受、哪些被拒绝、哪些需要人工复核；
- WER、CER、DER、JER、term F1、latency 等实验指标。

### 2.1 四个研究问题

当前项目应该围绕下面四个 RQ 写论文：

```text
RQ1:
带说话人和时间结构的 transcript，能不能提升多人会议可读性和 speaker consistency？

RQ2:
如果系统知道某一段有重叠说话，能不能减少 LLM 在不确定区域的乱改？

RQ3:
RAG 检索专业词，能不能减少 ASR 对专业名词、公司名、财经词的错误？

RQ4:
本地 ASR 预处理和小模型/量化模型，在准确率和速度之间有什么 trade-off？
```

### 2.2 主贡献和辅助贡献

主贡献应该写成三条：

1. **Conversation Evidence Map**
   把 ASR、speaker、overlap、interruption、RAG、correction audit 合并到一个带时间锚点的 transcript 里。

2. **Overlap-aware Safe Correction**
   LLM 不是自由改文本，而是在时间、说话人、重叠区域、RAG 候选词的约束下保守修正，并记录证据。

3. **RAG-based Domain Term Recovery**
   对专有词、公司名、财经术语这类 ASR 易错点，使用外部词表/RAG 给 LLM 第二次判断机会，但通过 v2/v3 gate 防止数据泄露和乱改。

辅助贡献：

- 多语言 ASR 小规模评测；
- whisper.cpp/mobile Level 1 trade-off 通道；
- Streamlit detective-style demo；
- 自动化实验脚本、manifest、结果表和文档。

---

## 3. 当前数据流

### 3.1 运行时数据流

```text
audio file
  |
  v
backend/preprocessing.py
  - 转 mono
  - 转 16kHz
  - 可选 normalize/denoise
  |
  v
backend/asr.py
  - faster-whisper baseline
  - mock fallback
  - 输出 segments / words / timestamps
  |
  v
backend/diarization.py
  - pyannote diarization，如果有 HF_TOKEN
  - mock fallback
  - 或实验中使用 public reference speaker turns
  |
  v
backend/alignment.py + backend/overlap.py
  - 把 ASR words/segments 对齐到 speaker turns
  - 检测 overlap
  - 生成 temporal-anchor JSON
  |
  v
backend/rag.py
  - 从 domain_terms 或 controlled_terms 找可能专业词
  - 返回候选词和分数
  |
  v
backend/llm_correction.py / backend/term_verifier.py
experiments/evaluate_rag_llm_correction.py
  - 构造带证据的 prompt
  - LLM 只允许做局部、可解释、可审计的改动
  - v2/v3 gate 判断是否接受
  |
  v
outputs / experiments/results / webapp
  - transcript
  - corrected transcript
  - audit trail
  - metrics
  - charts
```

### 3.2 实验数据流

实验不直接依赖网页。实验的基本链路是：

```text
manifest CSV
  -> download/prepared public audio + references
  -> run ASR benchmark
  -> run diarization / overlap / workflow ablation
  -> run RAG correction
  -> run ablation
  -> write CSV + markdown analysis
```

重要文件：

- `data/manifests/formal_eval_real.csv`
- `data/manifests/english_meeting_heldout_real.csv`
- `data/manifests/earnings22_final_blind_12x180.csv`
- `data/manifests/earnings22_v3_blind_smoke_1x180.csv`
- `experiments/results/asr_benchmark_real.csv`
- `experiments/results/asr_benchmark_summary_real.csv`
- `experiments/results/workflow_ablation_real.csv`
- `experiments/results/workflow_ablation_summary_real.csv`
- `experiments/results/earnings22_final_blind_rag_llm_v2_summary.csv`
- `experiments/results/earnings22_v3_blind_smoke_rag_llm_v3_summary.csv`

---

## 4. 已经完成了什么

### 4.1 真实 ASR baseline

已经完成：

- FLEURS 英语 10 条；
- FLEURS 法语 10 条；
- FLEURS 中文 10 条；
- AMI 英文多人会议 8 条；
- AISHELL-4 普通话会议 12 条；
- 共 50 条真实 public dataset 样本；
- 跑了 faster-whisper `tiny` 和 `base`；
- 保存了 WER/CER、cleaned WER、RTF、runtime。
- 另外新增了独立 AISHELL-4 60x20 benchmark subset，覆盖 20 个 test recordings。

结果位置：

- `experiments/results/asr_benchmark_real.csv`
- `experiments/results/asr_benchmark_summary_real.csv`
- `docs/asr_benchmark.md`

当前关键数字：

| 模型 | 数据集 | 语言 | 指标 | 平均错误率 |
|---|---|---|---|---:|
| tiny | FLEURS | en | WER | 0.210358 |
| base | FLEURS | en | WER | 0.114374 |
| tiny | FLEURS | fr | WER | 0.387342 |
| base | FLEURS | fr | WER | 0.227136 |
| tiny | FLEURS | zh-CN | CER | 0.222614 |
| base | FLEURS | zh-CN | CER | 0.113336 |
| tiny | AMI meeting | en | WER | 0.432335 |
| base | AMI meeting | en | WER | 0.398364 |

怎么解释：

- `base` 比 `tiny` 明显好，符合预期。
- AMI 多人会议比 FLEURS 单句语音更难，说明“混乱会议场景”确实有问题。
- 目前 AMI 只有小样本，不能写成完整数据集结论，只能写成 small-subset evidence。

### 4.2 Workflow ablation

已经完成：

- `asr_only`
- `temporal_anchor_only`
- `reference_speaker_time`
- `overlap_aware`
- `term_rescue`
- `constrained_correction`
- `full_talkweaver`

结果位置：

- `experiments/results/workflow_ablation_real.csv`
- `experiments/results/workflow_ablation_summary_real.csv`
- `docs/workflow_ablation.md`

当前关键数字：

在 AMI 8 条小样本上，`full_talkweaver` 平均生成：

- 4.5 个 temporal anchors；
- 3.625 个 speaker-labeled anchors；
- 1.25 个 overlap anchors；
- 3.75 个 overlap/interruption-related events；
- 4.5 个 correction audits；
- 0 个 unsupported changes；
- 2.125 个 needs review flags。

怎么解释：

- 这证明系统能把“原始 ASR 文本”变成“带说话人、时间、overlap、审计记录的证据地图”。
- 但这个 ablation 目前用的是 reference speaker-time rows，所以不能声称自动 diarization 已经成功。
- 论文里应该把它叫做 reference-assisted evidence-map ablation。

### 4.3 Earnings-22 RAG v2

已经完成：

- Earnings-22 财报语音；
- 12 个 held-out/final blind 片段；
- faster-whisper tiny/base ASR；
- RAG glossary；
- DeepSeek LLM verifier；
- v2 evidence gate；
- 不把 reference transcript 发给 LLM；
- 输出修正前后 WER、term recall、term F1。

结果位置：

- `experiments/results/earnings22_final_blind_rag_llm_v2.csv`
- `experiments/results/earnings22_final_blind_rag_llm_v2_summary.csv`
- `docs/earnings22_final_blind_rag_v2_error_analysis.md`
- `docs/earnings22_final_blind_ablation_v2.md`

当前关键数字：

| 模型 | WER before | WER after | Term F1 before | Term F1 after | 解释 |
|---|---:|---:|---:|---:|---|
| tiny | 0.221805 | 0.221978 | 0.888889 | 0.930556 | term F1 提升，但 WER 极小变差 |
| base | 0.186844 | 0.187018 | 0.972222 | 0.930556 | 发生假阳性，term F1 变差 |

怎么解释：

- RAG 对弱模型 `tiny` 有帮助信号。
- 对强一点的 `base`，baseline 已经识别得比较好，RAG/LLM 反而可能乱改。
- 所以 v2 不能作为最终方法直接报告为“稳定提升”。
- 最合理的科研故事是：RAG 有潜力，但必须加入更强的安全门控，这就是 v3 的动机。

### 4.4 RAG v3 安全门控

已经完成：

- 在 `experiments/evaluate_rag_llm_correction.py` 里加入 `--gate-version v2|v3`。
- 在 `experiments/evaluate_earnings22_ablation.py` 里加入 `--gate-version`。
- v3 比 v2 更保守：
  - 只接受预定义 ASR 错误形式；
  - 常见词不能随便映射成公司名；
  - 数字和单位不能乱改；
  - 如果 glossary 有上下文条件，必须命中上下文；
  - 不确定就 no-op 或 needs_review。
- 增加了测试。

结果位置：

- `experiments/results/earnings22_v3_blind_smoke_rag_llm_v3.csv`
- `experiments/results/earnings22_v3_blind_smoke_rag_llm_v3_summary.csv`
- `docs/earnings22_v3_blind_smoke_rag_v3_error_analysis.md`
- `docs/earnings22_v3_blind_smoke_ablation_v3.md`

当前结果：

- 只完成 1 个 v3 blind smoke 文件。
- tiny/base 都没有触发候选词修正。
- WER 和 term F1 都没有变化。
- `api_used=false`，说明没有调用 LLM。

怎么解释：

- 这只能证明 v3 pipeline 能跑通。
- 不能证明 v3 性能好。
- 下一步必须找 6-12 个新的 blind Earnings-22 文件，并且里面要有真实 ASR 易错词候选。

### 4.5 AMI held-out 和 interruption 标签流程

已经完成：

- `scripts/download_meeting_subset.py` 增加：
  - `--meeting-ids`
  - `--max-clips-per-meeting`
- 生成了 AMI held-out manifest：
  - `data/manifests/english_meeting_heldout_real.csv`
  - `data/manifests/english_meeting_heldout_real_checksums.csv`
- 生成了 interruption 候选：
  - `data/reference/public/english_meeting_heldout/interruption_label_candidates.csv`
- 增加了候选生成脚本：
  - `scripts/generate_interruption_label_candidates.py`
- 增加了标签验证脚本：
  - `scripts/validate_interruption_labels.py`
- 增加了说明文档：
  - `docs/interruption_labeling.md`

当前情况：

- AMI held-out 覆盖 ES2002a/b/c/d 共 24 条。
- interruption candidates 目前有 10 条。
- 2026-06-21 人工审核确认 10/10 候选窗口包含 interruption event。
- 这只能支持 event-level candidate precision = 1.0；不能支持 recall/F1。
- `interrupter` / `interrupted` speaker pair 继承自 AMI reference timing，没有由人工通过音色独立确认。

怎么解释：

- 可以说系统能生成高精度 interruption 候选，减少人工从整段会议里寻找事件的成本。
- 不能说系统完整检测了所有打断，也不能说人工确认了匿名 speaker pair 的声纹身份。

### 4.6 pyannote diarization DER/JER 通道

已经完成：

- 标准 DER/JER wrapper：
  - `experiments/metrics/standard_diarization_metrics.py`
- 自动 pyannote benchmark：
  - `experiments/run_pyannote_diarization_benchmark.py`
- 结果文档：
  - `docs/pyannote_diarization_benchmark.md`

运行过：

```bash
python experiments/run_pyannote_diarization_benchmark.py \
  --manifest data/manifests/english_meeting_heldout_real.csv \
  --output experiments/results/pyannote_diarization_heldout_real.csv \
  --summary-output experiments/results/pyannote_diarization_heldout_summary_real.csv \
  --predictions-dir outputs/diarization/pyannote_heldout_real
```

当前结果：

- 24 rows 全部 `ok`。
- mean DER = 0.106035
- mean JER = 0.307202
- DER skip-overlap = 0.081325
- JER skip-overlap = 0.209765
- overlap F1 = 0.490214
- mean RTF = 0.692477

怎么解释：

- 现在可以写 pyannote 自动 diarization 的真实 24-clip AMI held-out 数值。
- 仍然要说明这是小规模 held-out，不是完整 AMI corpus benchmark。

### 4.7 multilingual ASR

已经完成：

- FLEURS 英语、法语、中文小规模 ASR。
- 中文用 CER。
- 结果已经在 `asr_benchmark_summary_real.csv` 里。

没完成：

- 普通话多人会议。
- Mandarin meeting diarization/overlap。

当前 blocker：

- AISHELL-4 archive 很大，之前没有完整下载。
- AliMeeting 没有稳定的小文件直链自动路径。

怎么解释：

- 现在可以写“multilingual ASR smoke evaluation”。
- 还不能写“multilingual meeting evaluation”。

### 4.8 mobile / whisper.cpp Level 1

已经完成两个层级：

第一层：local CPU proxy

- `experiments/benchmark_mobile_asr.py`
- `experiments/results/v1/mobile_asr.csv`
- `docs/mobile_asr_tradeoff.md`

这个结果来自 faster-whisper CPU int8，用来近似“小模型本地运行 trade-off”。

第二层：真实 whisper.cpp 通道

- `experiments/benchmark_whisper_cpp.py`
- `docs/whisper_cpp_level1.md`
- `experiments/results/v1/whisper_cpp_mobile_level1.csv`
- `experiments/results/v1/whisper_cpp_mobile_level1_summary.csv`

当前结果：

- 76 行全部 `ok`，skipped=0。
- tiny/base 各 38 条，覆盖 formal eval 中的 FLEURS en/fr/zh-CN 和 AMI en。
- 这次 AISHELL-4 扩到 50 条后还没有重新跑 whisper.cpp。
- AMI en 结果：
  - base WER 0.351123，cleaned WER 0.280012，RTF 0.056347
  - tiny WER 0.382383，cleaned WER 0.313909，RTF 0.030548
- FLEURS fr/zh-CN 错误率很高，说明当前默认 whisper.cpp 命令不是强多语言 baseline。

怎么解释：

- mobile proxy 可以作为早期工程指标。
- 真实 Level 1 whisper.cpp 还没完成。
- 论文不能把 proxy 写成真实手机结果。

### 4.9 测试和质量检查

已经跑过并通过：

```bash
python -m py_compile \
  experiments/run_pyannote_diarization_benchmark.py \
  experiments/benchmark_whisper_cpp.py \
  experiments/evaluate_rag_llm_correction.py \
  experiments/evaluate_earnings22_ablation.py \
  scripts/download_meeting_subset.py \
  scripts/generate_interruption_label_candidates.py \
  scripts/validate_interruption_labels.py \
  experiments/metrics/standard_diarization_metrics.py
```

```bash
python -m pytest \
  tests/test_diarization_research_tracks.py \
  tests/test_earnings22_finance_correction.py \
  tests/test_dataset_acquisition.py \
  tests/test_speaker_overlap_baseline.py \
  tests/test_mobile_asr_tradeoff.py \
  -q
```

结果：30 passed。

```bash
python -m pytest \
  tests/test_asr_benchmark.py \
  tests/test_earnings22_heldout_reports.py \
  tests/test_real_error_and_term_audit.py \
  -q
```

结果：25 passed。

```bash
git diff --check
```

结果：通过。

---

## 5. 现在效果到底怎么样

### 5.1 可以比较确定地说

1. **baseline 在多人会议上确实更差。**

   FLEURS 英文 `base` WER 是 0.114374，而 AMI meeting `base` WER 是 0.398364。虽然样本少，但方向清楚：会议语音更难。

2. **系统已经能把普通 ASR 输出变成证据地图。**

   Workflow ablation 显示 full pipeline 能产生 speaker-labeled anchors、overlap anchors、events、audit trails。

3. **RAG 修专有词不是无脑有效。**

   在 `tiny` 上 term F1 从 0.888889 到 0.930556，是好信号。
   但在 `base` 上 term F1 从 0.972222 掉到 0.930556，说明强模型本来对，RAG/LLM 可能错误介入。

4. **安全门控是必须的，不是装饰。**

   v2 的假阳性直接证明：没有严格 gate，RAG 会变成“看见像专业词就想改”的风险源。

### 5.2 不能说过头的地方

不能说：

```text
TalkWeaver improves WER on all datasets.
```

因为现在 RAG v2 对 WER 没有明显提升，甚至略变差。

不能说：

```text
Our automatic diarization achieves good DER/JER.
```

因为 pyannote 没有 token，真实 DER/JER 还没跑出来。

不能说：

```text
We completed multilingual meeting evaluation.
```

因为目前多语言只有 FLEURS，不是多人会议。

不能说：

```text
We benchmarked mobile deployment.
```

只能说：

```text
We implemented a Level 1 whisper.cpp benchmark path, and currently report a local CPU proxy until true whisper.cpp/device runs are available.
```

---

## 6. 离最终论文成品还差什么

### 6.1 最重要缺口一：自动 diarization 真实 DER/JER

现在状态：

- 脚本完成。
- 结果全 skipped。
- blocker 是 `HF_TOKEN`。

需要做：

1. 申请或使用 Hugging Face token。
2. 确认已接受 pyannote 模型 license / terms。
3. 在 `.env` 中设置 `HF_TOKEN`，值填写你自己的 Hugging Face access token。

4. 运行：

```bash
python experiments/run_pyannote_diarization_benchmark.py \
  --manifest data/manifests/english_meeting_heldout_real.csv \
  --output experiments/results/pyannote_diarization_heldout_real.csv \
  --summary-output experiments/results/pyannote_diarization_heldout_summary_real.csv \
  --predictions-dir outputs/diarization/pyannote_heldout_real
```

5. 打开结果：

```bash
sed -n '1,40p' experiments/results/pyannote_diarization_heldout_summary_real.csv
```

判定标准：

- 如果 `metric_status=ok`，才能写真实 DER/JER。
- 如果还是 `skipped`，只能写工具实现，不能写效果。

论文写法：

- 有真实结果时：写 DER/JER 表格，并分析 overlap 对 diarization 的影响。
- 没真实结果时：把 pyannote 作为 future work 或 implementation-ready baseline，不要放主结果。

### 6.2 最重要缺口二：更多 held-out AMI 会议

现在状态：

- 目标是多会议 held-out。
- 当前自动下载只稳定得到 ES2002a 的 6 条。

需要做：

尝试继续扩大：

```bash
python scripts/download_meeting_subset.py \
  --max-clips 24 \
  --max-clips-per-meeting 6 \
  --meeting-ids ES2002a ES2002b ES2002c ES2002d \
  --output-root data/raw/public/english_meeting_heldout \
  --reference-root data/reference/public/english_meeting_heldout \
  --manifest-out data/manifests/english_meeting_heldout_real.csv
```

如果还是只能拿 ES2002a：

- 需要手动检查 AMI 文件结构；
- 或者换别的公开会议数据集；
- 或者明确写成 AMI ES2002a small held-out。

论文最低可接受：

- 8 条原 formal AMI + 6 条 held-out AMI 可以作为 demo-level evidence。

更好目标：

- 至少 20-30 条 AMI clips；
- 覆盖 3-4 个 meeting IDs；
- 每条 20-60 秒；
- 有 reference speaker turns。

### 6.3 最重要缺口三：人工 interruption 标签

现在状态：

- 自动生成了候选。
- 还没人工确认。

候选文件：

```text
data/reference/public/english_meeting_heldout/interruption_label_candidates.csv
```

当前例子：

```text
ami_es2002a_05, 5.930-7.810, SPEAKER_B interrupts SPEAKER_A, label=interruption
ami_es2002a_05, 5.930-6.390, SPEAKER_B interrupts SPEAKER_D, label=interruption
```

已完成：

1. 打开对应音频。
2. 听候选时间附近。
3. 把 10 条 label 改成 `interruption`。
4. 填 `annotator=human_1`。
5. 运行验证和评估：

```bash
python scripts/validate_interruption_labels.py \
  --labels data/reference/public/english_meeting_heldout/interruption_label_candidates.csv \
  --manifest data/manifests/english_meeting_heldout_real.csv

python experiments/evaluate_interruption_labels.py \
  --labels data/reference/public/english_meeting_heldout/interruption_label_candidates.csv \
  --output experiments/results/interruption_label_summary_heldout_real.csv
```

后续还要补：

- interruption scoring script；
- precision/recall/F1；
- 每类错误案例。

论文写法：

- 有人工标签：可以写 interruption detection evaluation。
- 没人工标签：只能写 interruption candidate generation and annotation protocol。

### 6.4 最重要缺口四：RAG v3 新 blind 评测

现在状态：

- v3 gate 实现了。
- 只有 1 个 smoke blind 文件。
- 这个文件没有触发候选词，不能证明 v3 改进。

需要做：

1. 准备 6-12 个新的 Earnings-22 blind 文件。
2. 确保不要用之前调 v2 的文件。
3. 确保 glossary 不来自 reference transcript。
4. 跑 ASR：

```bash
python experiments/run_asr_benchmark.py \
  --manifest data/manifests/earnings22_v3_blind_6x180.csv \
  --models tiny base \
  --device cpu \
  --compute-type int8 \
  --vad-filter true \
  --output experiments/results/asr_benchmark_earnings22_v3_blind_6x180.csv \
  --predictions-dir experiments/results/asr_predictions_earnings22_v3_blind_6x180
```

5. 跑 v3 RAG：

```bash
python experiments/evaluate_rag_llm_correction.py \
  --input experiments/results/asr_benchmark_earnings22_v3_blind_6x180.csv \
  --glossary data/controlled_terms/earnings22_multi_context_terms.json \
  --output experiments/results/earnings22_v3_blind_rag_llm_v3.csv \
  --summary-output experiments/results/earnings22_v3_blind_rag_llm_v3_summary.csv \
  --markdown-output docs/earnings22_v3_blind_rag_v3_error_analysis.md \
  --gate-version v3
```

6. 跑 ablation：

```bash
python experiments/evaluate_earnings22_ablation.py \
  --asr-input experiments/results/asr_benchmark_earnings22_v3_blind_6x180.csv \
  --llm-input experiments/results/earnings22_v3_blind_rag_llm_v3.csv \
  --glossary data/controlled_terms/earnings22_multi_context_terms.json \
  --output experiments/results/earnings22_v3_blind_ablation_v3.csv \
  --summary-output experiments/results/earnings22_v3_blind_ablation_v3_summary.csv \
  --markdown-output docs/earnings22_v3_blind_ablation_v3.md \
  --gate-version v3
```

判定标准：

- v3 不一定要大幅降低 WER。
- 关键是：
  - false positive 更少；
  - 不乱改 base 已经对的地方；
  - tiny 上能救一部分真实专有词；
  - rejected/no-op/needs_review 有合理比例。

论文写法：

- 如果 v3 提升 tiny 且不伤 base：写成主要结果。
- 如果 v3 很保守、几乎不改：写成 hallucination-safe correction，强调 precision over recall。
- 如果 v3 仍然乱改：说明 RAG correction 只能作为 human-in-the-loop 辅助，不作为自动改写。

### 6.5 普通话/多语言会议

现在状态：

- FLEURS 多语言 ASR 有了。
- 普通话多人会议已有 AISHELL-4 固定 benchmark 子集。
- `data/manifests/mandarin_meeting_real.csv` 仍保留 12 条 sanity/formal-manifest 子集。
- 新主结果应引用 `data/manifests/aishell4_benchmark_60x20.csv`：
  - 60 条 20s clips，总计 1200s
  - 覆盖 AISHELL-4 test split 本地 20 个 recordings
  - 每个 recording 最多 3 条，避免只评一个录音
  - 29 条多说话人可评分片段
  - 10 条 reference overlap clips
- ASR 60 条结果：
  - tiny CER=0.648291，RTF=0.063637
  - base CER=0.536940，RTF=0.071076
  - small CER=0.481842，RTF=0.136653
- pyannote diarization 29 条多说话人结果：
  - mean DER=0.326501
  - mean JER=0.712577
  - DER skip-overlap=0.326214
  - JER skip-overlap=0.693182
  - overlap F1=0.261905
  - mean RTF=0.504550

复现命令：

```bash
python scripts/download_mandarin_meeting_subset.py \
  --dataset aishell4 \
  --split test \
  --aishell4-extracted-root data/raw/public/mandarin_meeting/_source/aishell4_test \
  --max-clips 60 \
  --max-clips-per-recording 3 \
  --clip-duration-seconds 20 \
  --output-root data/raw/public/aishell4_benchmark \
  --reference-root data/reference/public/aishell4_benchmark \
  --manifest-out data/manifests/aishell4_benchmark_60x20.csv
```

```bash
python experiments/run_asr_benchmark.py \
  --manifest data/manifests/aishell4_benchmark_60x20.csv \
  --models tiny base small \
  --device cpu \
  --compute-type int8 \
  --vad-filter true \
  --output experiments/results/asr_benchmark_aishell4_60x20_real.csv \
  --predictions-dir experiments/results/asr_predictions_aishell4_60x20_real
```

限制：

- 当前结果是固定 60-clip subset，不是完整 AISHELL-4 test-set benchmark。
- 60 条覆盖全部 20 个 test recordings，但每个 recording 只取 3 条 20s clips。
- raw audio、archive 和 extracted source 都不提交 Git。

### 6.6 最重要缺口六：手机真机 mobile benchmark

现在状态：

- proxy 有结果。
- whisper.cpp Level 1 local-machine benchmark 已经跑通。
- 手机真机延迟、内存、电量仍未测。

已完成：

1. 编译或安装 whisper.cpp。
2. 下载模型到：

```text
models/whisper.cpp/ggml-tiny.bin
models/whisper.cpp/ggml-base.bin
```

3. 确认命令可运行：

```bash
whisper-cli --help
```

4. 运行：

```bash
python experiments/benchmark_whisper_cpp.py \
  --manifest data/manifests/formal_eval_real.csv \
  --output experiments/results/v1/whisper_cpp_mobile_level1.csv \
  --summary-output experiments/results/v1/whisper_cpp_mobile_level1_summary.csv
```

判定标准：

- 如果 `status=ok`，可以写 Level 1 local whisper.cpp result。
- 如果在真实手机上跑，还需要记录：
  - device name；
  - CPU/GPU/NPU；
  - model quantization；
  - battery/thermal notes；
  - runtime；
  - RTF。

论文写法：

- 本机 whisper.cpp：Level 1 local deployment trade-off。
- 真手机：mobile deployment benchmark。
- faster-whisper CPU proxy：只能叫 proxy，不能叫 mobile result。

### 6.7 最重要缺口七：系统错误分析

现在有一些错误分析，但还不够像完整论文。

还要补：

- ASR 错误类别占比：
  - 公司名；
  - 财经术语；
  - 数字/单位；
  - filler/disfluency；
  - speaker overlap；
  - pronunciation/accent；
  - RAG false positive；
  - RAG false negative。
- 每一类至少 2-3 个代表样本。
- 说明为什么错：
  - acoustic ambiguity；
  - overlap；
  - baseline model too small；
  - domain term rare；
  - glossary candidate too broad；
  - LLM over-normalization。

最终论文最好有一张这样的表：

| Error type | Count | Example raw | Reference | Our action | Outcome |
|---|---:|---|---|---|---|
| Company name | 8 | ... | ... | RAG candidate | fixed/rejected |
| Number/unit | 5 | ... | ... | v3 gate no-op | safe |
| Overlap | 6 | ... | ... | needs_review | safe |

---

## 7. 文件地图

### 7.1 后端核心

- `backend/asr.py`
  - ASR baseline。
  - faster-whisper / mock。

- `backend/diarization.py`
  - pyannote diarization / mock。

- `backend/alignment.py`
  - word/segment 对齐 speaker turns。

- `backend/overlap.py`
  - overlap / interruption-like event detection。

- `backend/rag.py`
  - domain term retrieval。

- `backend/llm_correction.py`
  - structured correction。

- `backend/term_verifier.py`
  - RAG/LLM correction safety helper。

### 7.2 实验脚本

- `experiments/run_asr_benchmark.py`
  - 跑 ASR WER/CER/RTF。

- `experiments/run_workflow_ablation.py`
  - 跑 TalkWeaver evidence-map ablation。

- `experiments/evaluate_rag_llm_correction.py`
  - 跑 RAG + LLM correction。
  - 支持 `--gate-version v2|v3`。

- `experiments/evaluate_earnings22_ablation.py`
  - Earnings-22 ablation。

- `experiments/run_pyannote_diarization_benchmark.py`
  - 自动 pyannote DER/JER。

- `experiments/benchmark_mobile_asr.py`
  - faster-whisper CPU proxy mobile-style benchmark。

- `experiments/benchmark_whisper_cpp.py`
  - true whisper.cpp Level 1 benchmark。

### 7.3 数据脚本

- `scripts/download_common_voice_subset.py`
  - FLEURS/Common Voice style multilingual subset。

- `scripts/download_meeting_subset.py`
  - AMI meeting subset。

- `scripts/download_earnings22_subset.py`
  - Earnings-22 subset。

- `scripts/generate_interruption_label_candidates.py`
  - 自动生成 interruption 标注候选。

- `scripts/validate_interruption_labels.py`
  - 验证人工标签格式。

### 7.4 关键文档

- `PROJECT_REPORT.md`
  - 当前主报告草稿。

- `docs/asr_benchmark.md`
  - ASR baseline 结果。

- `docs/workflow_ablation.md`
  - evidence-map ablation。

- `docs/earnings22_final_blind_rag_v2_error_analysis.md`
  - RAG v2 final blind 分析。

- `docs/earnings22_v3_blind_smoke_rag_v3_error_analysis.md`
  - RAG v3 smoke 分析。

- `docs/pyannote_diarization_benchmark.md`
  - pyannote benchmark 状态。

- `docs/interruption_labeling.md`
  - 人工 interruption 标注说明。

- `docs/mobile_asr_tradeoff.md`
  - CPU proxy mobile trade-off。

- `docs/whisper_cpp_level1.md`
  - true whisper.cpp Level 1。

---

## 8. 防数据泄露规则

这部分非常重要，否则 RAG 实验会被质疑。

不能做：

- 不能从 reference transcript 里抽出正确词，再放进 glossary，然后说模型修对了。
- 不能把 reference transcript 发给 LLM。
- 不能根据 held-out 错误样本现场手写规则，然后在同一批 held-out 上报最终效果。
- 不能把常见词强行映射成公司名。
- 不能为了 WER 降低，让 LLM 大段重写句子。

可以做：

- 使用公开、预先定义的 domain glossary。
- 使用公司官方名称、公开财报词表、课程术语词表。
- dev set 上观察错误，冻结 v3 方法。
- 新 blind set 上只跑一次最终结果。
- 报告 accepted/rejected/no-op/needs_review。
- 报告 false positive 和 false negative。

最终论文必须写清楚：

```text
The reference transcript is used only for evaluation, not for retrieval or prompting.
```

---

## 9. 最终论文建议结构

### Abstract

写清楚：

- 多人会议 ASR 不只是转写问题，还有 speaker、overlap、interruption、term hallucination。
- TalkWeaver 提出 evidence-grounded conversation map。
- RAG 是专有词辅助模块。
- 实验包括 ASR、workflow ablation、RAG term recovery、diarization/mobile/multilingual tracks。

### Introduction

讲动机：

- 现实会议有多人说话、打断、重叠、专业词。
- 单纯 ASR 给一段文字，不知道谁说的、哪里不确定、哪里被 LLM 改过。
- 我们要做可审计的会议侦探系统。

### Related Work

对应：

- Whisper / faster-whisper；
- pyannote diarization；
- DiarizationLM；
- DM-ASR；
- TagSpeech；
- retrieval-augmented ASR correction；
- whisper.cpp/mobile ASR。

### Method

按模块写：

1. Temporal-anchor transcript。
2. Diarization-structured prompting。
3. Overlap-aware uncertainty control。
4. RAG domain term recovery。
5. LLM correction audit。
6. Mobile/latency trade-off。

### Experiments

至少包含：

- ASR baseline；
- evidence-map workflow ablation；
- Earnings-22 RAG term recovery；
- diarization DER/JER，如果跑通；
- interruption labels，如果标注完成；
- multilingual ASR；
- whisper.cpp Level 1，如果跑通。

### Results

结果要分 claim level：

| Track | Current claim level |
|---|---|
| ASR baseline | real small-subset, plus 24-clip AMI held-out |
| Pyannote DER/JER | real 24-clip AMI held-out |
| Automatic evidence map | real ASR + automatic pyannote on 24 AMI clips |
| Workflow ablation | reference-assisted real small-subset |
| RAG v2 | real held-out, mixed result |
| RAG v3 | real 6-file blind, conservative/no WER gain |
| interruption | 10 human-confirmed event-level candidates, precision only |
| multilingual | ASR-level FLEURS plus AISHELL-4 60-clip Mandarin meeting subset |
| whisper.cpp | real Level 1 local-machine benchmark, 76 ok rows, not yet rerun on AISHELL-4 |
| mobile proxy | 100-row local CPU proxy only |

### Error Analysis

必须诚实写：

- RAG 会 false positive。
- base 模型已经识别正确时，额外 correction 可能伤害结果。
- overlap 区域应该标记不确定，而不是强行改。
- Mandarin meeting 已有 AISHELL-4 60-clip 固定子集结果，但不是完整 test-set benchmark。
- true mobile 还不完整。

### Limitations

可以写：

- small subset；
- public dataset access constraints；
- missing full automatic diarization result if HF_TOKEN not available；
- RAG glossary coverage；
- API LLM cost/reproducibility；
- no full native mobile app。

---

## 10. 下一步优先级

### P0：先让论文主结果可信

1. 扩大 AMI held-out 到至少 20 clips。
2. 配置 `HF_TOKEN`，跑 pyannote DER/JER。
3. 完成 6-12 文件 Earnings-22 v3 blind。
4. 写系统错误分析表。

### P1：补齐 mandatory tracks

1. 人工标 interruption labels。
2. 找普通话多人会议小子集。
3. 编译 whisper.cpp，跑 Level 1 benchmark。

### P2：论文和视频 polish

1. 更新 `PROJECT_REPORT.md`。
2. 更新 result charts。
3. 在 Streamlit UI 展示：
   - timeline；
   - overlap；
   - RAG rescue；
   - correction audit；
   - hallucination watchdog；
   - mobile trade-off chart。
4. 录 demo 视频。

---

## 11. 建议的一周执行计划

### Day 1：冻结当前 baseline

- 确认所有现有 tests 通过。
- 把当前结果表格整理进 `PROJECT_REPORT.md`。
- 不再修改 v2。

### Day 2：pyannote

- 配置 `HF_TOKEN`。
- 跑 `run_pyannote_diarization_benchmark.py`。
- 如果失败，记录原因。
- 如果成功，补 DER/JER 表格。

### Day 3：AMI 和 interruption

- 扩大 AMI clips。
- 重新生成 interruption candidates。
- 人工听至少 20 个 candidate。
- 跑 label validation。

### Day 4：RAG v3 blind

- 准备新的 Earnings-22 blind set。
- 跑 ASR。
- 跑 RAG v3。
- 跑 ablation。
- 写 false positive/false negative 分析。

### Day 5：multilingual meeting / fallback

- 尝试 AliMeeting/AISHELL-4 小子集。
- 如果拿不到，明确记录 blocker。
- 至少保留 FLEURS multilingual ASR 作为 mandatory multilingual smoke。

### Day 6：whisper.cpp

- 编译 whisper.cpp。
- 下载 tiny/base ggml 模型。
- 跑 benchmark。
- 更新 mobile 文档。

### Day 7：论文整理

- 更新 `PROJECT_REPORT.md`。
- 更新 `BLOG_ARTICLE.md`。
- 更新 `docs/video_script.md`。
- 生成 charts。
- 确认 `.env`、raw audio、大模型没有被 git track。

---

## 12. 最终成品判定标准

一个比较完整的最终论文版本，至少应该满足：

- 有真实 ASR baseline 表。
- 有多人会议比单人/单句更难的证据。
- 有 TalkWeaver evidence-map ablation。
- 有 RAG v3 blind 结果，哪怕提升不大，也要证明更安全。
- 有系统错误分析。
- 有至少一个 diarization DER/JER 结果，或者非常清楚地说明为什么没有。
- 有 interruption 标注协议和部分人工标签。
- 有多语言结果。
- 有 mobile/whisper.cpp trade-off，或者明确 proxy 与 true benchmark 的区别。
- 所有 claim 都能追溯到 CSV、manifest、代码和文档。

---

## 13. 当前最诚实的论文结论

如果现在立刻写论文，最稳的结论应该是：

```text
TalkWeaver demonstrates an evidence-grounded pipeline for chaotic meeting ASR:
it transforms raw ASR into temporal speaker anchors, overlap-aware events,
retrieval-supported term correction candidates, and auditable LLM edits.
On a small public-dataset subset, meeting speech is substantially harder than
clean single-speaker utterances. RAG-based term recovery shows promise on weaker
ASR models but can introduce false positives, motivating stricter v3 gating and
human-in-the-loop review for uncertain changes.
```

不要把当前项目吹成 state-of-the-art。更好的定位是：

```text
我们不是追求最高分，而是证明一个面向真实混乱会议的、证据可追踪的 ASR+diarization+RAG+LLM 系统设计。
```
