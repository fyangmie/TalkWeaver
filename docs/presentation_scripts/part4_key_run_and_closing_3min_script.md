# Part 4 - Key Run and Final Closing, 2 to 3 Minutes

用途：放在代码 walkthrough 后面，作为最后一段短演示和视频结尾。
目标：证明 TalkWeaver 不只是 PPT、网页和静态文档，而是真的能运行 pipeline，并产出可审计 artifacts。

注意：这一段不要跑大实验，不要展示 `.env`，不要展示 API key / HF token。只跑 mock pipeline 或打开已有 artifact。

## 0:00-0:20 - 中文操作指导：打开终端

**你要做什么：**

在 VS Code 或系统终端中进入 TalkWeaver 仓库根目录。

```bash
cd ~/机器学习/TalkWeaver
```

如果你已经在仓库根目录，就不用输入这句。

**嘴里用英语说：**

To finish the demo, I want to show that TalkWeaver is not only a presentation
or a static website. The repository also contains a runnable pipeline that
produces auditable transcript artifacts. I will use mock mode here, because it
is deterministic, fast, and does not require private API keys or large model
downloads.

## 0:20-0:55 - 中文操作指导：运行轻量 pipeline

**你要输入：**

```bash
python scripts/run_pipeline.py --mock
```

如果这个命令因为环境问题失败，可以改用：

```bash
python scripts/run_pipeline.py --help
```

然后口头说明真实 pipeline 已经在实验文件和网页 demo 中展示，这里只展示可复现入口。

**嘴里用英语说：**

This command runs the TalkWeaver pipeline in mock mode. Mock mode is not used
as a real experimental claim. It is a reproducibility and demonstration mode.
It lets us test the full data flow: preprocessing, ASR-style segments,
diarization-style speaker turns, overlap detection, temporal anchoring, RAG
term retrieval, constrained correction, and export.

## 0:55-1:30 - 中文操作指导：展示输出目录

**你要输入：**

```bash
ls outputs
```

然后继续输入：

```bash
find outputs -maxdepth 2 -type f | head -20
```

如果 Windows/PowerShell 不方便用 `find`，就直接在文件管理器里打开 `outputs/`，展示这些目录：

```text
outputs/transcripts/
outputs/diarization/
outputs/corrected_transcripts/
outputs/summaries/
outputs/exports/
```

**嘴里用英语说：**

The output is organized as artifacts, not just console text. This matters for
the research story. A meeting transcript should be inspectable after the model
runs. We save raw ASR output, diarization output, overlap warnings, temporal
anchor transcripts, RAG-enriched transcripts, corrected transcripts, summaries,
and a pipeline manifest.

## 1:30-2:10 - 中文操作指导：打开一个关键 artifact

**推荐打开：**

```bash
sed -n '1,80p' outputs/corrected_transcripts/mock_corrected_transcript.md
```

如果文件名不同，先输入：

```bash
find outputs/corrected_transcripts -type f
```

然后选择里面的 `.md` 文件打开。

也可以在 VS Code 里直接打开：

```text
outputs/corrected_transcripts/
```

**嘴里用英语说：**

Here we can see the audit trail. Each segment keeps the timestamp and speaker
assignment. It shows the raw text, the corrected text, retrieved terms, the
correction mode, and an audit note. This is the key design principle of
TalkWeaver: the system should not silently rewrite a meeting record. Every
change should have a traceable reason.

## 2:10-2:40 - 中文操作指导：连接回网页和论文

**你要做什么：**

切回网页或 VS Code 中的 `webapp/app.py`，不用再运行命令。

如果切回网页，指向 EvidenceMap、Before/After correction、timeline、results。

**嘴里用英语说：**

The web interface is built on the same idea. It turns these backend artifacts
into a user-facing EvidenceMap. The paper then evaluates the same components:
ASR difficulty on public data, speaker and overlap evidence, RAG term rescue,
and correction safety. So the project is connected end to end: research
question, backend pipeline, frontend evidence map, and experimental audit.

## 2:40-3:10 - Final Video Closing

**嘴里用英语说：**

To conclude, TalkWeaver does not claim that automatic meeting transcription is
solved. In fact, our negative results are part of the contribution: chaotic
meetings, overlapping speech, domain terms, and LLM post-processing can still
produce unsafe or unsupported edits.

Our main contribution is a more careful workflow. TalkWeaver converts chaotic
multi-speaker speech into an evidence-grounded conversation map. It shows what
was said, who said it, when it happened, where uncertainty appears, and why a
correction should or should not be trusted.

That is why we describe TalkWeaver as an AI Meeting Detective: it does not only
write a transcript; it helps users investigate the transcript.

## Emergency Short Version

如果视频时间快超了，只讲这一版，约 45 秒：

**你要输入：**

```bash
python scripts/run_pipeline.py --mock
```

**嘴里用英语说：**

I will close by running the lightweight mock pipeline. This does not claim real
model performance, but it proves the system architecture is executable. The
pipeline produces raw ASR artifacts, speaker-time evidence, overlap warnings,
RAG-enriched transcript segments, corrected transcript files, and a manifest.
The key idea is traceability: TalkWeaver keeps the raw text, corrected text,
timestamps, speakers, retrieved terms, and audit notes together. So the final
message of this project is simple: chaotic meeting transcription should not be
treated as only text generation. It should be treated as evidence-grounded
transcript auditing.
