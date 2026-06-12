# Baseline Feasibility

> Last verified: June 12, 2026

This document records whether recent paper systems and mobile runtimes can be
used as TalkWeaver baselines. It is a planning audit, not evidence that any
model has already been downloaded or executed.

Only paper-provided links, official repositories, official model pages, and
official runtime documentation were used. The current workspace is Linux
`x86_64` with approximately 15 GiB RAM. GPU access is not currently available
to the sandbox, Apple development tools are absent, and no Android device is
connected.

## Decision Labels

- **run:** prepare a controlled small-sample inference baseline after explicit
  approval for dependencies and model downloads;
- **proxy:** implement or retain the paper's core idea using TalkWeaver
  modules, without claiming that the original system was executed;
- **literature only:** use the work for motivation or qualitative comparison
  unless the missing hardware or release status changes.

The baseline levels and claim rules are defined in
`PRD2.md` under **Paper Baselines and Reproduction Policy**.

## Feasibility Matrix

| Paper / baseline | Official repo or package | Pretrained model available | Expected dependency risk | Expected hardware requirement | Can run inference? | Can train? | Integration output format | Decision: run / proxy / literature only | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DiarizationLM | [Google speaker-id / DiarizationLM](https://github.com/google/speaker-id/tree/master/DiarizationLM); `pip install diarizationlm` | [Google DiarizationLM-8b-Fisher-v2](https://huggingface.co/google/DiarizationLM-8b-Fisher-v2), including GGUF; model license is Llama 3, repository is Apache-2.0 | Medium-high: package utilities are light, but model inference adds `llama.cpp` or `llama-cpp-python`, tokenizer/model compatibility, and a multi-GB download | Q4 8B inference should fit a machine with adequate CPU RAM; GPU improves speed but is not strictly required | Yes, conditionally. Run only a short approved sample and label it `small-scale baseline run` | No project training. Official fine-tuning paths exist, but training is costly and does not reproduce the internal PaLM 2 setup | Convert word/speaker sequences or diarized text into TalkWeaver temporal anchors plus WDER/cpWER records | **run** | Start with official formatting, TPST, and metric utilities. Model inference requires explicit approval for the large download. Never call this a full paper reproduction. |
| TagSpeech | [AudenAI/Auden TagSpeech example](https://github.com/AudenAI/Auden/tree/main/examples/tagspeech), Apache-2.0 | [TagSpeech-AMI](https://huggingface.co/AudenAI/TagSpeech-AMI) and [TagSpeech-Alimeeting](https://huggingface.co/AudenAI/TagSpeech-Alimeeting), Apache-2.0 | High: PyTorch, Auden framework, dual encoders, Qwen2.5-7B backend, custom parsing, checkpoint download | A capable CUDA GPU is strongly preferred; the current CPU-only 15 GiB environment is not a reliable target for this 7B multimodal stack | Not in the current environment. Reassess if an approved GPU host and storage are available | No. Official training scripts exist, but AMI/AliMeeting training and 7B-model tuning are outside scope | Official XML-like output parsed to JSON segments with start, end, text, and speaker ID, then mapped to temporal anchors | **proxy** | Preserve the temporal-anchor adaptation now. If suitable hardware becomes available, promote to a Level A short inference run, not full reproduction. |
| DM-ASR | [arXiv:2604.22467](https://arxiv.org/abs/2604.22467); no paper-provided official repository or pretrained checkpoint was found in this audit | No verified official checkpoint found | Very high: trained Speech-LLM, diarization inputs, speaker/time query construction, optional word timestamp tokens | Multi-GPU or high-memory GPU training/inference is likely; exact runnable requirements are not published in an official release | No official runnable release verified | No | TalkWeaver speaker-time prompts and temporal anchors approximate the query decomposition | **proxy** | Label the adaptation `DM-ASR-inspired speaker-time conditioned correction`. Do not claim DM-ASR execution or reproduction. |
| Diarization-Aware Multi-Speaker ASR via LLMs | [arXiv:2506.05796](https://arxiv.org/abs/2506.05796); the paper links the [MLC-SLM challenge baseline](https://github.com/mubingshen/MLC-SLM-Baseline/tree/main), not an official release of the proposed model | No verified official checkpoint for the proposed system | Very high: semantic and speaker encoders, diarization triplets, Qwen2.5-3B fine-tuning, synthetic and meeting datasets | High-memory CUDA training and substantial data preparation | No official inference path for the proposed model was verified | No | Map speaker embedding/time-triplet ideas to TalkWeaver anchor evidence and per-anchor evaluation | **proxy** | The challenge baseline repository is not equivalent to the paper's proposed model. Treat it separately if evaluated. |
| Retrieval Augmented Correction of Named Entity ASR Errors | [arXiv:2409.06062](https://arxiv.org/abs/2409.06062); no paper-provided official code or pretrained correction model was found | No verified official checkpoint found | Medium-high: entity index, query generation, semantic or acoustic-neighbor embeddings, adapted LLM | Retrieval itself can run on CPU; the paper's adapted LLM and acoustic-neighbor setup require unavailable assets | No exact official system; TalkWeaver can run a controlled approximation | No | Candidate term JSON with scores, source glossary entries, correction decisions, and TER metrics | **proxy** | Compare TF-IDF, fuzzy, phonetic-like, and fused retrieval. Do not describe text-derived phonetic matching as the paper's acoustic-neighbor embeddings. |
| whisper.cpp mobile ASR baseline | [ggml-org/whisper.cpp](https://github.com/ggml-org/whisper.cpp), MIT | Official conversion/download scripts provide ggml Whisper models and quantized variants | Low-medium: CMake/C++ toolchain, FFmpeg conversion, model downloads, platform-specific acceleration | CPU-only supported; tiny/base models fit modest RAM. Real-device measurements require the named device | Yes. This is the mandatory Level 1 mobile-style baseline | No training required or planned | Normalize CLI JSON/text output into raw ASR segments; record WER/CER, RTF, latency, model size, and peak memory | **run** | First run tiny/base quantized variants on the frozen mobile subset. Distinguish desktop mobile-style measurements from real-device results. |
| WhisperKit / Core ML Apple baseline | [argmaxinc/argmax-oss-swift](https://github.com/argmaxinc/argmax-oss-swift), MIT; WhisperKit is included as a Swift package product | Package can download compatible models through its model-selection workflow | Medium-high: macOS 14+, Xcode 16+, Swift Package Manager, Core ML models, Apple Silicon/device testing | Compatible Mac, iPhone, or iPad required for meaningful measurements | Not in the current Linux environment | No training planned | Export transcription text/timestamps and device metrics to the mobile benchmark CSV schema | **literature only** | Should-have when compatible Apple hardware is available. If hardware is provided, change the decision to `run` and label it a small-scale Apple baseline. |
| ONNX Runtime or Android mobile ASR baseline | [ONNX Runtime mobile documentation](https://onnxruntime.ai/docs/tutorials/mobile/) and [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime), MIT | Runtime packages are available, but no single paper-provided TalkWeaver Whisper ONNX checkpoint/pipeline is selected | High: model export or model selection, tokenizer/decoder implementation, Android build, execution provider compatibility | Android device or emulator plus Android toolchain; device-specific memory and accelerator behavior | Not yet. Runtime feasibility exists, but a validated ASR model path and device are missing | No training planned | Adapt output to the same mobile ASR CSV and transcript JSON used by whisper.cpp | **literature only** | Could-have. Promote to `proxy` or `run` only after selecting a license-compatible ONNX ASR model and confirming Android hardware/tooling. |

## Immediate Feasibility Decisions

1. **Mandatory run:** whisper.cpp quantized tiny/base comparison.
2. **Planned small-scale official run after approval:** DiarizationLM
   formatting/metrics first, then optional Q4 model inference.
3. **Proxy now, possible official inference later:** TagSpeech, because the
   official code and checkpoints exist but current hardware is unsuitable.
4. **Proxy only with current releases:** DM-ASR, Diarization-Aware
   Multi-Speaker ASR via LLMs, and retrieval-augmented named-entity
   correction.
5. **Hardware-dependent extensions:** WhisperKit/Core ML on Apple hardware and
   ONNX Runtime on an Android target.

## Baseline Intake Checklist

Before running or integrating a baseline, create a run record containing:

- official repository and model URL;
- commit SHA, tag, package version, or model revision;
- code and model licenses;
- expected and actual download size;
- dependency environment;
- input clip IDs;
- hardware and operating system;
- expected and actual runtime;
- output conversion path;
- baseline level and claim label;
- failure reason when blocked.

Do not download large models, start training, or run long benchmarks until the
user explicitly approves the expected storage, runtime, and hardware use.
