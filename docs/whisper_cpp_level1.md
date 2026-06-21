# Whisper.cpp Level 1 Benchmark

## Purpose

This track replaces the earlier mobile-style proxy with a real local
`whisper.cpp` benchmark path. It is still Level 1 local-machine evidence, not
a phone measurement.

## Command

```bash
python experiments/benchmark_whisper_cpp.py \
  --manifest data/manifests/formal_eval_real.csv \
  --output experiments/results/v1/whisper_cpp_mobile_level1.csv \
  --summary-output experiments/results/v1/whisper_cpp_mobile_level1_summary.csv
```

Default model specs are:

```text
tiny=models/whisper.cpp/ggml-tiny.bin
base=models/whisper.cpp/ggml-base.bin
```

Use `--model-spec name=path` to provide local ggml/gguf model files.

## Current Result

Updated on 2026-06-21: `whisper.cpp` was built locally under `/tmp`, and
`ggml-tiny.bin` / `ggml-base.bin` were downloaded under the git-ignored
`models/whisper.cpp/` directory. The benchmark produced 76 `status=ok` rows
over `formal_eval_real.csv`:

```text
tiny: 38 rows
base: 38 rows
skipped: 0 rows
```

Summary:

```text
base AMI en: WER 0.351123, cleaned WER 0.280012, RTF 0.056347
tiny AMI en: WER 0.382383, cleaned WER 0.313909, RTF 0.030548
base FLEURS en: WER 0.133832, RTF 0.139889
tiny FLEURS en: WER 0.185512, RTF 0.071982
base FLEURS fr: WER 0.998636, RTF 0.134640
tiny FLEURS fr: WER 1.120450, RTF 0.093207
base FLEURS zh-CN: CER 1.997757, RTF 0.139627
tiny FLEURS zh-CN: CER 2.501653, RTF 0.057575
```

This is a Level 1 local-machine benchmark, not a phone-device measurement. The
very high French and Mandarin errors indicate that the current default
`whisper.cpp` command path is not yet a strong multilingual baseline; it is
still useful as a measured local deployment trade-off.
