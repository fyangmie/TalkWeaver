# Mobile ASR Trade-Off

## Purpose

The project requires a Level 1 mobile ASR trade-off track. The repository now
contains two measured local tracks: a conservative faster-whisper CPU int8
**mobile-style proxy** and a true local-machine `whisper.cpp` Level 1
benchmark. Neither is a phone-device measurement.

The proxy is still useful for the report because it uses the same frozen real
manifest, model variants, CPU int8 quantization, and WER/CER scoring as the
formal ASR baseline. It answers a narrower question:

```text
If we keep the runtime small, what accuracy/speed trade-off do tiny and base
show on the current public-data subset?
```

These local tracks do **not** answer:

```text
How fast is whisper.cpp on an iPhone, Android phone, or embedded device?
```

## Artifacts

```text
experiments/results/v1/mobile_asr.csv
experiments/results/v1/mobile_device_metadata.json
experiments/results/v1/whisper_cpp_mobile_level1.csv
experiments/results/v1/whisper_cpp_mobile_level1_summary.csv
```

Every row in `mobile_asr.csv` has:

```text
claim_level=mobile_style_proxy
true_mobile_device=false
runtime_environment=local_cpu_proxy
backend=faster-whisper
quantization=int8
```

The metadata file records the local platform and whether a `whisper.cpp`
command was found. The proxy metadata was produced before the later
`whisper.cpp` local build, so it should be interpreted as proxy metadata only.
The true `whisper.cpp` results are in `whisper_cpp_mobile_level1.csv`.

```text
true_mobile_device=false
row_count=76
```

## Command

```bash
python experiments/benchmark_mobile_asr.py \
  --source-asr-results experiments/results/asr_benchmark_real.csv \
  --output experiments/results/v1/mobile_asr.csv \
  --metadata-output experiments/results/v1/mobile_device_metadata.json \
  --mode auto
```

## Current Proxy Results

The proxy contains 100 rows: 50 real audio clips times two models.

| Model | Backend | Device | Quantization | Mean warm RTF | Mean mixed error |
| --- | --- | --- | --- | ---: | ---: |
| tiny | faster-whisper | local CPU | int8 | 0.050645 | 0.396320 |
| base | faster-whisper | local CPU | int8 | 0.065400 | 0.301099 |

The error column mixes WER and Mandarin CER, so it is only a compact
trade-off indicator. Dataset-specific WER/CER tables in
[`asr_benchmark.md`](asr_benchmark.md) remain the source for accuracy claims.

## Current Whisper.cpp Level 1 Results

The local `whisper.cpp` benchmark currently contains 76 rows over the earlier
38-clip formal manifest before AISHELL-4 was added: 38 clips times two ggml
models. All rows completed with `status=ok`. It has not yet been rerun on the
expanded 50-clip manifest.

| Model | Dataset | Language | Mean error | Mean cleaned error | Mean RTF |
| --- | --- | --- | ---: | ---: | ---: |
| tiny | AMI Meeting Corpus | en | 0.382383 | 0.313909 | 0.030548 |
| base | AMI Meeting Corpus | en | 0.351123 | 0.280012 | 0.056347 |
| tiny | Google FLEURS | en | 0.185512 | 0.185512 | 0.071982 |
| base | Google FLEURS | en | 0.133832 | 0.133832 | 0.139889 |

French and Mandarin FLEURS errors are much higher under the current default
command path, so they should be discussed as a limitation rather than a strong
multilingual deployment result.

## Interpretation

The proxy supports a cautious engineering point:

- `tiny` is faster on the current local CPU int8 setup.
- `base` is more accurate on the current multilingual and AMI subset.
- The gap is large enough to justify a mobile trade-off experiment.
- The repository now contains true local `whisper.cpp` rows, but still not
  true phone-device rows.

## Required True Level 1 Follow-Up

To convert these local results into a final phone-side ASR result, run the same
manifest on the target device and record:

- `whisper.cpp` commit or release;
- model file name and quantization, such as `ggml-tiny.en-q5_0.bin`;
- device model, OS version, CPU/GPU/NPU path if applicable;
- cold load time, warm runtime, and peak memory if available;
- WER/CER using the same references and normalization;
- failed clips and any preprocessing changes.

The local `whisper.cpp` benchmark entry point is documented in
[`whisper_cpp_level1.md`](whisper_cpp_level1.md). Proxy, local `whisper.cpp`,
and future phone-device rows must be reported separately.
