# Optional Dependency Setup

## Minimal Mode

TalkWeaver's mock and reference-assisted workflows use the core dependencies:

```bash
python -m pip install -r requirements.txt
```

These modes do not require `faster-whisper`, model weights, a GPU,
`pyannote.audio`, `HF_TOKEN`, or an LLM API key.

Check the optional environment without installing anything:

```bash
python scripts/check_optional_dependencies.py
```

The non-strict check exits successfully even when optional packages are
missing. To require real ASR support:

```bash
python scripts/check_optional_dependencies.py --strict faster-whisper
```

## Real ASR Mode With Faster-Whisper

TalkWeaver requires Python 3.10 or newer. The upstream `faster-whisper`
package supports Python 3.9 or newer, but Python 3.10-3.12 is the recommended
project environment when starting a fresh model environment because compiled
dependency wheels can lag behind the newest Python release.

Install all optional TalkWeaver dependencies:

```bash
python -m pip install -r requirements-optional.txt
```

For ASR only, the upstream installation command is:

```bash
python -m pip install faster-whisper
```

Real ASR is intentionally not part of `requirements.txt`. Mock and
reference-assisted modes must remain usable on machines without model
dependencies.

For Mandarin CER, install the optional OpenCC normalizer:

```bash
python -m pip install opencc-python-reimplemented
```

Without it, ASR inference still works, but Traditional/Simplified character
differences remain in CER and are explicitly flagged in metric metadata.

When a named model such as `tiny` is used for the first time,
`faster-whisper` automatically downloads the corresponding CTranslate2 model
from the Hugging Face Hub. This is a model download, not a TalkWeaver source
file. Model caches and weights must not be committed.

## CPU Recommended Setup

Start with a short AMI clip and the smallest model:

```bash
python scripts/run_talkweaver_workflow.py \
  --manifest data/manifests/formal_eval_real.csv \
  --clip-id ami_es2002a_01 \
  --asr-model tiny \
  --device cpu \
  --compute-type int8 \
  --diarization-source reference \
  --output outputs/conversation_maps/
```

Recommended CPU options:

```text
--asr-model tiny
--device cpu
--compute-type int8
```

This command uses real ASR and reference speaker/time evidence. The resulting
metadata should report:

```text
asr_mode=real
diarization_mode=reference
```

If real inference does not complete, no ConversationMap claiming
`asr_mode=real` is written.

## GPU Notes

GPU execution is optional. According to the official faster-whisper
documentation, current CTranslate2 releases require compatible NVIDIA
libraries, with CUDA 12 and cuDNN 9 expected by current releases. Verify the
exact CUDA, cuDNN, driver, and CTranslate2 compatibility before using:

```text
--device cuda
--compute-type float16
```

Do not assume that a PyTorch CUDA installation is sufficient for
CTranslate2. CPU `int8` is the reproducible first smoke path.

## Cache And Git Policy

- Public raw audio remains under `data/raw/public/` and is ignored by Git.
- Generated ConversationMaps remain under `outputs/conversation_maps/` and
  are ignored except for `.gitkeep`.
- Hugging Face and CTranslate2 model caches must remain outside Git.
- Do not copy downloaded model directories into this repository.
- Do not commit `.env`, API keys, `HF_TOKEN`, private audio, or model weights.

## Troubleshooting

### Missing Package

Expected failure:

```text
Real ASR requested but faster-whisper is not installed. Install with: pip install -r requirements-optional.txt
```

Install and verify:

```bash
python -m pip install -r requirements-optional.txt
python scripts/check_optional_dependencies.py --strict faster-whisper
```

### Model Download Failure

The first real run needs Hugging Face network access. Check proxy settings,
disk space, and Hugging Face availability. Do not replace a failed real run
with mock output unless `--mock-models` is explicitly requested.

### Unsupported Compute Type

Use the conservative CPU settings:

```text
--device cpu --compute-type int8
```

If the local CPU/backend rejects `int8`, test `--compute-type float32` and
record that change in run metadata.

### Automatic Diarization Is Unavailable

Real ASR can still be developed with:

```text
--diarization-source reference
```

This is oracle/reference-assisted diarization and must not be reported as
automatic diarization performance.

## Primary References

- faster-whisper official repository:
  <https://github.com/SYSTRAN/faster-whisper>
- faster-whisper PyPI:
  <https://pypi.org/project/faster-whisper/>
