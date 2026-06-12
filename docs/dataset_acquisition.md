# Dataset Acquisition

## Status

Phase 2A-REAL prepares a small formal evaluation subset without committing raw
audio or downloaded archives.

Current local acquisition status:

| Track | Result | Real clips | Notes |
| --- | --- | ---: | --- |
| Multilingual ASR | Prepared with Google FLEURS fallback | 15 | Five English, five French, and five Mandarin Chinese validation clips |
| Common Voice | Blocked for automatic partial acquisition | 0 | The attempted official Mozilla Hugging Face dataset endpoint returned HTTP 404; the current Mozilla Data Collective API requires credentials and terms acceptance and documents whole-dataset downloads |
| English meeting/overlap | Prepared from AMI | 2 | Two 20-second excerpts from `ES2002a` with manual word references and derived overlap events |
| Mandarin meeting | Blocked by archive size or missing file-level route | 0 | AISHELL-4 official archives exceed the 500 MB cap; no verified small AliMeeting audio-plus-annotation API was found |

The combined manifest currently contains 17 real rows and 190.6 seconds of
audio. Local raw files use about 72 MB, including the AMI source WAV and manual
annotation ZIP used to generate the two short excerpts.

## Sources And Access

### Mozilla Common Voice

- Official platform: <https://datacollective.mozillafoundation.org/>
- API documentation:
  <https://datacollective.mozillafoundation.org/api/docs>
- Access requires a Mozilla Data Collective API key and prior acceptance of
  the applicable dataset terms.
- The documented download endpoint returns a dataset archive rather than a
  clip-streaming subset. TalkWeaver does not download a multi-gigabyte release
  merely to select five clips.

The downloader first attempts the official Mozilla Hugging Face dataset path.
When partial access is unavailable, it records the failure and uses the
official Google FLEURS dataset as a real multilingual fallback:

- FLEURS dataset: <https://huggingface.co/datasets/google/fleurs>
- Paper: <https://arxiv.org/abs/2205.12446>
- License: CC BY 4.0

The manifest labels every fallback row as `Google FLEURS`; these rows must not
be described as Common Voice results.

### AMI Meeting Corpus

- Official corpus: <https://groups.inf.ed.ac.uk/ami/corpus/>
- Audio mirror:
  <https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/>
- Manual annotations:
  <https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/>
- License: CC BY 4.0, with source attribution required.

The script downloads the `ES2002a` mixed-headset recording and manual
annotation release 1.6.2, then writes two short WAV excerpts. Reference
speaker anchors are generated from AMI manual word intervals. Overlap events
are deterministic interval intersections; interruption labels remain
`needs_annotation`.

### Mandarin Meeting Corpora

- AISHELL-4 official page: <https://www.openslr.org/111/>
- AISHELL-4 official code: <https://github.com/felixfuyihui/AISHELL-4>
- AliMeeting official page:
  <https://www.modelscope.cn/datasets/modelscope/AliMeeting>

The official AISHELL-4 `test.tar.gz` is 5,241,010,904 bytes and
`train_L.tar.gz` is 7,063,541,045 bytes. Both exceed the Phase 2A-REAL
automatic-download ceiling of 524,288,000 bytes. The AliMeeting page was
reachable, but no verified file-level route for one to three audio clips plus
matching annotations was identified. See
[`manual_dataset_steps.md`](manual_dataset_steps.md).

## Reproduction Commands

Run these commands from the repository root:

```bash
python scripts/download_common_voice_subset.py \
  --languages en fr zh-CN \
  --max-clips-per-language 5 \
  --output-root data/raw/public/common_voice \
  --reference-root data/reference/public/common_voice \
  --manifest-out data/manifests/common_voice_multilingual_real.csv

python scripts/download_meeting_subset.py \
  --dataset auto \
  --max-clips 2 \
  --output-root data/raw/public/english_meeting \
  --reference-root data/reference/public/english_meeting \
  --manifest-out data/manifests/english_meeting_real.csv

python scripts/download_mandarin_meeting_subset.py \
  --dataset auto \
  --max-clips 2 \
  --output-root data/raw/public/mandarin_meeting \
  --reference-root data/reference/public/mandarin_meeting \
  --manifest-out data/manifests/mandarin_meeting_real.csv
```

The Mandarin command currently exits with status 2 after writing an empty,
header-only manifest and reporting the verified blockers. It does not insert
placeholder rows.

Build and validate the combined manifest:

```bash
python scripts/build_formal_eval_manifest.py \
  --inputs \
    data/manifests/common_voice_multilingual_real.csv \
    data/manifests/english_meeting_real.csv \
    data/manifests/mandarin_meeting_real.csv \
  --output data/manifests/formal_eval_real.csv

python experiments/validate_manifest.py \
  --manifest data/manifests/formal_eval_real.csv \
  --require-real-files
```

## Local Layout

```text
data/
├── raw/public/                 # ignored by Git
│   ├── common_voice/           # FLEURS fallback WAV files at present
│   ├── english_meeting/        # AMI excerpts and ignored source downloads
│   └── mandarin_meeting/
├── reference/public/           # small permitted TXT/JSON references
│   ├── common_voice/
│   ├── english_meeting/
│   └── mandarin_meeting/
└── manifests/                  # CSV manifests and SHA-256 inventories
```

Raw WAV, MP3, FLAC, archives, and all files under `data/raw/public/` are
ignored by Git. Small reference TXT/JSON files, manifests, scripts, and
checksum inventories may be committed when their source license permits.

## Failure Policy

Acquisition scripts:

- never add failed or planned rows to `formal_eval_real.csv`;
- enforce file-size limits before or during download;
- preserve completed datasets when another source fails;
- report the URL and HTTP or access error;
- require manual terms acceptance where the official provider requires it;
- never fabricate transcripts, speaker labels, or download status.
