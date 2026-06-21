# Dataset Acquisition

## Status

Phase 2A-REAL prepares a small formal evaluation subset without committing raw
audio or downloaded archives.

Current local acquisition status:

| Track | Result | Real clips | Notes |
| --- | --- | ---: | --- |
| Multilingual ASR | Prepared with Google FLEURS fallback | 30 | Ten English, ten French, and ten Mandarin Chinese validation clips |
| Common Voice | Blocked for automatic partial acquisition | 0 | The attempted official Mozilla Hugging Face dataset endpoint returned HTTP 404; the current Mozilla Data Collective API requires credentials and terms acceptance and documents whole-dataset downloads |
| English meeting/overlap | Prepared from AMI | 8 | Eight 20-second excerpts from `ES2002a` with manual word references and derived overlap events |
| English meeting held-out | Prepared from AMI | 24 | Six 20-second clips each from `ES2002a`, `ES2002b`, `ES2002c`, and `ES2002d` |
| Mandarin meeting | Prepared from AISHELL-4 | 12 | Twelve 20-second excerpts from one AISHELL-4 test recording parsed from TextGrid references |
| AISHELL-4 benchmark subset | Prepared from AISHELL-4 | 60 | Three 20-second clips from each of 20 test recordings; 1200 seconds, 29 multi-speaker clips, 10 overlap clips |

The combined manifest currently contains 50 real rows and 698.34 seconds of
audio. Local raw files include ignored FLEURS WAV files, the AMI source WAV,
the AMI manual annotation ZIP, and AISHELL-4 excerpts used to generate the
short meeting clips. The full local AISHELL-4 `test.tar.gz` archive and
extracted FLAC/TextGrid/RTTM files are ignored by Git.

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

The script downloads mixed-headset AMI recordings and manual annotation
release 1.6.2, then writes short 20-second WAV excerpts. Reference speaker
anchors are generated from AMI manual word intervals. Overlap events are
deterministic interval intersections; interruption labels remain
`needs_annotation`.

### Mandarin Meeting Corpora

- AISHELL-4 official page: <https://www.openslr.org/111/>
- AISHELL-4 official code: <https://github.com/felixfuyihui/AISHELL-4>
- AliMeeting official page:
  <https://www.modelscope.cn/datasets/modelscope/AliMeeting>

The official AISHELL-4 `test.tar.gz` is 5,241,010,904 bytes and
`train_L.tar.gz` is 7,063,541,045 bytes. Both exceed the default Phase
2A-REAL automatic-download ceiling of 524,288,000 bytes. On June 21, 2026,
`test.tar.gz` was completed locally with resumable `wget -c`, verified with
`gzip -t`, and extracted under the ignored raw-data tree. The fixed benchmark
manifest selects 60 short clips from the 20 extracted test recordings. This is
not a full AISHELL-4 test-set benchmark, but it is much stronger than the
earlier one-recording sanity subset.

## Reproduction Commands

Run these commands from the repository root:

```bash
python scripts/download_common_voice_subset.py \
  --languages en fr zh-CN \
  --max-clips-per-language 10 \
  --output-root data/raw/public/common_voice \
  --reference-root data/reference/public/common_voice \
  --manifest-out data/manifests/common_voice_multilingual_real.csv

python scripts/download_meeting_subset.py \
  --dataset auto \
  --max-clips 8 \
  --output-root data/raw/public/english_meeting \
  --reference-root data/reference/public/english_meeting \
  --manifest-out data/manifests/english_meeting_real.csv

python scripts/download_meeting_subset.py \
  --max-clips 24 \
  --max-clips-per-meeting 6 \
  --meeting-ids ES2002a ES2002b ES2002c ES2002d \
  --output-root data/raw/public/english_meeting_heldout \
  --reference-root data/reference/public/english_meeting_heldout \
  --manifest-out data/manifests/english_meeting_heldout_real.csv

python scripts/download_mandarin_meeting_subset.py \
  --dataset aishell4 \
  --split test \
  --allow-large-download \
  --max-clips 12 \
  --clip-duration-seconds 20 \
  --output-root data/raw/public/mandarin_meeting \
  --reference-root data/reference/public/mandarin_meeting \
  --manifest-out data/manifests/mandarin_meeting_real.csv
```

After the full test archive is available and extracted, build the fixed
AISHELL-4 benchmark subset with:

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

If a small, license-compliant local Mandarin meeting subset is already
available, import complete audio+transcript pairs without downloading the
large official archives:

```bash
python scripts/download_mandarin_meeting_subset.py \
  --local-audio-root data/raw/public/mandarin_meeting \
  --local-transcript-root data/reference/public/mandarin_meeting_transcripts \
  --max-clips 3 \
  --reference-root data/reference/public/mandarin_meeting \
  --manifest-out data/manifests/mandarin_meeting_real.csv
```

This local import creates ASR reference rows only. Speaker and overlap fields
are not treated as diarization gold unless separate speaker/time annotations
are added.

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

## Added Held-Out Artifacts

```text
data/manifests/english_meeting_heldout_real.csv
data/reference/public/english_meeting_heldout/interruption_label_candidates.csv
data/manifests/earnings22_v3_blind_6x180.csv
```

The AMI held-out manifest is a real public-data subset balanced across four
AMI meetings. The Earnings-22 v3 blind manifest contains six reliable
180-second files selected with the existing reference-quality guard. It is
still a small blind set, but it is no longer only a one-file smoke check.

## Failure Policy

Acquisition scripts:

- never add failed or planned rows to `formal_eval_real.csv`;
- enforce file-size limits before or during download;
- preserve completed datasets when another source fails;
- report the URL and HTTP or access error;
- require manual terms acceptance where the official provider requires it;
- never fabricate transcripts, speaker labels, or download status.
