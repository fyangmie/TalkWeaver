# Manual Dataset Steps

This document covers datasets that cannot currently be acquired as a legal,
small automatic subset.

## Mozilla Common Voice

1. Create a Mozilla Data Collective account.
2. Review and accept the terms for the exact Common Voice language release.
3. Create an API key using the official instructions:
   <https://datacollective.mozillafoundation.org/api/docs>.
4. Check the archive size before downloading. Do not use this route when the
   required language archives would violate the project storage budget.
5. Keep downloaded archives and audio under `data/raw/public/common_voice/`.
6. Record dataset release, language, source clip ID, license, and checksum in
   the manifest.

Current automatic blocker:

```text
GET https://datasets-server.huggingface.co/rows
dataset=mozilla-foundation/common_voice_17_0
config=en|fr|zh-CN
split=validation
result=HTTP 404 Not found
```

The current Mozilla Data Collective API requires credentials and does not
provide the clip-level streaming route needed by this task. FLEURS is used as
the immediate multilingual fallback and is labeled accordingly.

## AISHELL-4

Official source: <https://www.openslr.org/111/>

Verified archive sizes on June 12, 2026:

```text
test.tar.gz     5,241,010,904 bytes
train_L.tar.gz  7,063,541,045 bytes
train_S.tar.gz 14,130,002,566 bytes
train_M.tar.gz 25,499,868,580 bytes
```

These files exceed the automatic 500 MB Phase 2A-REAL limit. Do not download
them without explicit approval and adequate storage.

Approved manual route:

1. Review the official dataset terms and citation requirements.
2. Obtain explicit approval for the required archive size.
3. Download into an ignored directory outside Git history.
4. Extract one to three short meeting segments with matching transcript and
   speaker/time annotations.
5. Store local audio under `data/raw/public/mandarin_meeting/`.
6. Store matching transcript text files under a local transcript directory,
   for example `data/reference/public/mandarin_meeting_transcripts/`.
7. Generate the manifest with:

```bash
python scripts/download_mandarin_meeting_subset.py \
  --local-audio-root data/raw/public/mandarin_meeting \
  --local-transcript-root data/reference/public/mandarin_meeting_transcripts \
  --max-clips 3 \
  --reference-root data/reference/public/mandarin_meeting \
  --manifest-out data/manifests/mandarin_meeting_real.csv
```

8. Rebuild and validate `formal_eval_real.csv`.

## AliMeeting

Official source:
<https://www.modelscope.cn/datasets/modelscope/AliMeeting>

Current blocker: the official page is reachable, but this implementation has
not verified a supported file-level API that returns one to three audio clips
and their matching transcript and speaker/time annotations. Do not scrape
unofficial mirrors or infer annotation links.

Approved manual route:

1. Review and accept the official ModelScope dataset terms.
2. Use the provider-supported SDK or download workflow.
3. Confirm archive sizes before downloading.
4. Select a small evaluation subset only after audio and matching references
   can be identified reliably.
5. Record source recording IDs, dataset version, license/access notes, and
   checksums.
6. Run the combined-manifest builder and strict validator.

## Validation

After any manually approved acquisition:

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
