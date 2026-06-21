# Earnings-22 Diagnostic Held-Out Protocol

This protocol freezes the first diagnostic held-out expansion for the
TalkWeaver domain-term recovery experiment. The subset was unseen before the
v1 run, but its errors have now been inspected; it must not be reused as the
final blind benchmark after v2 gate tuning.

## Goal

Evaluate whether evidence-constrained RAG + LLM correction improves
domain-term recovery on Earnings-22 files, and use the v1 failure modes to
define a safer v2 gate.

## Diagnostic Setup

- Dataset source: Rev.com `speech-datasets` Earnings-22, public GitHub source.
- License/access: Earnings-22 README advertises CC BY-SA 4.0; raw audio and
  generated WAV slices are kept local and ignored by git.
- Dev/diagnostic files excluded from held-out:
  `4453225`, `4467434`, `4481221`, `4462231`.
- Held-out candidate order:
  `4474955`, `4483046`, `4468919`, `4475604`, `4471586`, `4482968`,
  `4482110`, `4469075`, `4446796`, `4483623`, `4485206`, `4480850`.
- Fallback order if a candidate fails download or reference quality guards:
  `4470290`, `4469528`, `4470010`, `4470570`, `4329526`, `4450779`.
- Target: 12 reliable 180-second slices.
- ASR baseline: `faster-whisper` `tiny` and `base`, CPU, int8,
  `vad_filter=false`.
- Diagnostic RAG glossary:
  `data/controlled_terms/earnings22_multi_context_terms.json`.

## v2 Gate Policy

- Treat weak entity evidence as unsafe.
- Reject common-token entity rewrites such as `U.S. -> UEPS`.
- Treat equivalent wording such as `cents per share -> cents a share` as
  `no_op`, not a rescue win.
- Accept short glossary repairs only when the source text is a predefined error
  form and the local context is sufficient.
- Keep LLM output constrained to candidate verification; the deterministic gate
  is the final authority before text replacement.

## Final Blind Protocol

- Build a second 12-file Earnings-22 blind subset only after v2 code and
  glossary are frozen.
- Exclude all dev and diagnostic source file IDs:
  `4453225`, `4467434`, `4481221`, `4462231`, `4474955`, `4483046`,
  `4468919`, `4475604`, `4471586`, `4482110`, `4446796`, `4483623`,
  `4470290`, `4469528`, `4470570`, `4329526`.
- Do not add observed final-blind ASR errors to the glossary before scoring.
- Report final claims from the second blind subset; use this diagnostic subset
  only for method development and error-analysis narrative.

## Final Blind v2 Run

- Manifest: `data/manifests/earnings22_final_blind_12x180.csv`.
- Source file IDs: `4481952`, `4482383`, `4449269`, `4450779`,
  `4467079`, `4481601`, `4483296`, `4483668`, `4483678`, `4483680`,
  `4423872`, `4472895`.
- Overlap with dev and diagnostic file IDs: none.
- ASR baseline: `faster-whisper` `tiny` and `base`, CPU, int8,
  `vad_filter=false`.
- v2 results:
  - `base`: WER `0.186844 -> 0.187018`, term F1 `0.972222 -> 0.930556`.
  - `tiny`: WER `0.221805 -> 0.221978`, term F1 `0.888889 -> 0.930556`.
- Interpretation: v2 successfully blocks the known `U.S. -> UEPS` false
  positive and improves tiny-model term F1, but the final blind run exposes a
  residual `non-gap -> non-GAAP` false positive for base and tiny. The paper
  should claim improved safety over v1 and model-dependent term recovery, not
  universal improvement across all ASR models.

## Leakage Rules

- Reference transcripts are used only for scoring.
- Reference transcripts are not sent to the LLM.
- Diagnostic ASR errors must not be used to add new glossary entries before
  scoring the same diagnostic run.
- Any newly observed diagnostic error forms belong in error analysis and may only
  inform a future dev iteration.

## Claim Boundary

This experiment supports claims about evidence-constrained domain-term recovery
in earnings-call ASR. It does not by itself prove the full multi-speaker
meeting-chaos contribution.
