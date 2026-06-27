# Part 3 - Code Walkthrough Script, 5 to 6 Minutes

Goal: show that the project is engineered as a real pipeline, not a slide-only
concept. Focus on the most important files and line ranges. Do not open every
file. Keep the walkthrough moving.

Recommended editor setup: open the repository in VS Code. Use search or quick
open for each file. Keep line numbers visible.

## Walkthrough Path

### 0:00-0:35 - Start with the Pipeline

**Open:** `backend/pipeline.py:66-257`

**Say:**

I will start from the backend pipeline because this file shows the whole system
end to end. The pipeline is not a single Whisper call. It connects
preprocessing, ASR, diarization, overlap detection, temporal anchoring, RAG term
retrieval, constrained correction, summary export, and artifact manifests.

**Point to:** `backend/pipeline.py:80-97`

**Say:**

Here we preprocess audio and run ASR. The implementation supports real mode and
mock fallback, so the app can run even without heavy models or API keys. That
is important for reproducibility and demo reliability.

**Point to:** `backend/pipeline.py:108-128`

**Say:**

Then diarization produces speaker turns, and overlap detection identifies time
regions with multiple active speakers. This is the first key difference from a
plain transcript: we preserve speaker-time evidence.

### 0:35-1:25 - Temporal Anchors

**Open:** `backend/alignment.py:69-162`

**Say:**

This file builds temporal anchors. At line 69, words are aligned to speakers
using the midpoint of each word timestamp. If more than one speaker is active,
the word is marked as overlap. If no speaker is active, it is marked UNKNOWN.

**Point to:** `backend/alignment.py:75-99`

**Say:**

Each aligned word stores the word, start time, end time, speaker, active
speakers, overlap flag, confidence, and uncertainty label. This is the core
data structure behind the EvidenceMap.

**Point to:** `backend/alignment.py:118-162`

**Say:**

Then consecutive words with the same speaker assignment are grouped into
segments. These segments become auditable transcript units instead of one large
unstructured paragraph.

### 1:25-2:05 - Overlap Detection

**Open:** `backend/overlap.py:8-65`

**Say:**

Overlap detection is intentionally simple and inspectable. The code takes
speaker turns, collects all time boundaries, and checks each interval's
midpoint. If two or more speakers are active, it records an overlap region.
This gives the UI and correction layer a clear signal: this part of the audio
is risky and should be reviewed more conservatively.

**Point to:** `backend/overlap.py:56-63`

**Say:**

The output is a small evidence object: start, end, speakers, duration, and type.
That is exactly the kind of object the frontend can visualize.

### 2:05-3:05 - RAG and Term Rescue

**Open:** `backend/rag.py:178-260`

**Say:**

The first RAG layer is a local TF-IDF knowledge base. It loads Markdown
knowledge documents, extracts glossary terms and correction pairs, then ranks
relevant chunks by cosine similarity. This is lightweight and reproducible,
which is useful for a class project and for auditability.

**Open:** `backend/term_rescue.py:19-40`

**Say:**

The second layer is more targeted. A `TermMatch` keeps the canonical term, the
matched form, score, retrieval method, whether it is safe to apply, whether it
needs review, and the context reason. So even before an LLM is involved, the
candidate carries evidence and risk state.

**Open:** `backend/term_rescue.py:333-394`

**Say:**

Here we retrieve controlled candidates using exact, fuzzy, phonetic-like, or
fused strategies. The important part is line 369: context support is checked
before we decide a candidate is safe. That prevents mistakes such as changing a
physical "rack" into "RAG" when the context is not about retrieval-augmented
generation.

**Open:** `backend/term_rescue.py:479-520`

**Say:**

This function retrieves candidates without automatically changing transcript
text. That design is central to our research story. Retrieval creates evidence.
It does not directly rewrite the meeting record.

### 3:05-4:05 - Constrained LLM Correction

**Open:** `backend/term_verifier.py:135-187`

**Say:**

If an LLM verifier is used, the prompt is constrained. It asks for a JSON
decision: accept, needs_review, reject, or no_op. It also includes a language
policy for English, Mandarin, and French, because misrecognition patterns are
language-dependent.

**Open:** `backend/llm_correction.py:225-259`

**Say:**

This validation function rejects unsupported rewrites. The corrected text must
stay grounded in the raw text and retrieved terms. If the model introduces
unsupported tokens or changes content beyond glossary substitutions, the
correction is rejected.

**Open:** `backend/llm_correction.py:262-363`

**Say:**

This is the segment-level correction function. It corrects each temporal
segment independently. If the API is unavailable or the output fails validation,
the system falls back to deterministic rules. If the segment is overlapping or
unknown speaker, it is marked uncertain. This is why TalkWeaver is conservative
instead of blindly trusting the LLM.

### 4:05-4:55 - EvidenceGate and Negative Results

**Open:** `backend/evidence_gate.py:303-388`

**Say:**

EvidenceGate extracts numeric features from correction decisions: term recall,
error delta, changed token ratio, overlap flags, uncertainty, unsupported
changes, invented content, speaker attribution changes, and context risk. This
is our attempt to model whether a correction is safe to apply.

**Open:** `backend/evidence_gate.py:391-458`

**Say:**

The label policy is conservative. If a correction invents content, changes
speaker attribution, introduces forbidden changes, or increases reference text
error, it is rejected. If it is unresolved or ambiguous, it is routed to review.
This also explains our negative result: automated safety prediction is useful
for analysis, but it is not strong enough to replace auditing.

### 4:55-5:45 - Frontend Evidence Map

**Open:** `webapp/app.py:623-679`

**Say:**

The frontend starts from existing local artifacts. It does not run expensive
models during the demo. It loads a ConversationMap, resolves safe local audio,
and shows playable audio with the selected evidence sample.

**Open:** `webapp/app.py:698-808`

**Say:**

This is the EvidenceMap view. It shows anchor counts, speakers, overlap, review
flags, correction cards, the timeline, the anchor table, and the raw-versus-
corrected diff. This connects the backend research artifacts to a user-facing
investigation workflow.

**Open:** `webapp/data_loader.py:132-173`

**Say:**

The loader also protects the frontend. It validates ConversationMap files and
only resolves audio paths inside the repository. This matters because a public
demo should be safe and reproducible.

## Closing Line

**Say:**

The codebase matches the research story. Audio becomes ASR text, speaker-time
evidence, overlap evidence, retrieved term candidates, constrained corrections,
and finally an auditable EvidenceMap. The main achievement is not that every
automatic correction is perfect. The achievement is that every correction has a
traceable reason, a timestamp, and a review path.
