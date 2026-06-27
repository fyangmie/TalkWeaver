# Part 1 - PPT Presentation Script, 5 to 6 Minutes

File to present: `docs/TalkWeaver_Project_Presentation_EN.pptx`

Goal: introduce the research story, not every implementation detail. This part
should convince the teacher that TalkWeaver is a research-driven project about
auditable chaotic meeting transcription, not just a nice web app.

## Timing and Slide Cues

### 0:00-0:30 - Slide 1

**Action:** Open slide 1, title page.

**Say:**

Today I will present TalkWeaver, an AI Meeting Detective for chaotic
multi-speaker speech. The key phrase is "evidence-grounded conversation maps."
We are not only trying to produce a clean transcript. We are trying to preserve
the evidence behind the transcript: who spoke, when they spoke, where voices
overlap, which words are likely misheard, and whether a correction is actually
supported.

### 0:30-1:10 - Slide 2

**Action:** Move to slide 2, motivation.

**Say:**

The motivation is that meeting transcription is not just WER. In a normal ASR
benchmark, we usually ask whether the words are right. In a real meeting, that
is not enough. If two people talk at the same time, if one person interrupts
another, if a speaker is low-energy, or if a domain term is misheard, a fluent
transcript can still be misleading. LLM post-processing makes this even more
dangerous, because it can make the text look more natural while silently adding
unsupported edits. So our central framing is: chaotic meeting transcription
should be treated as an evidence-grounded auditing problem, not merely an ASR
or summarization problem.

### 1:10-1:55 - Slide 3

**Action:** Move to slide 3, research questions and contributions.

**Say:**

Our research questions follow from that framing. First, can speaker-time
structure make transcripts easier to inspect? Second, can overlap-aware
uncertainty reduce unsafe correction in messy regions? Third, can RAG help
recover domain terms that ASR often mishears? Fourth, what are the accuracy and
speed trade-offs for local or mobile-style ASR?

The contribution is the combination. TalkWeaver builds a temporal ConversationMap
that connects ASR text, speaker evidence, overlap evidence, RAG term evidence,
and correction audit fields. The important point is that each correction keeps
the original text and its evidence, so the user can inspect it instead of just
trusting a rewritten transcript.

### 1:55-2:45 - Slide 4

**Action:** Move to slide 4, architecture.

**Say:**

This is the full pipeline. Audio first goes through preprocessing, then ASR.
Then diarization gives speaker-time turns. We align ASR words to speaker turns
to create temporal anchors. Overlap detection marks intervals with more than
one active speaker. RAG retrieves possible domain terms, but retrieval alone
does not mean we rewrite the transcript. A constrained correction layer decides
whether the evidence is strong enough. Finally, the UI shows the original text,
corrected text, evidence terms, time, speaker, overlap, and review flags.

**Speaker note:** Point to the pipeline from left to right. Do not spend more
than one minute here. The code walkthrough will cover implementation later.

### 2:45-3:25 - Slide 5

**Action:** Move to slide 5, Web MVP.

**Say:**

The web MVP is designed for non-expert users. Instead of showing a long
technical table first, it answers practical questions: what changed, why did it
change, who was speaking, and where should I be careful? The before-and-after
view is important because it makes RAG correction visible. If the system changes
a term, the evidence term is shown beside the change. If evidence is weak, the
system should preserve the raw text or route the segment to review.

### 3:25-4:10 - Slide 6

**Action:** Move to slide 6, real public-data evaluation.

**Say:**

We also ran real public-data validation. The point is not to claim
state-of-the-art. The point is to show that the problem exists across several
settings. We used FLEURS for read speech and multilingual ASR, AMI for English
meeting speech, AISHELL-4 for Mandarin meetings, and Earnings-22 for
finance-domain term recovery. The pattern is clear: read speech is easier, while
meeting speech is harder because speaker boundaries, overlap, and domain terms
interact.

### 4:10-4:45 - Slide 7

**Action:** Move to slide 7, speaker and overlap evidence.

**Say:**

For speaker and overlap evidence, pyannote gives useful speaker-time structure,
but overlap remains difficult. This is exactly why TalkWeaver does not hide
uncertainty. It marks overlap, review regions, and speaker-time anchors so the
final transcript becomes auditable. In the paper, this supports the claim that
meeting transcription needs provenance, not just prettier text.

### 4:45-5:20 - Slide 8

**Action:** Move to slide 8, RAG term recovery.

**Say:**

The RAG module is intentionally narrow. We are not building a general meeting
chatbot. We use retrieval mainly to recover misheard domain terms, such as
company names, financial terminology, or ASR research terms. Our strongest safe
claim is that conservative RAG can improve term recall in a controlled and
audited way, but it must be gated. If the evidence is ambiguous, changing text
can hurt.

### 5:20-5:50 - Slides 9 and 10

**Action:** Move to slide 9, then slide 10 for closing.

**Say:**

Slide 9 is important because it shows our honesty. EvidenceGate and correction
safety experiments include negative results. We found that automated correction
decisions do not generalize strongly enough to be trusted blindly. That is not
a failure of the project story. It supports the design choice: make corrections
auditable, conservative, and evidence-backed.

To conclude, TalkWeaver is not a state-of-the-art claim. It is a research
prototype that turns chaotic multi-speaker speech into an inspectable evidence
map. The strongest takeaway is that we should not only ask "what does the
transcript say?" We should also ask "what evidence supports this transcript?"

## Backup Lines for Questions

- If asked why this is not just Whisper plus RAG: "Because the core output is
  not a rewritten transcript. The output is a temporal evidence map with
  speaker, overlap, term, and correction audit fields."
- If asked about limitations: "The system is evidence-complete, but automatic
  correction and interruption evaluation still require more human-labeled
  data."
- If asked about contribution: "The contribution is the auditing framework and
  conservative evidence flow, with RAG term recovery as one important module."
