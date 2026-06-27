# Part 2 - Web Demo Script, 5 to 6 Minutes

Demo target: deployed Docker Space or local Streamlit app.

Local command if needed:

```bash
streamlit run app.py
```

Goal: show that TalkWeaver is understandable and useful to a normal user. Do
not read code in this section. Show interaction: language switch, playable
audio, correction evidence, EvidenceMap, timeline, anchor table, and results.

## Demo Path

### 0:00-0:35 - Landing View

**Action:** Open the website. Keep language in English first.

**Say:**

This is the public-facing TalkWeaver MVP. The first screen avoids heavy
technical language. It tells the user the real purpose: TalkWeaver is an
AI Meeting Detective. It does not just summarize a meeting. It preserves who
spoke, when they spoke, where voices overlap, and whether a correction is
supported by evidence.

**Point to:** The badges "EvidenceMap", "Speaker timeline", "Overlap alerts",
and "Edit audit".

**Code anchor for later:** `webapp/app.py:432-470` renders this hero and the
three value cards.

### 0:35-1:00 - Language Switch

**Action:** Click the language selector and switch English to Chinese, then to
French, then back to English.

**Say:**

The website supports Chinese, English, and French. This is not only a cosmetic
feature. Multilingual evaluation is part of our project, and the interface
should also communicate the same evidence story to different users.

**Code anchor for later:** `webapp/app.py:45-49` defines the three languages,
and `webapp/app.py:51-223` stores the UI text.

### 1:00-1:50 - Play the Default Evidence Sample

**Action:** Scroll to "1. Listen to an auditable audio clip." Click the audio
player for the default sample.

**Say:**

Here the user can listen to the audio first. This is important because our
project is evidence-grounded. We do not want users to only see a polished
transcript. They should be able to connect the transcript back to the audio and
the time region.

**Action:** Point to the current sample card and the badges such as playable,
term corrections, and overlap evidence if present.

**Say:**

The selected example is not random. The app scores available ConversationMaps
and prefers samples with playable audio, term corrections, and multi-speaker
or overlap evidence. That helps the demo show the core idea quickly.

**Code anchor for later:** `webapp/app.py:539-552` scores samples, and
`webapp/app.py:623-679` renders the audio and sample selector.

### 1:50-2:45 - Show the Correction Evidence

**Action:** On the right side, point to "Before" and "After" correction cards.
If needed, open "Switch other samples" and select the multi-speaker term
correction example:

```text
earnings22_4481221_0000_180s_conversation_map
```

**Say:**

This is the most important user-facing feature. We do not hide the correction.
TalkWeaver shows the original ASR text, the corrected or preserved text, and
the evidence terms used to justify the change. If the correction is supported,
the user sees why. If the evidence is weak, the system should not silently force
the edit.

**Point to:** The "Evidence terms" line.

**Say:**

This is where the RAG idea becomes practical. RAG is not being used to invent a
summary. It is used as a domain-term rescue mechanism, and every rescue remains
auditable.

**Code anchor for later:** `webapp/app.py:488-520` renders before-and-after
correction cards.

### 2:45-3:50 - EvidenceMap Section

**Action:** Scroll to "2. EvidenceMap: turn transcript into inspectable
evidence."

**Say:**

Now the transcript is organized as an EvidenceMap. The top metrics count
anchors, speakers, overlap events, review flags, and audits. This turns a long
transcript into a set of inspectable evidence units.

**Action:** Point to "Who spoke", "Where speech overlapped", and "Where review
is needed."

**Say:**

These cards answer practical meeting questions. Who is involved? Where is the
audio chaotic? Which parts should not be trusted automatically? This is why the
project is more than ASR accuracy. It is a workflow for checking the transcript.

**Action:** Point to the timeline.

**Say:**

The timeline shows temporal anchors by speaker. Overlap and review regions are
visually marked, so the user can see where uncertainty happens in time.

**Code anchor for later:** `webapp/app.py:698-741` renders the EvidenceMap
summary and timeline. `webapp/components/speaker_timeline.py:60-179` builds the
anchor timeline visualization.

### 3:50-4:40 - Inspect an Anchor

**Action:** Scroll to the anchor table. Click the "Inspect anchor" dropdown and
choose a row with a correction or overlap flag.

**Say:**

Each row is a temporal anchor. It has start time, end time, speaker, raw text,
corrected text, overlap status, confidence, and retrieved terms. When I inspect
one anchor, the app shows a side-by-side raw versus corrected view. This is the
audit trail. It means a teacher, researcher, or meeting user can challenge any
edit instead of accepting a black-box transcript.

**Code anchor for later:** `webapp/app.py:743-808` renders the anchor table,
dropdown, diff view, and evidence badges.

### 4:40-5:35 - Results Section

**Action:** Scroll to "3. Project validation results."

**Say:**

The final panel connects the demo back to research evidence. It shows that
meeting speech is harder than read speech, that the EvidenceMap workflow runs
end to end, that domain terms can be recovered more safely with conservative
RAG, and that automatic correction still needs caution.

This last point is important. A weaker project would only show success cases.
TalkWeaver also shows negative results. The web demo therefore matches the
paper story: the system is useful because it makes evidence and uncertainty
visible.

**Code anchor for later:** `webapp/app.py:839-872` loads result summaries and
renders the four evidence metrics.

## Closing Line

**Say:**

In short, the website is not just decoration. It is the human-facing form of
the research idea: turn a chaotic transcript into evidence that users can
listen to, inspect, and audit.
