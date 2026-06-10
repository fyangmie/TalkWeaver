# RAG Background

Retrieval-augmented generation supplies selected external context to a
generation or correction model. In TalkWeaver, retrieval is deliberately
narrow: it proposes domain terms that may explain an ASR substitution.

The planned baseline loads local Markdown, uses TF-IDF retrieval, and returns a
small candidate list for each transcript segment. Evaluation should measure
term retrieval precision and recall as well as final Term Error Rate.

RAG is not used to invent meeting facts or to replace diarization and overlap
analysis.
