# Diarization Background

Speaker diarization estimates who spoke when. Its output is usually a sequence
of timestamped speaker turns with anonymous labels such as `SPEAKER_00`.

TalkWeaver will use pyannote when model access and an `HF_TOKEN` are available.
Mock turns keep the project runnable without external access. Evaluation must
distinguish missed speech, false alarms, speaker confusion, boundary errors,
and overlap handling.

Speaker labels are session-local identifiers unless a separate speaker
recognition system is explicitly implemented.
