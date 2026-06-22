#!/usr/bin/env python3
"""
Generate R2 independent heldout benchmark for binary safe-to-apply correction.
Creates 100-120 independent test cases covering all required categories.
"""

import pandas as pd
import itertools


def generate_heldout_dataset() -> pd.DataFrame:
    """Generate complete independent test set"""
    test_cases = []
    proposal_id = 1
    
    def add_case(raw, proposed, reference, category, safe, overlap=False, heavy=False,
                 speaker_amb=False, partial=False, retrieved_terms="", needs_human=False):
        nonlocal proposal_id
        
        # Auto-infer retrieved_terms
        if not retrieved_terms:
            if "pyannote" in proposed or "piano" in raw:
                retrieved_terms = '["pyannote", "pyannote.audio"]'
            elif "diarization" in proposed or "diary" in raw:
                retrieved_terms = '["diarization", "speaker diarization"]'
            elif "WER" in proposed or "where" in raw:
                retrieved_terms = '["WER"]'
            elif "DER" in proposed or "dear" in raw:
                retrieved_terms = '["DER"]'
            elif "RAG" in proposed or "rack" in raw:
                retrieved_terms = '["RAG"]'
            elif "anchor" in proposed or "anger" in raw:
                retrieved_terms = '["temporal anchor"]'
            else:
                retrieved_terms = '[]'
        
        error_before = 0.3 if raw != reference else 0.0
        error_after = 0.0 if safe and proposed == reference else (0.3 if not safe else 0.0)
        
        case = {
            "proposal_id": f"heldout::{proposal_id:04d}",
            "source": "heldout_controlled",
            "category": category,
            "language": "en",
            "raw_asr_text": raw,
            "proposed_corrected_text": proposed,
            "reference_text": reference if reference else raw,
            "context": f"Test context for {category}",
            "retrieved_terms": retrieved_terms,
            "overlap_flag": str(overlap),
            "heavy_overlap_flag": str(heavy),
            "speaker_ambiguity_flag": str(speaker_amb),
            "partial_utterance_flag": str(partial),
            "error_before": error_before,
            "error_after": error_after,
            "error_delta": error_before - error_after,
            "binary_label": "safe_to_apply" if safe else "do_not_apply",
            "label_source": "controlled_reference",
            "label_rule": "error_after + 0.010 < error_before" if safe else "error_after + 0.010 >= error_before",
            "needs_human_check": str(needs_human),
            "notes": f"Heldout test case {proposal_id}; independent from R1 templates.",
            "case_group": f"heldout_group_{(proposal_id - 1) // 20 + 1}"
        }
        test_cases.append(case)
        proposal_id += 1
    
    # ============================================================
    # 1. Technical term recovery cases - Generate via combinations
    # ============================================================
    terms = [
        ("piano note", "pyannote", "pyannote"),
        ("pie anode", "pyannote", "pyannote"),
        ("piano note audio", "pyannote.audio", "pyannote.audio"),
        ("diary station", "diarization", "diarization"),
        ("diary zation", "diarization", "diarization"),
        ("diary station output", "diarization output", "diarization"),
        ("rack", "RAG", "RAG"),
        ("rack glossary", "RAG glossary", "RAG"),
        ("rack retrieval", "RAG retrieval", "RAG"),
        ("where", "WER", "WER"),
        ("where metric", "WER metric", "WER"),
        ("where score", "WER score", "WER"),
        ("ear", "DER", "DER"),
        ("dear", "DER", "DER"),
        ("ear score", "DER score", "DER"),
        ("temporal anger", "temporal anchor", "temporal anchor"),
        ("anger anchor", "temporal anchor", "temporal anchor"),
        ("temporal anger anchor", "temporal anchor", "temporal anchor"),
        ("faster whisper", "faster-whisper", "faster-whisper"),
        ("c translate two", "CTranslate2", "CTranslate2"),
        ("q when", "Qwen", "Qwen"),
        ("tag speech", "TagSpeech", "TagSpeech"),
        ("dm asr", "DM-ASR", "DM-ASR"),
    ]
    
    prefixes = ["", "We use ", "The system uses ", "In our work, we use ", "Consider using "]
    suffixes = ["", " for diarization", " in the pipeline", " for ASR", " in our benchmark"]
    
    for term_raw, term_correct, term_ref in terms:
        for prefix in prefixes[:2]:  # Use first 2 prefixes to keep size manageable
            for suffix in suffixes[:2]:  # Use first 2 suffixes
                raw = f"{prefix}{term_raw}{suffix}".strip()
                proposed = f"{prefix}{term_correct}{suffix}".strip()
                reference = f"{prefix}{term_ref}{suffix}".strip()
                add_case(raw, proposed, reference, "technical_term_recovery", True)
    
    # ============================================================
    # 2. Ordinary word negative controls (should reject)
    # ============================================================
    negative_templates = [
        ("put the router on the rack", "put the router on the RAG", "put the router on the rack"),
        ("where is the microphone", "WER is the microphone", "where is the microphone"),
        ("dear team please review", "DER team please review", "dear team please review"),
        ("play piano note for me", "play pyannote for me", "play piano note for me"),
        ("the diary is on my desk", "the diarization is on my desk", "the diary is on my desk"),
        ("write tag speech in HTML", "write TagSpeech in HTML", "write tag speech in HTML"),
        ("the rack is in the server room", "the RAG is in the server room", "the rack is in the server room"),
        ("where did you put the file", "WER did you put the file", "where did you put the file"),
        ("dear sir or madam", "DER sir or madam", "dear sir or madam"),
        ("the piano note was beautiful", "the pyannote was beautiful", "the piano note was beautiful"),
        ("I keep a daily diary", "I keep a daily diarization", "I keep a daily diary"),
        ("please tag speech in the document", "please TagSpeech in the document", "please tag speech in the document"),
    ]
    
    for raw, proposed, reference in negative_templates:
        add_case(raw, proposed, reference, "ordinary_word_negative_control", False)
    
    # ============================================================
    # 3. Single speaker safety cases
    # ============================================================
    single_speaker_cases = [
        ("we use pyannote for diarization", "we use pyannote for diarization", "we use pyannote for diarization", True),
        ("WER is 0.2 after correction", "WER is 0.2 after correction", "WER is 0.2 after correction", True),
        ("DER improved by 5 percent", "DER improved by 5 percent", "DER improved by 5 percent", True),
        ("faster-whisper runs on CPU", "faster-whisper runs on CPU", "faster-whisper runs on CPU", True),
        ("CTranslate2 enables fast inference", "CTranslate2 enables fast inference", "CTranslate2 enables fast inference", True),
        ("Qwen is a good LLM", "Qwen is a good LLM", "Qwen is a good LLM", True),
        ("TagSpeech provides temporal labels", "TagSpeech provides temporal labels", "TagSpeech provides temporal labels", True),
        ("DM-ASR conditions on speaker", "DM-ASR conditions on speaker", "DM-ASR conditions on speaker", True),
    ]
    
    for raw, proposed, reference, safe in single_speaker_cases:
        add_case(raw, proposed, reference, "single_speaker_safety", safe)
    
    # ============================================================
    # 4. Overlap speech cases
    # ============================================================
    overlap_templates = [
        ("speaker A says piano note while B agrees", "speaker A says pyannote", "speaker A says pyannote", True, False),
        ("A: diary station B: yes", "A: diarization", "A: diarization", True, False),
        ("A: where metric B: okay", "A: WER metric", "A: WER metric", True, False),
        ("A: rack glossary B: correct", "A: RAG glossary", "A: RAG glossary", True, False),
        ("A: temporal anger B: anchor", "A: temporal anchor", "A: temporal anchor", True, False),
        ("A: dear score B: 0.05", "A: DER score", "A: DER score", True, False),
        ("A: faster whisper B: yes", "A: faster-whisper", "A: faster-whisper", True, False),
        ("A: c translate two B: okay", "A: CTranslate2", "A: CTranslate2", True, False),
        ("A: piano... B: no...", "pyannote is better", "", False, True),
        ("A: diary... B: station...", "diarization works well", "", False, True),
        ("A: where... B: I think...", "WER improved", "", False, True),
        ("A: rack... B: retrieval...", "RAG solved it", "", False, True),
    ]
    
    for raw, proposed, reference, safe, heavy in overlap_templates:
        add_case(raw, proposed, reference, "overlap_correction", safe, 
                overlap=True, heavy=heavy, needs_human=heavy)
    
    # ============================================================
    # 5. Speaker attribution ambiguity cases
    # ============================================================
    speaker_cases = [
        ("[unclear speaker]: we use piano note", "we use pyannote", "pyannote", False),
        ("[overlap] A or B: diary station", "diarization is correct", "diarization", False),
        ("[unknown]: where is the metric", "WER is the metric", "WER", False),
        ("[speaker ?]: rack glossary", "RAG glossary", "RAG", False),
        ("A or B said temporal anger", "temporal anchor", "temporal anchor", False),
    ]
    
    for raw, proposed, reference, safe in speaker_cases:
        add_case(raw, proposed, reference, "speaker_attribution_risk", safe,
                speaker_amb=True, needs_human=True)
    
    # ============================================================
    # 6. Partial utterance cases
    # ============================================================
    partial_cases = [
        ("we should compare WER with...", "we should compare WER with baseline", "", False),
        ("the result is...", "the result is significant", "", False),
        ("use pyannote for...", "use pyannote for all diarization", "", False),
        ("the rack...", "the RAG glossary", "", False),
        ("where metric...", "WER metric", "", False),
        ("dear score...", "DER score", "", False),
    ]
    
    for raw, proposed, reference, safe in partial_cases:
        add_case(raw, proposed, reference, "partial_utterance", safe,
                partial=True, needs_human=True)
    
    # ============================================================
    # 7. Fluent hallucination cases
    # ============================================================
    hallucination_cases = [
        ("we evaluate where", "we evaluate WER and prove 50% improvement", "we evaluate WER", False),
        ("rack retrieves terms", "RAG retrieves terms and guarantees zero errors", "RAG retrieves terms", False),
        ("tag speech provides labels", "TagSpeech outperforms all prior systems", "TagSpeech provides labels", False),
        ("pyannote works well", "pyannote achieves state-of-the-art results", "pyannote works well", False),
    ]
    
    for raw, proposed, reference, safe in hallucination_cases:
        add_case(raw, proposed, reference, "fluent_hallucination", safe, needs_human=True)
    
    # ============================================================
    # 8. No-change cases
    # ============================================================
    no_change_cases = [
        ("the meeting starts at nine", "the meeting starts at nine", "the meeting starts at nine", True),
        ("put the laptop on the table", "put the laptop on the table", "put the laptop on the table", True),
        ("where should we meet", "where should we meet", "where should we meet", True),
        ("bonjour dear colleague", "bonjour dear colleague", "bonjour dear colleague", True),
        ("the audio is clear", "the audio is clear", "the audio is clear", True),
        ("let's start the meeting", "let's start the meeting", "let's start the meeting", True),
    ]
    
    for raw, proposed, reference, safe in no_change_cases:
        add_case(raw, proposed, reference, "no_change_case", safe)
    
    # ============================================================
    # 9. Multilingual cases
    # ============================================================
    multilingual_cases = [
        ("wo men yong piano note", "wo men yong pyannote", "wo men yong pyannote", "zh-CN", True),
        ("shuo hua ren ri ji zhan", "shuo hua ren diarization", "shuo hua ren diarization", "zh-CN", True),
        ("on utilise piano note", "on utilise pyannote", "on utilise pyannote", "fr", True),
        ("le rack glossary est utile", "le RAG glossary est utile", "le RAG glossary est utile", "fr", True),
        ("请把路由器放在 rack 上", "请把路由器放在 RAG 上", "请把路由器放在 rack 上", "zh-CN", False),
        ("我们的 where 很高", "我们的 WER 很高", "我们的 WER 很高", "zh-CN", True),
        ("le score dear est bas", "le score DER est bas", "le score DER est bas", "fr", True),
        ("utiliser piano note", "utiliser pyannote", "utiliser pyannote", "fr", True),
    ]
    
    for raw, proposed, reference, lang, safe in multilingual_cases:
        add_case(raw, proposed, reference, "technical_term_recovery", safe)
        test_cases[-1]["language"] = lang
    
    return pd.DataFrame(test_cases)


def main():
    """Main function"""
    print("Generating R2 independent heldout benchmark...")
    df = generate_heldout_dataset()
    
    print(f"\n✅ Generated {len(df)} test cases")
    print("\n📊 Label distribution:")
    print(df['binary_label'].value_counts())
    print("\n📂 Category distribution:")
    print(df['category'].value_counts())
    
    output_path = "data/pilot/binary_safe_apply_independent_heldout.csv"
    df.to_csv(output_path, index=False)
    print(f"\n💾 Saved to {output_path}")
    
    # Validate
    df_loaded = pd.read_csv(output_path)
    if len(df_loaded) >= 100:
        print(f"✅ Validation passed: {len(df_loaded)} cases (≥100)")
    else:
        print(f"⚠️ Only {len(df_loaded)} cases, need at least 100")
        print("   Consider adding more templates or increasing combinations")


if __name__ == "__main__":
    main()