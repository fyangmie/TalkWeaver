"""TalkWeaver public website.

This Streamlit app is presentation-first: it explains the research idea,
lets visitors play audio, and visualizes ConversationMap evidence.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from webapp.components.speaker_timeline import render_anchor_timeline
from webapp.data_loader import (
    build_clip_detective_summary,
    conversation_map_label,
    list_available_conversation_maps,
    load_asr_summary,
    load_conversation_map,
    load_evidence_gate_validation_metrics,
    load_workflow_ablation,
    resolve_local_audio_path,
)
from webapp.detective_ui import anchor_table, map_stats, number
from webapp.ui_components import build_text_diff


TERM_CORRECTION_DEMO = (
    ROOT_DIR / "outputs" / "conversation_maps" / "fleurs_en_1548_conversation_map.json"
)
REAL_MULTISPEAKER_TERM_DEMO = (
    ROOT_DIR
    / "outputs"
    / "conversation_maps"
    / "earnings22_multi_speaker_term_rescue"
    / "earnings22_4481221_0000_180s_conversation_map.json"
)

LANGUAGES = {
    "zh": "中文",
    "en": "English",
    "fr": "Français",
}

TEXT: dict[str, dict[str, str]] = {
    "zh": {
        "page_title": "TalkWeaver",
        "nav_demo": "Demo",
        "nav_map": "EvidenceMap",
        "nav_results": "论文证据",
        "hero_kicker": "AI Meeting Detective",
        "hero_title": "TalkWeaver",
        "hero_subtitle": "把混乱多人语音变成可审计的对话证据地图",
        "hero_body": "TalkWeaver 不只是生成一段会议纪要。它保留谁在说、什么时候说、哪里重叠、哪些词可能被听错，以及每一次修正是否有证据。",
        "hero_primary": "打开 EvidenceMap",
        "hero_secondary": "查看论文证据",
        "value_1_title": "谁说了什么",
        "value_1_body": "把 ASR 文本连接到时间和说话人，避免一整段混在一起。",
        "value_2_title": "哪里混乱",
        "value_2_body": "标出重叠、打断和低置信度区域，提醒用户先审计再相信。",
        "value_3_title": "修正有没有证据",
        "value_3_body": "专有词修正会显示证据；证据不足时保留原文，交给用户判断。",
        "demo_title": "1. 听一段可审计音频",
        "demo_intro": "默认示例展示 TalkWeaver 如何把听错词连接到证据词；也可以切换到会议样本查看说话人和重叠证据。",
        "upload_label": "上传自己的音频",
        "upload_help": "支持 wav/mp3/m4a。上传音频只用于页面播放，证据地图使用下方已处理示例。",
        "sample_audio_title": "示例音频",
        "uploaded_audio_title": "你的音频",
        "artifact_label": "选择示例",
        "current_sample_title": "当前展示样本",
        "other_samples_title": "切换其他示例",
        "term_demo_label": "词语修正示例",
        "multispeaker_term_label": "多说话人术语修正示例",
        "meeting_demo_label": "多人会议示例",
        "playable_label": "可播放",
        "term_badge": "有词语修正",
        "overlap_badge": "有重叠证据",
        "map_title": "2. EvidenceMap：把 transcript 变成可检查证据",
        "before_after_title": "改了哪句话",
        "before_label": "改前",
        "after_label": "改后",
        "correction_preview_body": "下面是这段音频里最典型的证据修正。",
        "term_corrections_title": "词语修正",
        "term_corrections_body": "TalkWeaver 保留原始识别文本，同时显示证据支持的修正。",
        "evidence_terms": "证据词",
        "summary_title": "发生了什么",
        "speakers_title": "谁参与了",
        "crosstalk_title": "哪里有重叠/打断",
        "review_title": "哪里需要人工复核",
        "timeline_title": "时间线",
        "anchor_table": "证据锚点",
        "inspect_anchor": "查看一个锚点",
        "raw_text": "原始文本",
        "corrected_text": "修正/保留文本",
        "no_map": "暂时没有可展示的示例。",
        "results_title": "3. 项目验证结果",
        "results_intro": "这些结果说明 TalkWeaver 重点解决什么问题，以及当前系统的边界在哪里。",
        "result_meeting": "会议语音更难",
        "result_map": "EvidenceMap 已跑通",
        "result_rag": "专有词恢复更稳",
        "result_gate": "自动修正仍需谨慎",
        "language": "语言",
    },
    "en": {
        "page_title": "TalkWeaver",
        "nav_demo": "Demo",
        "nav_map": "EvidenceMap",
        "nav_results": "Evidence",
        "hero_kicker": "AI Meeting Detective",
        "hero_title": "TalkWeaver",
        "hero_subtitle": "Evidence-grounded conversation maps for chaotic multi-speaker speech",
        "hero_body": "TalkWeaver is not just a meeting-summary generator. It preserves who spoke, when they spoke, where speech overlapped, which terms may be misheard, and whether each correction is supported.",
        "hero_primary": "Open EvidenceMap",
        "hero_secondary": "View paper evidence",
        "value_1_title": "Who said what",
        "value_1_body": "Connect ASR text to time and speakers instead of showing one tangled transcript.",
        "value_2_title": "Where it got messy",
        "value_2_body": "Expose overlap, interruptions, and low-confidence regions before users trust fluent text.",
        "value_3_title": "Whether edits have evidence",
        "value_3_body": "RAG helps domain-term recovery; weak evidence is routed to review instead of forcing edits.",
        "demo_title": "1. Listen to an auditable audio clip",
        "demo_intro": "The default sample shows how TalkWeaver connects misheard words to evidence terms; meeting samples show speaker and overlap evidence.",
        "upload_label": "Upload your audio",
        "upload_help": "wav/mp3/m4a supported. Uploaded audio is played on the page; the EvidenceMap uses a processed sample below.",
        "sample_audio_title": "Sample audio",
        "uploaded_audio_title": "Your audio",
        "artifact_label": "Choose a sample",
        "current_sample_title": "Current sample",
        "other_samples_title": "Switch to another sample",
        "term_demo_label": "Term correction sample",
        "multispeaker_term_label": "Multi-speaker term correction sample",
        "meeting_demo_label": "Multi-speaker meeting sample",
        "playable_label": "playable",
        "term_badge": "term corrections",
        "overlap_badge": "overlap evidence",
        "map_title": "2. EvidenceMap: turn transcript into inspectable evidence",
        "before_after_title": "What changed",
        "before_label": "Before",
        "after_label": "After",
        "correction_preview_body": "The clearest evidence-backed corrections in this clip.",
        "term_corrections_title": "Term corrections",
        "term_corrections_body": "TalkWeaver keeps the original recognition text while showing the evidence-backed correction.",
        "evidence_terms": "Evidence terms",
        "summary_title": "What happened",
        "speakers_title": "Who spoke",
        "crosstalk_title": "Where speech overlapped",
        "review_title": "What needs review",
        "timeline_title": "Timeline",
        "anchor_table": "Evidence anchors",
        "inspect_anchor": "Inspect one anchor",
        "raw_text": "Original text",
        "corrected_text": "Corrected / retained text",
        "no_map": "No sample is available right now.",
        "results_title": "3. What the project validates",
        "results_intro": "These highlights show what TalkWeaver is built to handle and where the current system still needs caution.",
        "result_meeting": "Meeting speech is harder",
        "result_map": "EvidenceMap runs end to end",
        "result_rag": "Domain terms are recovered safely",
        "result_gate": "Automatic edits still need caution",
        "language": "Language",
    },
    "fr": {
        "page_title": "TalkWeaver",
        "nav_demo": "Démo",
        "nav_map": "EvidenceMap",
        "nav_results": "Preuves",
        "hero_kicker": "Détective de réunion IA",
        "hero_title": "TalkWeaver",
        "hero_subtitle": "Cartes de conversation vérifiables pour des réunions multi-locuteurs chaotiques",
        "hero_body": "TalkWeaver ne produit pas seulement un résumé. Il garde qui parle, quand, où les voix se chevauchent, quels termes peuvent être mal reconnus, et si chaque correction est justifiée.",
        "hero_primary": "Ouvrir EvidenceMap",
        "hero_secondary": "Voir les preuves",
        "value_1_title": "Qui a dit quoi",
        "value_1_body": "Relier le texte ASR au temps et aux locuteurs au lieu d'afficher un bloc confus.",
        "value_2_title": "Où la réunion devient confuse",
        "value_2_body": "Montrer chevauchements, interruptions et zones incertaines avant de faire confiance au texte.",
        "value_3_title": "Les corrections ont-elles une preuve",
        "value_3_body": "Le RAG aide les termes techniques; une preuve faible mène à une revue humaine.",
        "demo_title": "1. Écouter un audio vérifiable",
        "demo_intro": "L'exemple par défaut montre comment TalkWeaver relie des mots mal reconnus à des termes de preuve; les exemples de réunion montrent locuteurs et chevauchements.",
        "upload_label": "Charger votre audio",
        "upload_help": "wav/mp3/m4a acceptés. L'audio chargé est lu sur la page; l'EvidenceMap utilise un exemple déjà traité.",
        "sample_audio_title": "Audio d'exemple",
        "uploaded_audio_title": "Votre audio",
        "artifact_label": "Choisir un exemple",
        "current_sample_title": "Exemple affiché",
        "other_samples_title": "Changer d'exemple",
        "term_demo_label": "Exemple de correction de termes",
        "multispeaker_term_label": "Exemple multi-locuteurs avec correction de termes",
        "meeting_demo_label": "Exemple de réunion multi-locuteurs",
        "playable_label": "lisible",
        "term_badge": "corrections de termes",
        "overlap_badge": "preuves de chevauchement",
        "map_title": "2. EvidenceMap : rendre la transcription vérifiable",
        "before_after_title": "Ce qui a changé",
        "before_label": "Avant",
        "after_label": "Après",
        "correction_preview_body": "Les corrections les plus claires appuyées par des preuves dans cet extrait.",
        "term_corrections_title": "Corrections de termes",
        "term_corrections_body": "TalkWeaver conserve le texte reconnu original et montre la correction appuyée par des preuves.",
        "evidence_terms": "Termes de preuve",
        "summary_title": "Ce qui s'est passé",
        "speakers_title": "Qui a parlé",
        "crosstalk_title": "Où les voix se chevauchent",
        "review_title": "Ce qui demande une revue",
        "timeline_title": "Chronologie",
        "anchor_table": "Ancres de preuve",
        "inspect_anchor": "Inspecter une ancre",
        "raw_text": "Texte original",
        "corrected_text": "Texte corrigé / conservé",
        "no_map": "Aucun exemple n'est disponible pour le moment.",
        "results_title": "3. Ce que le projet valide",
        "results_intro": "Ces points montrent ce que TalkWeaver sait mettre en évidence et où le système reste prudent.",
        "result_meeting": "La parole de réunion est plus difficile",
        "result_map": "EvidenceMap fonctionne de bout en bout",
        "result_rag": "Les termes techniques sont mieux récupérés",
        "result_gate": "Les corrections automatiques restent prudentes",
        "language": "Langue",
    },
}


def tr(lang: str, key: str) -> str:
    """Translate UI copy with English fallback."""

    return TEXT.get(lang, TEXT["en"]).get(key, TEXT["en"].get(key, key))


def apply_mvp_style() -> None:
    st.markdown(
        """
        <style>
        :root {
            --tw-ink: #182025;
            --tw-muted: #5C6870;
            --tw-line: #D8E0E4;
            --tw-paper: #F7FAF9;
            --tw-teal: #10796F;
            --tw-amber: #B7791F;
            --tw-red: #B23B3B;
            --tw-blue: #2F6F8F;
        }
        .block-container {
            max-width: 1220px;
            padding-top: 1.1rem;
            padding-bottom: 3rem;
        }
        [data-testid="stSidebar"] {display: none;}
        [data-testid="stMetric"] {
            background: #FFFFFF;
            border: 1px solid var(--tw-line);
            border-left: 4px solid var(--tw-teal);
            border-radius: 8px;
            padding: 0.7rem 0.85rem;
            min-height: 92px;
        }
        [data-testid="stMetricValue"] {font-size: 1.35rem;}
        .tw-topbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 1rem;
            border-bottom: 1px solid var(--tw-line);
            padding-bottom: 0.85rem;
            margin-bottom: 1.2rem;
        }
        .tw-brand {
            font-weight: 800;
            color: var(--tw-ink);
            font-size: 1.05rem;
        }
        .tw-hero {
            background: linear-gradient(135deg, #F6FBFA 0%, #FFFFFF 58%, #F7F1E5 100%);
            border: 1px solid var(--tw-line);
            border-radius: 12px;
            padding: 2.1rem;
            margin-bottom: 1rem;
        }
        .tw-kicker {
            color: var(--tw-teal);
            font-weight: 800;
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0;
            margin-bottom: 0.35rem;
        }
        .tw-hero h1 {
            color: var(--tw-ink);
            font-size: 3.4rem;
            line-height: 1.02;
            margin: 0 0 0.45rem 0;
        }
        .tw-hero h2 {
            color: var(--tw-ink);
            font-size: 1.35rem;
            line-height: 1.35;
            margin: 0 0 0.8rem 0;
            font-weight: 700;
        }
        .tw-hero p, .tw-muted {
            color: var(--tw-muted);
            font-size: 1.02rem;
            line-height: 1.7;
        }
        .tw-panel {
            background: #FFFFFF;
            border: 1px solid var(--tw-line);
            border-radius: 8px;
            padding: 1rem;
            margin: 0.5rem 0;
        }
        .tw-panel h3 {
            margin-top: 0;
            color: var(--tw-ink);
            font-size: 1.02rem;
        }
        .tw-badge {
            display: inline-block;
            border-radius: 999px;
            padding: 0.22rem 0.58rem;
            margin: 0.1rem 0.18rem 0.1rem 0;
            font-size: 0.78rem;
            font-weight: 700;
            border: 1px solid var(--tw-line);
            background: var(--tw-paper);
            color: var(--tw-ink);
        }
        .tw-badge-green {background: #E3F3EF; color: #0E665E;}
        .tw-badge-amber {background: #FFF4D8; color: #8A5A12;}
        .tw-badge-red {background: #FCE4E2; color: #8F2E2E;}
        .tw-section {
            margin-top: 2.1rem;
            padding-top: 0.7rem;
            border-top: 1px solid var(--tw-line);
        }
        .tw-section h2 {
            color: var(--tw-ink);
            font-size: 1.55rem;
            margin-bottom: 0.25rem;
        }
        .tw-diff {
            border-left: 4px solid var(--tw-teal);
            background: #FFFFFF;
            padding: 0.9rem;
            border-radius: 6px;
            line-height: 1.65;
        }
        .tw-diff-raw {border-left-color: #71808A;}
        .tw-diff-removed {
            background: #FBE1DF;
            color: #8A2E29;
            padding: 0.05rem 0.15rem;
        }
        .tw-diff-added {
            background: #DDF1EA;
            color: #0B665D;
            padding: 0.05rem 0.15rem;
        }
        .tw-correction-card {
            background: #FFFFFF;
            border: 1px solid var(--tw-line);
            border-radius: 8px;
            padding: 0.9rem;
            margin: 0.75rem 0;
        }
        .tw-correction-label {
            display: block;
            color: var(--tw-muted);
            font-size: 0.74rem;
            font-weight: 800;
            margin-bottom: 0.2rem;
        }
        .tw-before, .tw-after {
            border-left: 4px solid var(--tw-red);
            border-radius: 6px;
            padding: 0.62rem 0.72rem;
            line-height: 1.55;
            overflow-wrap: anywhere;
        }
        .tw-before {background: #FFF2F0;}
        .tw-after {
            background: #EAF7F1;
            border-left-color: var(--tw-teal);
            margin-top: 0.55rem;
        }
        .tw-term-line {
            color: var(--tw-muted);
            font-size: 0.86rem;
            margin-top: 0.55rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _html(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def render_topbar(lang: str) -> str:
    st.markdown(
        """
        <div class="tw-topbar">
          <div class="tw-brand">TalkWeaver</div>
          <div class="tw-muted">Evidence-grounded meeting transcription</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return st.radio(
        tr(lang, "language"),
        options=list(LANGUAGES),
        format_func=lambda code: LANGUAGES[code],
        horizontal=True,
        label_visibility="collapsed",
        key="language_switch",
    )


def render_hero(lang: str) -> None:
    architecture = ROOT_DIR / "assets" / "architecture.png"
    left, right = st.columns([1.15, 0.85], vertical_alignment="center")
    with left:
        st.markdown(
            f"""
            <div class="tw-hero">
              <div class="tw-kicker">{_html(tr(lang, "hero_kicker"))}</div>
              <h1>{_html(tr(lang, "hero_title"))}</h1>
              <h2>{_html(tr(lang, "hero_subtitle"))}</h2>
              <p>{_html(tr(lang, "hero_body"))}</p>
              <span class="tw-badge tw-badge-green">EvidenceMap</span>
              <span class="tw-badge">Speaker timeline</span>
              <span class="tw-badge tw-badge-amber">Overlap alerts</span>
              <span class="tw-badge tw-badge-red">Edit audit</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        if architecture.exists():
            st.image(architecture, caption="TalkWeaver evidence flow", width="stretch")

    value_cols = st.columns(3)
    values = (
        ("value_1_title", "value_1_body", "tw-badge-green"),
        ("value_2_title", "value_2_body", "tw-badge-amber"),
        ("value_3_title", "value_3_body", "tw-badge-red"),
    )
    for column, (title_key, body_key, badge_class) in zip(value_cols, values):
        column.markdown(
            f"""
            <div class="tw-panel">
              <span class="tw-badge {badge_class}">{_html(tr(lang, title_key))}</span>
              <p>{_html(tr(lang, body_key))}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _anchor_texts(anchor: dict[str, Any]) -> tuple[str, str]:
    raw = str(anchor.get("raw_text") or "").strip()
    corrected = str(anchor.get("corrected_text") or raw).strip()
    return raw, corrected


def _changed_anchors(conversation_map: dict[str, Any]) -> list[dict[str, Any]]:
    changed: list[dict[str, Any]] = []
    for anchor in conversation_map.get("anchors", []):
        raw, corrected = _anchor_texts(anchor)
        if raw and corrected and raw != corrected:
            changed.append(anchor)
    return changed


def render_correction_cards(
    lang: str,
    conversation_map: dict[str, Any],
    *,
    limit: int = 3,
    show_body: bool = True,
) -> None:
    changed = _changed_anchors(conversation_map)
    if not changed:
        return
    st.markdown(f"### {tr(lang, 'before_after_title')}")
    if show_body:
        st.write(tr(lang, "correction_preview_body"))
    for anchor in changed[:limit]:
        raw, corrected = _anchor_texts(anchor)
        terms = ", ".join(anchor.get("retrieved_terms", [])) or "none"
        st.markdown(
            f"""
            <div class="tw-correction-card">
              <div class="tw-before">
                <span class="tw-correction-label">{_html(tr(lang, 'before_label'))}</span>
                {_html(raw)}
              </div>
              <div class="tw-after">
                <span class="tw-correction-label">{_html(tr(lang, 'after_label'))}</span>
                {_html(corrected)}
              </div>
              <div class="tw-term-line">
                <strong>{_html(tr(lang, 'evidence_terms'))}</strong>: {_html(terms)}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _has_overlap_evidence(conversation_map: dict[str, Any]) -> bool:
    metadata = conversation_map.get("metadata", {})
    if str(metadata.get("has_overlap", "")).lower() == "false":
        return False
    return any(event.get("type") == "overlap" for event in conversation_map.get("events", [])) or any(
        bool(anchor.get("overlap")) for anchor in conversation_map.get("anchors", [])
    )


def _map_score(path: Path) -> tuple[int, str]:
    data = load_conversation_map(path)
    if "_warning" in data:
        return (-1, str(path))
    metadata = data.get("metadata", {})
    score = 0
    if path.resolve() == REAL_MULTISPEAKER_TERM_DEMO.resolve():
        score += 1000
    if path.resolve() == TERM_CORRECTION_DEMO.resolve():
        score += 700
    if resolve_local_audio_path(data) is not None:
        score += 200
    score += min(len(_changed_anchors(data)) * 80, 320)
    score += min(len(data.get("term_rescues", [])) * 20, 160)
    speaker_count = str(metadata.get("speaker_count", ""))
    if speaker_count and speaker_count != "1":
        score += 120
    if metadata.get("dataset_name") in {"AMI Meeting Corpus", "AISHELL-4"}:
        score += 80
    if _has_overlap_evidence(data):
        score += 60
    if metadata.get("is_mock") is False:
        score += 40
    return (score, str(path))


def _available_maps() -> list[Path]:
    paths = list_available_conversation_maps()
    if not paths:
        return []
    return sorted(paths, key=_map_score, reverse=True)


def _recommended_map(paths: list[Path]) -> Path:
    for path in paths:
        if path.resolve() == REAL_MULTISPEAKER_TERM_DEMO.resolve():
            return path
    return paths[0]


def _sample_label(path: Path, lang: str) -> str:
    data = load_conversation_map(path)
    if "_warning" in data:
        return conversation_map_label(path)
    metadata = data.get("metadata", {})
    changed = len(_changed_anchors(data))
    term_rescues = len(data.get("term_rescues", []))
    playable = resolve_local_audio_path(data) is not None
    speaker_count = str(metadata.get("speaker_count", ""))
    dataset = metadata.get("dataset_name", "unknown source")
    if changed and speaker_count and speaker_count != "1":
        if metadata.get("dataset_name") == "Earnings-22":
            kind = tr(lang, "multispeaker_term_label")
        else:
            kind = tr(lang, "meeting_demo_label")
    elif changed:
        kind = tr(lang, "term_demo_label")
    elif _has_overlap_evidence(data):
        kind = tr(lang, "meeting_demo_label")
    else:
        kind = "EvidenceMap"
    extras = []
    if playable:
        extras.append(tr(lang, "playable_label"))
    if changed or term_rescues:
        extras.append(f"{tr(lang, 'term_badge')} x{max(changed, term_rescues)}")
    if _has_overlap_evidence(data):
        extras.append(tr(lang, "overlap_badge"))
    extra_text = f" | {', '.join(extras)}" if extras else ""
    return f"{kind} | {data.get('clip_id', path.stem)} | {dataset}{extra_text}"


def select_conversation_map(lang: str) -> dict[str, Any]:
    paths = _available_maps()
    if not paths:
        st.warning(tr(lang, "no_map"))
        return load_conversation_map(None)
    recommended = _recommended_map(paths)
    selected = recommended
    with st.expander(tr(lang, "other_samples_title"), expanded=False):
        selected = st.selectbox(
            tr(lang, "artifact_label"),
            paths,
            index=paths.index(recommended),
            format_func=lambda path: _sample_label(path, lang),
            key="sample_selector_v3",
        )
    return load_conversation_map(selected)


def render_audio_and_selector(lang: str) -> dict[str, Any]:
    st.markdown('<div class="tw-section" id="demo">', unsafe_allow_html=True)
    st.markdown(f"## {tr(lang, 'demo_title')}")
    st.write(tr(lang, "demo_intro"))
    conversation_map = select_conversation_map(lang)
    sample_audio = None
    if "_warning" not in conversation_map:
        sample_audio = resolve_local_audio_path(conversation_map)
    left, right = st.columns([0.95, 1.05])
    with left:
        if sample_audio is not None:
            st.markdown(f"**{tr(lang, 'sample_audio_title')}**")
            st.audio(str(sample_audio))
        uploaded = st.file_uploader(
            tr(lang, "upload_label"),
            type=("wav", "mp3", "m4a", "ogg", "flac"),
            help=tr(lang, "upload_help"),
        )
        if uploaded is not None:
            st.markdown(f"**{tr(lang, 'uploaded_audio_title')}**")
            st.audio(uploaded)
    with right:
        metadata = conversation_map.get("metadata", {})
        if "_warning" not in conversation_map:
            source_path = conversation_map.get("_source_path")
            current_label = (
                _sample_label(Path(source_path), lang)
                if source_path
                else str(conversation_map.get("clip_id", "ConversationMap"))
            )
            st.markdown(
                f"""
                <div class="tw-panel">
                  <span class="tw-badge tw-badge-green">{_html(tr(lang, 'current_sample_title'))}</span>
                  <p>{_html(current_label)}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            badges = [
                metadata.get("dataset_name", "unknown"),
                metadata.get("language", "unknown"),
                metadata.get("variant", "ConversationMap"),
            ]
            if _changed_anchors(conversation_map):
                badges.append(tr(lang, "term_badge"))
            if sample_audio is not None:
                badges.append(tr(lang, "playable_label"))
            st.markdown(" ".join(f'<span class="tw-badge">{_html(item)}</span>' for item in badges), unsafe_allow_html=True)
            render_correction_cards(
                lang,
                conversation_map,
                limit=3,
                show_body=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)
    return conversation_map


def _speaker_label_list(conversation_map: dict[str, Any]) -> str:
    anchors = conversation_map.get("anchors", [])
    speakers = sorted(
        {
            str(speaker)
            for anchor in anchors
            for speaker in (
                anchor.get("speakers")
                or ([anchor.get("speaker")] if anchor.get("speaker") else [])
            )
            if speaker not in {"", "UNKNOWN", "OVERLAP", "None"}
        }
    )
    return ", ".join(speakers) or "unknown"


def render_map_summary(lang: str, conversation_map: dict[str, Any]) -> None:
    if "_warning" in conversation_map:
        st.warning(conversation_map["_warning"])
        return

    st.markdown('<div class="tw-section" id="map">', unsafe_allow_html=True)
    st.markdown(f"## {tr(lang, 'map_title')}")
    metadata = conversation_map.get("metadata", {})
    stats = map_stats(conversation_map)
    detective = build_clip_detective_summary(conversation_map)

    metrics = st.columns(5)
    metrics[0].metric("Anchors", stats["anchors"])
    metrics[1].metric("Speakers", stats["speakers"])
    metrics[2].metric("Overlap", stats["overlap_events"])
    metrics[3].metric("Review", stats["needs_review"])
    metrics[4].metric("Audits", stats["audits"])

    summary_cols = st.columns(4)
    cards = (
        ("summary_title", detective["what_happened"], "tw-badge-green"),
        ("speakers_title", _speaker_label_list(conversation_map), ""),
        ("crosstalk_title", detective["where_cross_talk"], "tw-badge-amber"),
        ("review_title", detective["what_needs_review"], "tw-badge-red"),
    )
    for column, (title_key, body, badge_class) in zip(summary_cols, cards):
        column.markdown(
            f"""
            <div class="tw-panel">
              <span class="tw-badge {badge_class}">{_html(tr(lang, title_key))}</span>
              <p>{_html(body)}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    changed = _changed_anchors(conversation_map)
    if changed:
        st.markdown(f"### {tr(lang, 'term_corrections_title')}")
        st.write(tr(lang, "term_corrections_body"))
        render_correction_cards(lang, conversation_map, limit=5, show_body=False)

    st.markdown(f"### {tr(lang, 'timeline_title')}")
    render_anchor_timeline(conversation_map.get("anchors", []))

    anchors = conversation_map.get("anchors", [])
    if anchors:
        table = anchor_table(anchors)
        st.markdown(f"### {tr(lang, 'anchor_table')}")
        st.dataframe(
            table[
                [
                    "start",
                    "end",
                    "speaker",
                    "speakers",
                    "raw_text",
                    "corrected_text",
                    "overlap",
                    "needs_review",
                    "confidence",
                    "retrieved_terms",
                ]
            ],
            width="stretch",
            hide_index=True,
            column_config={
                "start": st.column_config.NumberColumn(format="%.2f s"),
                "end": st.column_config.NumberColumn(format="%.2f s"),
                "confidence": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0),
            },
        )
        selected = st.selectbox(
            tr(lang, "inspect_anchor"),
            anchors,
            format_func=lambda anchor: (
                f"{number(anchor.get('start')):.2f}-{number(anchor.get('end')):.2f}s | "
                f"{anchor.get('speaker', 'UNKNOWN')} | "
                f"{str(anchor.get('raw_text', ''))[:54]}"
            ),
        )
        diff = build_text_diff(
            selected.get("raw_text", ""),
            selected.get("corrected_text", ""),
        )
        raw_col, corrected_col = st.columns(2)
        raw_col.markdown(
            f"""
            <div class="tw-diff tw-diff-raw">
              <strong>{_html(tr(lang, 'raw_text'))}</strong><br>{diff['raw_html']}
            </div>
            """,
            unsafe_allow_html=True,
        )
        corrected_col.markdown(
            f"""
            <div class="tw-diff">
              <strong>{_html(tr(lang, 'corrected_text'))}</strong><br>{diff['corrected_html'] or '(empty)'}
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            " ".join(
                [
                    f'<span class="tw-badge">overlap={bool(selected.get("overlap"))}</span>',
                    f'<span class="tw-badge tw-badge-amber">review={bool(selected.get("needs_review"))}</span>',
                    f'<span class="tw-badge">terms={_html(", ".join(selected.get("retrieved_terms", [])) or "none")}</span>',
                ]
            ),
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


def _asr_claims() -> tuple[str, str]:
    frame = load_asr_summary()
    if frame.empty:
        return "AMI/AISHELL", "ASR summary unavailable"
    zh_fleurs = frame[
        frame["dataset_name"].eq("Google FLEURS")
        & frame["language"].eq("zh-CN")
        & frame["model_name"].eq("base")
    ]
    ami = frame[
        frame["dataset_name"].eq("AMI Meeting Corpus")
        & frame["model_name"].eq("base")
    ]
    aishell = frame[
        frame["dataset_name"].eq("AISHELL-4")
        & frame["model_name"].eq("base")
    ]
    if zh_fleurs.empty or ami.empty or aishell.empty:
        return "public evaluation", "Evaluation summary unavailable"
    return (
        f"FLEURS zh CER {zh_fleurs.iloc[0]['mean_error_rate']:.3f}",
        f"AMI WER {ami.iloc[0]['mean_error_rate']:.3f}; AISHELL CER {aishell.iloc[0]['mean_error_rate']:.3f}",
    )


def render_results(lang: str) -> None:
    st.markdown('<div class="tw-section" id="results">', unsafe_allow_html=True)
    st.markdown(f"## {tr(lang, 'results_title')}")
    st.write(tr(lang, "results_intro"))

    read_claim, meeting_claim = _asr_claims()
    workflow = load_workflow_ablation()
    evidence_gate = load_evidence_gate_validation_metrics()

    map_value = "Meeting maps"
    if not workflow.empty and "num_anchors" in workflow:
        map_value = f"{int(workflow['num_anchors'].fillna(0).sum())} anchors across workflow rows"

    gate_value = "held-out macro F1 0.325"
    if not evidence_gate.empty:
        strict = evidence_gate[
            evidence_gate["split"].eq("independent_heldout")
            & evidence_gate["feature_set"].eq("evidence_only")
            & evidence_gate["model_name"].eq("random_forest")
        ]
        if not strict.empty:
            row = strict.iloc[0]
            gate_value = f"macro F1 {row['macro_f1']:.3f}; unsafe accept {row['unsafe_accept_rate']:.3f}"

    cols = st.columns(4)
    cols[0].metric(tr(lang, "result_meeting"), meeting_claim, read_claim)
    cols[1].metric(tr(lang, "result_map"), map_value)
    cols[2].metric(
        tr(lang, "result_rag"),
        "term recall 0.833 → 1.000",
        "overall error unchanged",
    )
    cols[3].metric(tr(lang, "result_gate"), gate_value, "negative result")
    st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(
        page_title="TalkWeaver",
        page_icon="TW",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    apply_mvp_style()
    current_lang = st.session_state.get("language_switch", "zh")
    lang = render_topbar(str(current_lang))
    render_hero(lang)
    conversation_map = render_audio_and_selector(lang)
    render_map_summary(lang, conversation_map)
    render_results(lang)


if __name__ == "__main__":
    main()
