"""Evidence-linked Speaker Cards view."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from webapp.data_loader import build_speaker_evidence_cards
from webapp.detective_ui import page_header, require_map, source_boundary
from webapp.ui_components import render_audio_evidence, render_text_diff


def _quote_frame(card: dict[str, Any]) -> pd.DataFrame:
    raw_by_id = {
        quote.get("anchor_id"): quote
        for quote in card.get("representative_raw_quotes", [])
    }
    corrected_by_id = {
        quote.get("anchor_id"): quote
        for quote in card.get("representative_corrected_quotes", [])
    }
    rows = []
    for anchor_id in card.get("evidence_anchor_ids", []):
        raw = raw_by_id.get(anchor_id, {})
        corrected = corrected_by_id.get(anchor_id, {})
        if not raw and not corrected:
            continue
        rows.append(
            {
                "anchor_id": anchor_id,
                "start": raw.get("start", corrected.get("start", 0.0)),
                "end": raw.get("end", corrected.get("end", 0.0)),
                "raw_quote": raw.get("text", ""),
                "corrected_quote": corrected.get("text", ""),
            }
        )
    return pd.DataFrame(rows)


def render_speaker_cards(conversation_map: dict[str, Any]) -> None:
    page_header(
        "Speaker Evidence Cards",
        "Evidence-linked speaker summaries show activity, cross-talk, quotes, and review burden without inventing opinions.",
    )
    if not require_map(conversation_map):
        return
    source_boundary(conversation_map.get("metadata", {}))
    if conversation_map.get("metadata", {}).get("diarization_mode") == "reference":
        st.warning(
            "Speaker timing for this clip is oracle/reference-assisted. It is "
            "not automatic diarization performance."
        )

    cards = build_speaker_evidence_cards(conversation_map)
    if not cards:
        st.info("No named speaker evidence is available for this clip.")
        return
    st.caption(
        "Claims and stance text are shown only when linked to source anchors. "
        "The current AMI maps use extractive fallback."
    )

    for card in cards:
        with st.expander(
            f"{card['speaker_id']} | {card['speaking_time_seconds']:.2f}s speaking time",
            expanded=True,
        ):
            metrics = st.columns(4)
            metrics[0].metric("Anchors", card["num_anchors"])
            metrics[1].metric("Overlap anchors", card["num_overlap_anchors"])
            metrics[2].metric("Needs review", card["needs_review_anchors"])
            metrics[3].metric("Summary mode", card["summary_mode"])

            st.markdown(
                "**Top evidence terms:** "
                + (", ".join(card["top_terms"]) or "none extracted")
            )
            if card["summary_mode"] == "llm_assisted":
                st.warning(
                    "This speaker summary is LLM-assisted. Use the linked "
                    "anchor IDs below to verify every claim."
                )
            else:
                st.info(
                    "Extractive fallback: the card uses timestamped anchor "
                    "quotes and does not infer an opinion or personality."
                )
            st.write(card["stance_summary"])
            st.caption(
                "Evidence anchors: "
                + (", ".join(card["evidence_anchor_ids"]) or "none")
            )
            if card["main_claims"]:
                st.markdown("**Evidence-linked claims / representative statements**")
                st.dataframe(
                    pd.DataFrame(card["main_claims"]),
                    width="stretch",
                    hide_index=True,
                )
            if card["action_items"]:
                st.markdown("**Evidence-linked action items**")
                st.dataframe(
                    pd.DataFrame(card["action_items"]),
                    width="stretch",
                    hide_index=True,
                )

            quotes = _quote_frame(card)
            if quotes.empty:
                st.info("No representative anchor quote is available.")
                continue
            st.dataframe(quotes, width="stretch", hide_index=True)
            selected_id = st.selectbox(
                "Inspect speaker evidence",
                quotes["anchor_id"].tolist(),
                key=f"speaker_quote_{card['speaker_id']}",
            )
            row = quotes[quotes["anchor_id"].eq(selected_id)].iloc[0]
            render_text_diff(row["raw_quote"], row["corrected_quote"])
            anchor = next(
                (
                    item
                    for item in conversation_map.get("anchors", [])
                    if item.get("anchor_id") == selected_id
                ),
                None,
            )
            if anchor is not None:
                render_audio_evidence(
                    conversation_map,
                    anchor,
                    item_type="anchor",
                    label=f"{card['speaker_id']} anchor audio",
                )
