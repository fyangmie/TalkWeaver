"""Constrained segment correction with API and deterministic fallbacks."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from backend.prompting import build_correction_messages, format_segment_prompt


LOGGER = logging.getLogger(__name__)
WORD_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[.-][A-Za-z0-9]+)*")
CORRECTIONS = (
    ("piano note", "pyannote"),
    ("diary station", "diarization"),
    ("the ear", "DER"),
    ("where", "WER"),
    ("rack", "RAG"),
)


@dataclass(frozen=True)
class LLMProvider:
    """OpenAI-compatible provider configuration."""

    name: str
    api_key: str
    base_url: str
    model: str


def select_provider(
    *,
    provider: str = "auto",
    openai_api_key: str = "",
    deepseek_api_key: str = "",
    qwen_api_key: str = "",
    openai_model: str = "gpt-4.1-mini",
    deepseek_model: str = "deepseek-v4-pro",
    qwen_model: str = "qwen-plus",
    openai_base_url: str = "https://api.openai.com/v1",
    deepseek_base_url: str = "https://api.deepseek.com",
    qwen_base_url: str = (
        "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ),
) -> LLMProvider | None:
    """Select the requested configured OpenAI-compatible provider."""

    candidates = {
        "openai": LLMProvider(
            "openai",
            openai_api_key,
            openai_base_url,
            openai_model,
        ),
        "deepseek": LLMProvider(
            "deepseek",
            deepseek_api_key,
            deepseek_base_url,
            deepseek_model,
        ),
        "qwen": LLMProvider(
            "qwen",
            qwen_api_key,
            qwen_base_url,
            qwen_model,
        ),
    }
    normalized = provider.strip().lower()
    if normalized != "auto":
        if normalized not in candidates:
            raise ValueError(
                "LLM provider must be auto, openai, deepseek, or qwen."
            )
        selected = candidates[normalized]
        return selected if selected.api_key else None
    return next(
        (candidate for candidate in candidates.values() if candidate.api_key),
        None,
    )


def _supports_replacement(replacement: str, terms: list[str]) -> bool:
    target = replacement.lower()
    return any(
        target == term.lower()
        or target in term.lower()
        or term.lower() in target
        for term in terms
    )


def rule_based_correction(text: str, terms: list[str]) -> str:
    """Apply only glossary substitutions supported by retrieved terms."""

    corrected = text
    for source, replacement in CORRECTIONS:
        if not _supports_replacement(replacement, terms):
            continue
        corrected = re.sub(
            rf"\b{re.escape(source)}\b",
            replacement,
            corrected,
            flags=re.IGNORECASE,
        )
    return corrected


def _extract_json_object(content: str) -> dict[str, Any]:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("LLM response did not contain a JSON object.")
    payload = json.loads(cleaned[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("LLM response JSON must be an object.")
    return payload


def _chat_completion(
    llm: LLMProvider,
    messages: list[dict[str, str]],
    *,
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    endpoint = f"{llm.base_url.rstrip('/')}/chat/completions"
    request = Request(
        endpoint,
        data=json.dumps(
            {
                "model": llm.model,
                "messages": messages,
                "temperature": 0,
                "stream": False,
            }
        ).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {llm.api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"{llm.name} returned HTTP {exc.code}: {details[:300]}"
        ) from exc
    except (URLError, TimeoutError) as exc:
        raise RuntimeError(f"{llm.name} request failed: {exc}") from exc

    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            f"{llm.name} returned an unexpected response schema."
        ) from exc
    return _extract_json_object(str(content))


def _tokens(text: str) -> list[str]:
    return [match.group(0).lower() for match in WORD_PATTERN.finditer(text)]


def validate_corrected_text(
    raw_text: str,
    corrected_text: str,
    retrieved_terms: list[str],
) -> tuple[bool, str]:
    """Reject empty, excessive, or unsupported model rewrites."""

    if not corrected_text.strip():
        return False, "Correction was empty."
    raw_tokens = _tokens(raw_text)
    corrected_tokens = _tokens(corrected_text)
    allowed_tokens = set(raw_tokens)
    for term in retrieved_terms:
        allowed_tokens.update(_tokens(term))
    for _source, replacement in CORRECTIONS:
        if _supports_replacement(replacement, retrieved_terms):
            allowed_tokens.update(_tokens(replacement))

    unsupported = sorted(set(corrected_tokens) - allowed_tokens)
    if unsupported:
        return (
            False,
            "Correction introduced unsupported tokens: "
            + ", ".join(unsupported),
        )
    if len(corrected_tokens) > len(raw_tokens) + 2:
        return False, "Correction added too much content."
    grounded_tokens = _tokens(rule_based_correction(raw_text, retrieved_terms))
    if corrected_tokens != grounded_tokens:
        return (
            False,
            "Correction changed word order or content beyond supported "
            "glossary substitutions.",
        )
    return True, "Correction passed lexical grounding checks."


def _correct_one_segment(
    segment: dict[str, Any],
    *,
    mock: bool,
    llm: LLMProvider | None,
) -> dict[str, Any]:
    raw_text = str(segment["raw_text"])
    terms = [str(term) for term in segment.get("retrieved_terms", [])]
    prompt = format_segment_prompt(segment)
    fallback_text = rule_based_correction(raw_text, terms)
    corrected_text = fallback_text
    mode = "mock_rule_based" if mock else "no_api_rule_based"
    note = "Deterministic glossary correction grounded in retrieved terms."
    validation_note = "API validation was not required."
    api_uncertain = False

    if not mock and llm is not None:
        try:
            response = _chat_completion(
                llm,
                build_correction_messages(segment),
            )
            candidate = str(response.get("corrected_text", ""))
            valid, validation_note = validate_corrected_text(
                raw_text,
                candidate,
                terms,
            )
            if not valid:
                raise ValueError(validation_note)
            corrected_text = candidate.strip()
            mode = f"api_{llm.name}"
            note = str(response.get("note", "")).strip() or validation_note
            api_uncertain = bool(response.get("uncertain", False))
        except (RuntimeError, ValueError, json.JSONDecodeError) as exc:
            LOGGER.warning(
                "LLM correction fell back to deterministic rules: %s",
                exc,
            )
            mode = f"api_{llm.name}_fallback_rule_based"
            note = f"API correction rejected or unavailable: {exc}"

    overlap = bool(segment.get("overlap"))
    correction_uncertain = (
        overlap
        or str(segment.get("speaker")) == "UNKNOWN"
        or api_uncertain
    )
    updated = dict(segment)
    updated["corrected_text"] = corrected_text
    updated["correction_mode"] = mode
    updated["correction_prompt"] = prompt
    updated["correction_uncertain"] = correction_uncertain
    updated["correction_note"] = (
        "Overlapping speech: conservative correction retained for review. "
        + note
        if overlap
        else note
    )
    updated["correction_validation"] = validation_note
    if correction_uncertain:
        updated["uncertainty"] = "uncertain"
    return updated


def correct_segments(
    segments: list[dict[str, Any]],
    *,
    mock: bool = False,
    provider: str = "auto",
    openai_api_key: str = "",
    deepseek_api_key: str = "",
    qwen_api_key: str = "",
    openai_model: str = "gpt-4.1-mini",
    deepseek_model: str = "deepseek-v4-pro",
    qwen_model: str = "qwen-plus",
    openai_base_url: str = "https://api.openai.com/v1",
    deepseek_base_url: str = "https://api.deepseek.com",
    qwen_base_url: str = (
        "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ),
) -> list[dict[str, Any]]:
    """Correct segments independently while preserving their audit fields."""

    llm = None
    if not mock:
        llm = select_provider(
            provider=provider,
            openai_api_key=openai_api_key,
            deepseek_api_key=deepseek_api_key,
            qwen_api_key=qwen_api_key,
            openai_model=openai_model,
            deepseek_model=deepseek_model,
            qwen_model=qwen_model,
            openai_base_url=openai_base_url,
            deepseek_base_url=deepseek_base_url,
            qwen_base_url=qwen_base_url,
        )
    return [
        _correct_one_segment(segment, mock=mock, llm=llm)
        for segment in segments
    ]
