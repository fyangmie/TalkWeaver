"""Secure optional LLM configuration for OpenAI-compatible providers."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping
from urllib.parse import urlparse

from backend.config import ROOT_DIR


CORRECTION_MODES = {
    "rule_fallback",
    "llm",
    "llm_with_rule_fallback",
}
PROMPT_VERSION = "talkweaver.correction.v1"
PROVIDER_DEFAULTS = {
    "deepseek": {
        "model": "deepseek-chat",
        "base_url": "https://api.deepseek.com",
        "key_env": "DEEPSEEK_API_KEY",
        "model_env": "DEEPSEEK_MODEL",
        "base_url_env": "DEEPSEEK_BASE_URL",
    },
    "qwen": {
        "model": "qwen-plus",
        "base_url": (
            "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ),
        "key_env": "QWEN_API_KEY",
        "model_env": "QWEN_MODEL",
        "base_url_env": "QWEN_BASE_URL",
    },
    "openai": {
        "model": "gpt-4.1-mini",
        "base_url": "https://api.openai.com/v1",
        "key_env": "OPENAI_API_KEY",
        "model_env": "OPENAI_MODEL",
        "base_url_env": "OPENAI_BASE_URL",
    },
}


def mask_api_key(api_key: str) -> str:
    """Mask a credential without exposing enough characters to reuse it."""

    value = str(api_key or "").strip()
    if not value:
        return "(not configured)"
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:3]}...{value[-2:]}"


def _load_dotenv_if_available(path: Path) -> None:
    if not path.is_file():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(path, override=False)


def _float_value(
    environment: Mapping[str, str],
    name: str,
    default: float,
) -> float:
    value = environment.get(name)
    return default if value in {None, ""} else float(value)


@dataclass(frozen=True)
class LLMConfig:
    """Validated runtime settings with a credential-safe representation."""

    provider: str
    api_key: str = field(default="", repr=False)
    model: str = ""
    base_url: str = ""
    temperature: float = 0.0
    timeout_seconds: float = 30.0
    prompt_version: str = PROMPT_VERSION

    @property
    def is_configured(self) -> bool:
        return bool(
            self.api_key
            and self.api_key != "replace_me"
            and self.provider
            and self.model
            and self.base_url
        )

    @property
    def masked_api_key(self) -> str:
        return mask_api_key(self.api_key)

    def safe_metadata(self) -> dict[str, str | float | bool]:
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "timeout_seconds": self.timeout_seconds,
            "prompt_version": self.prompt_version,
            "api_key": self.masked_api_key,
            "configured": self.is_configured,
        }

    def validate(self, *, require_api: bool) -> None:
        if self.provider not in PROVIDER_DEFAULTS:
            raise ValueError(
                "LLM_PROVIDER must be deepseek, qwen, or openai."
            )
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("LLM_TEMPERATURE must be between 0 and 2.")
        if self.timeout_seconds <= 0:
            raise ValueError("LLM_TIMEOUT_SECONDS must be greater than 0.")
        parsed = urlparse(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(
                "LLM_BASE_URL must be an absolute HTTP(S) URL."
            )
        if require_api and not self.is_configured:
            raise RuntimeError(
                "Real LLM correction requested but LLM API configuration "
                "is incomplete. Copy .env.example to .env and set "
                "LLM_API_KEY, LLM_PROVIDER, LLM_MODEL, and LLM_BASE_URL."
            )


def load_llm_config(
    *,
    correction_mode: str = "rule_fallback",
    environment: Mapping[str, str] | None = None,
    load_dotenv_file: bool = True,
    dotenv_path: str | Path | None = None,
) -> LLMConfig:
    """Load generic settings with legacy provider-specific compatibility."""

    if correction_mode not in CORRECTION_MODES:
        raise ValueError(
            "correction_mode must be rule_fallback, llm, or "
            "llm_with_rule_fallback."
        )
    if environment is None and load_dotenv_file:
        _load_dotenv_if_available(
            Path(dotenv_path) if dotenv_path else ROOT_DIR / ".env"
        )
    values = os.environ if environment is None else environment
    provider = str(values.get("LLM_PROVIDER", "deepseek")).strip().lower()
    if provider == "auto":
        provider = next(
            (
                name
                for name, defaults in PROVIDER_DEFAULTS.items()
                if values.get(str(defaults["key_env"]), "").strip()
            ),
            "deepseek",
        )
    defaults = PROVIDER_DEFAULTS.get(
        provider,
        PROVIDER_DEFAULTS["deepseek"],
    )
    api_key = str(
        values.get("LLM_API_KEY")
        or values.get(str(defaults["key_env"]), "")
    ).strip()
    model = str(
        values.get("LLM_MODEL")
        or values.get(str(defaults["model_env"]))
        or defaults["model"]
    ).strip()
    base_url = str(
        values.get("LLM_BASE_URL")
        or values.get(str(defaults["base_url_env"]))
        or defaults["base_url"]
    ).strip()
    config = LLMConfig(
        provider=provider,
        api_key=api_key,
        model=model,
        base_url=base_url,
        temperature=_float_value(
            values,
            "LLM_TEMPERATURE",
            0.0,
        ),
        timeout_seconds=_float_value(
            values,
            "LLM_TIMEOUT_SECONDS",
            30.0,
        ),
        prompt_version=str(
            values.get("LLM_PROMPT_VERSION", PROMPT_VERSION)
        ).strip(),
    )
    config.validate(require_api=correction_mode == "llm")
    return config
