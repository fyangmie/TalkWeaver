"""Central configuration for TalkWeaver."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


def _load_dotenv_if_available() -> None:
    env_path = ROOT_DIR / ".env"
    if not env_path.is_file():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(env_path, override=False)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """Runtime settings loaded from environment variables."""

    root_dir: Path = ROOT_DIR
    asr_model_size: str = "medium"
    hf_token: str = ""
    openai_api_key: str = ""
    deepseek_api_key: str = ""
    qwen_api_key: str = ""
    llm_provider: str = "auto"
    openai_model: str = "gpt-4.1-mini"
    deepseek_model: str = "deepseek-v4-pro"
    qwen_model: str = "qwen-plus"
    openai_base_url: str = "https://api.openai.com/v1"
    deepseek_base_url: str = "https://api.deepseek.com"
    qwen_base_url: str = (
        "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    use_mock_asr: bool = False
    use_mock_diarization: bool = False
    use_mock_llm: bool = True

    @property
    def output_dir(self) -> Path:
        return self.root_dir / "outputs"

    @property
    def knowledge_base_dir(self) -> Path:
        return self.root_dir / "docs" / "knowledge_base"


def get_settings() -> Settings:
    """Build settings from the current process environment."""

    _load_dotenv_if_available()
    return Settings(
        asr_model_size=os.getenv("ASR_MODEL_SIZE", "medium"),
        hf_token=os.getenv("HF_TOKEN", ""),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        deepseek_api_key=os.getenv("DEEPSEEK_API_KEY", ""),
        qwen_api_key=os.getenv("QWEN_API_KEY", ""),
        llm_provider=os.getenv("LLM_PROVIDER", "auto"),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        deepseek_model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro"),
        qwen_model=os.getenv("QWEN_MODEL", "qwen-plus"),
        openai_base_url=os.getenv(
            "OPENAI_BASE_URL",
            "https://api.openai.com/v1",
        ),
        deepseek_base_url=os.getenv(
            "DEEPSEEK_BASE_URL",
            "https://api.deepseek.com",
        ),
        qwen_base_url=os.getenv(
            "QWEN_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
        use_mock_asr=_env_bool("USE_MOCK_ASR", False),
        use_mock_diarization=_env_bool("USE_MOCK_DIARIZATION", False),
        use_mock_llm=_env_bool("USE_MOCK_LLM", True),
    )
