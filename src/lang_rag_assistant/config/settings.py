"""Application settings read from environment variables (and a .env file)."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from lang_rag_assistant.errors import ConfigurationError

PROJECT_ROOT = Path(__file__).resolve().parents[3]

SUPPORTED_PROVIDERS: tuple[str, ...] = ("google_genai", "groq", "openai", "requesty", "nvidia")
DEFAULT_PROVIDER = "google_genai"
DEFAULT_CHAT_MODELS: dict[str, str] = {
    "google_genai": "gemini-3.1-flash-lite",
    "groq": "llama-3.1-8b-instant",
    "openai": "gpt-4o-mini",
    "requesty": "nemotron-3-ultra-550b-a55b",
    "nvidia": "nvidia/nemotron-3-ultra-550b-a55b",
}
MODEL_ENV_VARS: dict[str, str] = {
    "google_genai": "GOOGLE_MODEL",
    "groq": "GROQ_MODEL",
    "openai": "OPENAI_MODEL",
    "requesty": "REQUESTY_MODEL",
    "nvidia": "NVIDIA_MODEL",
}

EMBEDDING_MODELS: dict[str, str] = {
    "BGE-M3": "BAAI/bge-m3",
    "nomic-embed-text": "nomic-ai/nomic-embed-text-v1.5",
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
}
DEFAULT_EMBEDDING = "all-MiniLM-L6-v2"

ANALYSTS_MIN = 2
ANALYSTS_MAX = 5
ANALYSTS_DEFAULT = 3

LOG_LEVELS: tuple[str, ...] = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


@dataclass(frozen=True)
class Settings:
    """Validated application settings."""

    llm_provider: str
    chat_model_names: dict[str, str]
    embedding_model: str
    max_concurrent_analysts: int
    log_level: str
    data_dir: Path
    bundled_pdf_dir: Path


def _get(env: Mapping[str, str], name: str) -> str | None:
    """Return the stripped value of `name`, or None when unset or blank."""
    value = env.get(name)
    if value is None or not value.strip():
        return None
    return value.strip()


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    """Read and validate settings.

    When `env` is None, the project `.env` file is loaded into the process
    environment first and `os.environ` is used.

    Raises:
        ConfigurationError: a value is outside its allowed set or range.
    """
    if env is None:
        load_dotenv(PROJECT_ROOT / ".env")
        env = os.environ

    provider = (_get(env, "LLM_PROVIDER") or DEFAULT_PROVIDER).lower()
    if provider not in SUPPORTED_PROVIDERS:
        raise ConfigurationError(
            f"LLM_PROVIDER must be one of {', '.join(SUPPORTED_PROVIDERS)}; got {provider!r}."
        )

    chat_models = {
        name: _get(env, MODEL_ENV_VARS[name]) or DEFAULT_CHAT_MODELS[name]
        for name in SUPPORTED_PROVIDERS
    }

    embedding = _get(env, "EMBEDDING_MODEL") or DEFAULT_EMBEDDING
    if embedding not in EMBEDDING_MODELS:
        raise ConfigurationError(
            f"EMBEDDING_MODEL must be one of {', '.join(EMBEDDING_MODELS)}; got {embedding!r}."
        )

    raw_analysts = _get(env, "MAX_CONCURRENT_ANALYSTS") or str(ANALYSTS_DEFAULT)
    try:
        analysts = int(raw_analysts)
    except ValueError as exc:
        raise ConfigurationError(
            f"MAX_CONCURRENT_ANALYSTS must be an integer; got {raw_analysts!r}."
        ) from exc
    if not ANALYSTS_MIN <= analysts <= ANALYSTS_MAX:
        raise ConfigurationError(
            f"MAX_CONCURRENT_ANALYSTS must be between {ANALYSTS_MIN} and {ANALYSTS_MAX}; got {analysts}."
        )

    log_level = (_get(env, "LOG_LEVEL") or "INFO").upper()
    if log_level not in LOG_LEVELS:
        raise ConfigurationError(f"LOG_LEVEL must be one of {', '.join(LOG_LEVELS)}; got {log_level!r}.")

    return Settings(
        llm_provider=provider,
        chat_model_names=chat_models,
        embedding_model=embedding,
        max_concurrent_analysts=analysts,
        log_level=log_level,
        data_dir=PROJECT_ROOT / "data",
        bundled_pdf_dir=PROJECT_ROOT / "pdfs",
    )
