"""Chat and embedding model options, gated by the API keys set in the environment."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from lang_rag_assistant.config.settings import (
    EMBEDDING_MODELS,
    SUPPORTED_PROVIDERS,
    Settings,
)
from lang_rag_assistant.errors import ConfigurationError

__all__ = [
    "EMBEDDING_MODELS",
    "PROVIDER_KEY_ENV",
    "ChatModelOption",
    "create_chat_model",
    "create_embeddings",
    "friendly_model_name",
    "list_chat_model_options",
]

PROVIDER_KEY_ENV: dict[str, str] = {
    "google_genai": "GOOGLE_API_KEY",
    "groq": "GROQ_API_KEY",
    "openai": "OPENAI_API_KEY",
    "requesty": "REQUESTY_API_KEY",
    "nvidia": "NVIDIA_API_KEY",
}

# Requesty is an OpenAI-compatible gateway. The free-tier default model is set in settings.
REQUESTY_BASE_URL = "https://router.requesty.ai/v1"
# NVIDIA Integrate API is OpenAI-compatible.
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

# Short display names for the configured defaults. Lookup falls back to a
# prettified form of the raw model id so custom overrides still read cleanly.
FRIENDLY_MODEL_NAMES: dict[str, str] = {
    "gemini-3.1-flash-lite": "Gemini 3.1 Flash Lite",
    "llama-3.1-8b-instant": "Llama 3.1 8B Instant",
    "gpt-4o-mini": "GPT-4o mini",
    "nemotron-3-ultra-550b-a55b": "Nemotron 3 Ultra 550B",
    "nvidia/nemotron-3-ultra-550b-a55b": "Nemotron 3 Ultra 550B",
}

_TOKEN_DISPLAY: dict[str, str] = {
    "gpt": "GPT",
    "llama": "Llama",
    "gemini": "Gemini",
    "nemotron": "Nemotron",
    "ultra": "Ultra",
    "flash": "Flash",
    "lite": "Lite",
    "instant": "Instant",
    "mini": "mini",
    "nano": "nano",
}


def friendly_model_name(model: str) -> str:
    """Short, readable name for a raw model id, e.g. "GPT-4o mini"."""
    if model in FRIENDLY_MODEL_NAMES:
        return FRIENDLY_MODEL_NAMES[model]
    base = model.split("/")[-1].replace("_", "-")
    parts: list[str] = []
    for token in base.split("-"):
        if not token:
            continue
        lower = token.lower()
        if lower in _TOKEN_DISPLAY:
            parts.append(_TOKEN_DISPLAY[lower])
        elif token.isdigit() or token[0].isdigit():
            parts.append(token)
        else:
            parts.append(token[:1].upper() + token[1:])
    return " ".join(parts) if parts else model


@dataclass(frozen=True)
class ChatModelOption:
    """One selectable chat model: one per provider, usable only when its key is set."""

    provider: str
    model: str
    api_key_env: str
    has_key: bool

    @property
    def key(self) -> str:
        """Stable identifier, `provider:model`."""
        return f"{self.provider}:{self.model}"

    @property
    def display_name(self) -> str:
        """Friendly model name, e.g. "GPT-4o mini"."""
        return friendly_model_name(self.model)


def list_chat_model_options(settings: Settings, env: Mapping[str, str]) -> list[ChatModelOption]:
    """Return one option per supported provider, in SUPPORTED_PROVIDERS order."""
    return [
        ChatModelOption(
            provider=provider,
            model=settings.chat_model_names[provider],
            api_key_env=PROVIDER_KEY_ENV[provider],
            has_key=bool((env.get(PROVIDER_KEY_ENV[provider]) or "").strip()),
        )
        for provider in SUPPORTED_PROVIDERS
    ]


def create_chat_model(option: ChatModelOption) -> Any:
    """Create the LangChain chat model for an option whose key is set.

    Raises:
        ConfigurationError: the option's API key is not set.
    """
    if not option.has_key:
        raise ConfigurationError(f"{option.api_key_env} is not set, so {option.model} cannot be used.")
    from langchain.chat_models import init_chat_model

    # One retry at most: the SDKs retry rate-limited calls with back-off, and a
    # long silent wait before the error is shown is worse than a plain message.
    model_provider = option.provider
    extra: dict[str, Any] = {}
    if option.provider == "requesty":
        # LangChain has no Requesty integration. The router speaks the OpenAI API.
        model_provider = "openai"
        key = os.environ.get(option.api_key_env, "").strip()
        if not key:
            raise ConfigurationError(f"{option.api_key_env} is not set, so {option.model} cannot be used.")
        extra["api_key"] = key
        extra["base_url"] = REQUESTY_BASE_URL
    elif option.provider == "nvidia":
        # NVIDIA Integrate API is OpenAI-compatible.
        model_provider = "openai"
        key = os.environ.get(option.api_key_env, "").strip()
        if not key:
            raise ConfigurationError(f"{option.api_key_env} is not set, so {option.model} cannot be used.")
        extra["api_key"] = key
        # Allow override via NVIDIA_BASE_URL for testing/custom endpoints.
        base_url = (os.environ.get("NVIDIA_BASE_URL") or "").strip() or NVIDIA_BASE_URL
        extra["base_url"] = base_url
    return init_chat_model(
        model=option.model, model_provider=model_provider, max_retries=1, **extra
    )


def create_embeddings(name: str) -> Any:
    """Create the local embedding model registered under `name`.

    Raises:
        ConfigurationError: `name` is not a supported embedding model.
    """
    if name not in EMBEDDING_MODELS:
        raise ConfigurationError(f"Unsupported embedding model {name!r}.")
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODELS[name])
