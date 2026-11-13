"""Chat and embedding model options, gated by the API keys set in the environment."""

from __future__ import annotations

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
    "list_chat_model_options",
]

PROVIDER_KEY_ENV: dict[str, str] = {
    "google_genai": "GOOGLE_API_KEY",
    "groq": "GROQ_API_KEY",
    "openai": "OPENAI_API_KEY",
}


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
    return init_chat_model(model=option.model, model_provider=option.provider, max_retries=1)


def create_embeddings(name: str) -> Any:
    """Create the local embedding model registered under `name`.

    Raises:
        ConfigurationError: `name` is not a supported embedding model.
    """
    if name not in EMBEDDING_MODELS:
        raise ConfigurationError(f"Unsupported embedding model {name!r}.")
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODELS[name])
