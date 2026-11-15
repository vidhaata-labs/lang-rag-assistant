"""Services shared by every page, built once per server process, plus navigation helpers."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import streamlit as st

from lang_rag_assistant.chat_service import ChatService
from lang_rag_assistant.config.logging_config import configure_logging, get_logger
from lang_rag_assistant.config.model_config import (
    ChatModelOption,
    create_chat_model,
    create_embeddings,
    list_chat_model_options,
)
from lang_rag_assistant.config.settings import Settings, load_settings
from lang_rag_assistant.database import Repository
from lang_rag_assistant.ingest import ingest_bundled
from lang_rag_assistant.retrieval import KnowledgeBase

logger = get_logger(__name__)

EMBEDDING_SETTING = "embedding_model"
CHAT_OPTION_SETTING = "chat_option_key"
PROVIDER_LABELS: dict[str, str] = {
    "google_genai": "Google GenAI",
    "groq": "Groq",
    "openai": "OpenAI",
    "requesty": "Requesty",
}
_PAGES_KEY = "_app_pages"


def option_label(option: ChatModelOption) -> str:
    """The label shown in pickers, for example "gpt-4o-mini · OpenAI"."""
    return f"{option.model} · {PROVIDER_LABELS[option.provider]}"


@dataclass
class Services:
    """Long-lived objects used by the pages."""

    settings: Settings
    repo: Repository
    embedding_model: str
    kb: KnowledgeBase
    chat: ChatService
    options: list[ChatModelOption]
    startup_warnings: list[str] = field(default_factory=list)

    @property
    def enabled(self) -> list[ChatModelOption]:
        """Chat models whose provider has an API key."""
        return [option for option in self.options if option.has_key]


def build_services(
    settings: Settings,
    env: Mapping[str, str],
    *,
    embeddings: Any | None = None,
) -> Services:
    """Open the database, load the embedding model and passages, and index the bundled PDFs.

    `embeddings` is injected by tests; otherwise the embedding model recorded by the
    Model page (or the configured default) is loaded.
    """
    repo = Repository(settings.data_dir / "lang_rag.sqlite")
    embedding_model = repo.get_setting(EMBEDDING_SETTING) or settings.embedding_model
    if embeddings is None:
        embeddings = create_embeddings(embedding_model)
    kb = KnowledgeBase(repo, embeddings, embedding_model)
    options = list_chat_model_options(settings, env)
    enabled = [option for option in options if option.has_key]
    default = _default_option(settings, enabled)
    results = ingest_bundled(
        repo=repo,
        kb=kb,
        chat_model=create_chat_model(default) if default else None,
        provider=default.provider if default else "",
        model_name=default.model if default else "",
        bundled_dir=settings.bundled_pdf_dir,
    )
    warnings: list[str] = []
    if settings.bundled_pdf_dir.is_dir():
        indexed = {d.name for d in repo.list_documents() if d.source_kind == "bundled"}
        for path in sorted(settings.bundled_pdf_dir.glob("*.pdf")):
            if path.name not in indexed:
                warnings.append(f"{path.name} could not be read and is not available for answers.")
    logger.info("Startup: %d bundled documents processed", len(results))
    return Services(
        settings=settings,
        repo=repo,
        embedding_model=embedding_model,
        kb=kb,
        chat=ChatService(repo, kb, settings),
        options=options,
        startup_warnings=warnings,
    )


def _default_option(settings: Settings, enabled: list[ChatModelOption]) -> ChatModelOption | None:
    """The configured provider's model if it has a key, else the first enabled model."""
    return next((o for o in enabled if o.provider == settings.llm_provider), enabled[0] if enabled else None)


@st.cache_resource(show_spinner="Loading the embedding model…")
def get_services() -> Services:
    """Load settings, the database, passages and the bundled documents once."""
    configure_logging()
    return build_services(load_settings(), os.environ)


@st.cache_resource(show_spinner=False)
def chat_model_for(option: ChatModelOption) -> Any:
    """The chat model for an option, created once per option."""
    return create_chat_model(option)


def selected_option(services: Services) -> ChatModelOption | None:
    """The chat model in use: the saved choice if it still has a key, else the default.

    Returns None when no model has a key, or when the saved choice has lost its key.
    """
    saved = services.repo.get_setting(CHAT_OPTION_SETTING)
    enabled = services.enabled
    if saved is None:
        return _default_option(services.settings, enabled)
    return next((o for o in enabled if o.key == saved), None)


def saved_option_key(services: Services) -> str | None:
    """The `provider:model` key the user saved on the Model page, if any."""
    return services.repo.get_setting(CHAT_OPTION_SETTING)


def save_selection(services: Services, option: ChatModelOption) -> None:
    services.repo.set_setting(CHAT_OPTION_SETTING, option.key)


def register_pages(pages: dict[str, st.Page]) -> None:
    """Remember the page objects so any page can switch to another."""
    st.session_state[_PAGES_KEY] = pages


def navigate_to(name: str) -> None:
    """Switch to a registered page by name ("chat", "history", "documents", "model")."""
    st.switch_page(st.session_state[_PAGES_KEY][name])
