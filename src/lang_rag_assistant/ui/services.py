"""Services shared by every page, built once per server process."""

from __future__ import annotations

import os
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
CHAT_OPTION_KEY = "chat_option_key"


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
    default_option_key: str | None = None

    @property
    def enabled_options(self) -> list[ChatModelOption]:
        """Chat models whose API key is set."""
        return [option for option in self.options if option.has_key]


@st.cache_resource(show_spinner="Loading the embedding model…")
def get_services() -> Services:
    """Load settings, the database, passages and the bundled documents once."""
    configure_logging()
    settings = load_settings()
    repo = Repository(settings.data_dir / "lang_rag.sqlite")
    embedding_model = repo.get_setting(EMBEDDING_SETTING) or settings.embedding_model
    kb = KnowledgeBase(repo, create_embeddings(embedding_model), embedding_model)
    options = list_chat_model_options(settings, os.environ)
    enabled = [option for option in options if option.has_key]
    default = next((o for o in enabled if o.provider == settings.llm_provider), enabled[0] if enabled else None)

    results = ingest_bundled(
        repo=repo,
        kb=kb,
        chat_model=create_chat_model(default) if default else None,
        provider=default.provider if default else "",
        model_name=default.model if default else "",
        bundled_dir=settings.bundled_pdf_dir,
    )
    warnings = []
    if settings.bundled_pdf_dir.is_dir():
        indexed = {d.name for d in repo.list_documents() if d.source_kind == "bundled"}
        for path in sorted(settings.bundled_pdf_dir.glob("*.pdf")):
            if path.name not in indexed:
                warnings.append(f"{path.name} could not be read and is not available for answers.")
    logger.info("Startup: %d bundled documents indexed", len(results))

    return Services(
        settings=settings,
        repo=repo,
        embedding_model=embedding_model,
        kb=kb,
        chat=ChatService(repo, kb, settings),
        options=options,
        startup_warnings=warnings,
        default_option_key=default.key if default else None,
    )


@st.cache_resource(show_spinner=False)
def chat_model_for(option: ChatModelOption) -> Any:
    """Return the chat model for an option, created once per option."""
    return create_chat_model(option)


def selected_option(services: Services) -> ChatModelOption | None:
    """Return the chat model chosen in this session, or the default, or None when no key is set."""
    enabled = services.enabled_options
    if not enabled:
        return None
    chosen = st.session_state.get(CHAT_OPTION_KEY) or services.default_option_key
    for option in enabled:
        if option.key == chosen:
            return option
    st.session_state[CHAT_OPTION_KEY] = enabled[0].key
    return enabled[0]
