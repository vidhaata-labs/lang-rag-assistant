"""Reusable rendering helpers for the Streamlit pages."""

from __future__ import annotations

import html
from collections.abc import Sequence

import streamlit as st

from lang_rag_assistant.errors import ProviderErrorKind
from lang_rag_assistant.retrieval import RetrievedChunk
from lang_rag_assistant.ui.navigation import page_ref
from lang_rag_assistant.ui.services import Services, selected_option

ERROR_TITLES: dict[ProviderErrorKind, str] = {
    ProviderErrorKind.RATE_LIMIT: "Too many requests to the model",
    ProviderErrorKind.QUOTA_EXCEEDED: "This key has reached its usage limit",
    ProviderErrorKind.AUTH_FAILED: "The API key was rejected",
    ProviderErrorKind.UNAVAILABLE: "Couldn't reach the model provider",
}
NO_KEY_TITLE = "No model has an API key"
NO_KEY_TEXT = (
    "Questions are disabled until a key is set. Add at least one of GOOGLE_API_KEY, "
    "GROQ_API_KEY or OPENAI_API_KEY to the environment file, then restart the app. "
    "Uploading and browsing documents still work."
)


def error_card(title: str, detail: str) -> None:
    """Show a provider or document failure with its exact message."""
    st.markdown(
        f'<div class="lrag-error"><h4>{html.escape(title)}</h4>'
        f"<div>{html.escape(detail)}</div></div>",
        unsafe_allow_html=True,
    )


def citation_chips(items: Sequence[tuple[int, str, int]]) -> None:
    """Show `[n] document · p. page` chips for cited passages."""
    if not items:
        return
    chips = "".join(
        f'<span class="lrag-chip"><b>[{number}]</b> {html.escape(name)} · p. {page}</span>'
        for number, name, page in items
    )
    st.markdown(chips, unsafe_allow_html=True)


def document_chips(names: Sequence[str]) -> None:
    """Show plain chips naming the documents a conversation used."""
    if not names:
        return
    chips = "".join(f'<span class="lrag-chip">{html.escape(name)}</span>' for name in names)
    st.markdown(chips, unsafe_allow_html=True)


def passage_list(passages: Sequence[tuple[int, RetrievedChunk]]) -> None:
    """Show the cited passages of an answer, with their text."""
    for number, chunk in passages:
        st.markdown(f"**[{number}] {chunk.document_name}** · p. {chunk.page_number}")
        st.caption(chunk.text)


def no_key_banner() -> None:
    """Warn that no chat model can be used."""
    st.warning(f"**{NO_KEY_TITLE}.** {NO_KEY_TEXT}")


def render_sidebar(services: Services) -> None:
    """Brand, primary actions, the five most recent conversations, and the chat model."""
    with st.sidebar:
        st.markdown('<div class="lrag-brand">Lang RAG <span>·</span> Assistant</div>', unsafe_allow_html=True)
        if st.button("New conversation", key="sidebar-new"):
            st.session_state.pop("conversation_id", None)
            st.switch_page(page_ref("chat"))
        st.markdown('<div class="lrag-label">Conversations</div>', unsafe_allow_html=True)
        for summary in services.repo.list_conversations(limit=5):
            if st.button(summary.title, key=f"recent-{summary.id}"):
                st.session_state["conversation_id"] = summary.id
                st.switch_page(page_ref("chat"))
        if st.button("See all history", key="sidebar-history"):
            st.switch_page(page_ref("history"))
        if st.button("Add a PDF", key="sidebar-add"):
            st.switch_page(page_ref("documents"))
        st.markdown('<div class="lrag-label" style="margin-top:16px">Chat model</div>', unsafe_allow_html=True)
        option = selected_option(services)
        label = option.model if option else NO_KEY_TITLE
        st.markdown(f'<div class="lrag-mono">{html.escape(label)}</div>', unsafe_allow_html=True)
