"""Reusable rendering helpers that reproduce the wireframe components."""

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
NAV_ITEMS: tuple[tuple[str, str], ...] = (
    ("chat", "Chat"),
    ("history", "History"),
    ("documents", "Documents"),
    ("model", "Model and keys"),
)
CONVERSATION_LIMIT = 5


def error_card(title: str, detail: str) -> None:
    """Show a provider or document failure as the wireframe's danger card."""
    st.html(
        f'<div class="lrag-error"><h4>{html.escape(title)}</h4>'
        f"<div>{html.escape(detail)}</div></div>"
    )


def no_key_banner() -> None:
    """Show the no-key state as the wireframe's danger card."""
    error_card(NO_KEY_TITLE, NO_KEY_TEXT)


def citation_chips(items: Sequence[tuple[int, str, int]]) -> None:
    """Show `[n] document · p. page` chips for cited passages."""
    if not items:
        return
    chips = "".join(
        f'<span class="lrag-chip"><b>[{number}]</b> {html.escape(name)} · p. {page}</span>'
        for number, name, page in items
    )
    st.html(f"<div>{chips}</div>")


def document_chips(names: Sequence[str]) -> None:
    """Show plain chips naming the documents a conversation used."""
    if not names:
        return
    chips = "".join(f'<span class="lrag-chip">{html.escape(name)}</span>' for name in names)
    st.html(f"<div>{chips}</div>")


def passage_list(passages: Sequence[tuple[int, RetrievedChunk]]) -> None:
    """Show the cited passages of an answer, with their text."""
    for number, chunk in passages:
        st.markdown(f"**[{number}] {chunk.document_name}** · p. {chunk.page_number}")
        st.caption(chunk.text)


def _nav_key(name: str, current: str) -> str:
    return "nav-active" if name == current else f"nav-{name}"


def render_sidebar(services: Services, current: str) -> None:
    """Sidebar per wireframe: brand, actions, and either conversations or navigation.

    The chat page lists recent conversations and the chat model. Other pages list
    the page navigation, as in the History, Documents and Model frames.
    """
    pages = {name: page_ref(name) for name, _ in NAV_ITEMS}
    with st.sidebar:
        st.html('<div class="lrag-brand">Lang RAG <span>·</span> Assistant</div>')
        if st.button("+ New conversation", key="sidebar-new", width="stretch"):
            st.session_state.pop("conversation_id", None)
            st.switch_page(pages["chat"])

        if current == "chat":
            st.html('<div class="lrag-label">Conversations</div>')
            active = st.session_state.get("conversation_id")
            for summary in services.repo.list_conversations(limit=CONVERSATION_LIMIT):
                key = "recent-active" if summary.id == active else f"recent-{summary.id}"
                if st.button(summary.title, key=key, width="stretch"):
                    st.session_state["conversation_id"] = summary.id
                    st.switch_page(pages["chat"])
            if st.button("See all history", key="sidebar-history"):
                st.switch_page(pages["history"])
            option = selected_option(services)
            st.html('<div class="lrag-label" style="margin-top:24px">Chat model</div>')
            label = f"{option.provider} · {option.model}" if option else NO_KEY_TITLE
            st.html(f'<div class="lrag-mono">{html.escape(label)}</div>')
        else:
            for name, label in NAV_ITEMS:
                if st.button(label, key=_nav_key(name, current), width="stretch"):
                    st.switch_page(pages[name])
