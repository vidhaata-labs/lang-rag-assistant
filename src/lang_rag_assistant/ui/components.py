"""Thin wrappers over native Streamlit elements, shared by the pages."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import streamlit as st

from lang_rag_assistant.errors import ProviderError, ProviderErrorKind

AlertKind = Literal["error", "info", "warning"]

PROVIDER_ERROR_TITLES: dict[ProviderErrorKind, str] = {
    ProviderErrorKind.RATE_LIMIT: "Rate limit reached",
    ProviderErrorKind.QUOTA_EXCEEDED: "Quota exhausted",
    ProviderErrorKind.UNAVAILABLE: "Provider unavailable",
    ProviderErrorKind.AUTH_FAILED: "API key rejected",
    ProviderErrorKind.MODEL_ERROR: "Model could not answer",
}


@dataclass(frozen=True)
class Passage:
    """A cited passage as the UI shows it."""

    document_name: str
    page_number: int
    text: str


def show_alert(kind: AlertKind, title: str, body: str) -> None:
    """Show an error, info or warning alert with a headline."""
    alert = {"error": st.error, "info": st.info, "warning": st.warning}[kind]
    alert(body, title=title)


def show_provider_error(err: ProviderError) -> None:
    """Show a model failure with its wireframe title and the core's user message."""
    show_alert("error", PROVIDER_ERROR_TITLES[err.kind], err.user_message)


def render_user_message(text: str) -> None:
    with st.chat_message("user"):
        st.markdown(text)


def render_sources(sources: Sequence[Passage]) -> None:
    """Collapsed list of the passages an answer cites."""
    if not sources:
        return
    with st.expander(f"Cited passages · {len(sources)}"):
        for source in sources:
            st.caption(f"{source.document_name} · p. {source.page_number}")
            st.text(source.text)


def render_suggestions(
    questions: Sequence[str],
    *,
    key: str,
    on_pick: Callable[[str], None],
) -> None:
    """Clickable question pills. Picking one calls `on_pick` with the question."""
    if not questions:
        return

    def _picked() -> None:
        choice = st.session_state.get(key)
        if choice:
            on_pick(choice)

    st.pills(
        "Suggested questions",
        list(questions),
        selection_mode="single",
        key=key,
        on_change=_picked,
        label_visibility="collapsed",
    )


def format_recency(updated_at: datetime, now: datetime) -> str:
    """Wireframe label for when a conversation was last updated."""
    seconds = max(0, int((now - updated_at).total_seconds()))
    if seconds < 3600:
        return "Updated just now"
    hours = seconds // 3600
    if hours < 24:
        return f"Updated {hours} hour ago" if hours == 1 else f"Updated {hours} hours ago"
    days = hours // 24
    if days == 1:
        return "Updated yesterday"
    if days < 7:
        return f"Updated {days} days ago"
    return f"Updated {updated_at:%b %d}"
