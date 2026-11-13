"""History page: find saved conversations and resume one in the Chat page."""

from __future__ import annotations

from datetime import UTC, datetime

import streamlit as st

from lang_rag_assistant.ui.components import format_recency, show_alert
from lang_rag_assistant.ui.pages.chat import CONVERSATION_KEY, ERROR_KEY
from lang_rag_assistant.ui.services import Services, navigate_to


def render(services: Services) -> None:
    st.title("History")
    st.caption("Saved conversations, newest first. Resume one to continue where you left off.")
    query = st.text_input(
        "Search conversations",
        placeholder="Search conversations",
        label_visibility="collapsed",
        key="history_search",
    ).strip()
    conversations = (
        services.repo.search_conversations(query) if query else services.repo.list_conversations()
    )
    if not conversations:
        if query:
            st.caption(f"No conversations match “{query}”.")
        else:
            show_alert(
                "info",
                "No conversations yet",
                "Ask a question on the Chat page and it will appear here.",
            )
            if st.button("Go to Chat", type="primary", key="history_go_chat"):
                navigate_to("chat")
        return

    now = datetime.now(UTC)
    for conversation in conversations:
        with st.container(border=True):
            info, action = st.columns([5, 1], vertical_alignment="center")
            with info:
                st.markdown(f"**{conversation.title}**")
                updated = datetime.fromisoformat(conversation.updated_at)
                st.caption(f"{format_recency(updated, now)} · {conversation.message_count} messages")
            with action:
                if st.button("Resume", key=f"resume_{conversation.id}"):
                    st.session_state[CONVERSATION_KEY] = conversation.id
                    st.session_state.pop(ERROR_KEY, None)
                    navigate_to("chat")
