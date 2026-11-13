"""Streamlit entry point: page registration and navigation.

Run with: uv run streamlit run src/lang_rag_assistant/main.py
"""

from __future__ import annotations

import streamlit as st

from lang_rag_assistant.config.logging_config import get_logger
from lang_rag_assistant.errors import AppError
from lang_rag_assistant.ui.pages import chat, documents, history, model
from lang_rag_assistant.ui.services import get_services, register_pages
from lang_rag_assistant.ui.theme import (
    PAGE_ICON,
    SIDEBAR_LOGO,
    apply_theme,
    render_footer,
)

logger = get_logger(__name__)


def _chat() -> None:
    chat.render(get_services())


def _history() -> None:
    history.render(get_services())


def _documents() -> None:
    documents.render(get_services())


def _model() -> None:
    model.render(get_services())


def main() -> None:
    st.set_page_config(
        page_title="Lang RAG Assistant",
        page_icon=str(PAGE_ICON),
        layout="centered",
        initial_sidebar_state="expanded",
    )
    apply_theme()
    try:
        get_services()
    except AppError as exc:
        st.error(str(exc))
        st.stop()
    except Exception:
        logger.exception("The app could not start")
        st.error("The app could not start. See the log for details.")
        st.stop()

    st.logo(str(SIDEBAR_LOGO), size="large")
    app_pages = {
        "chat": st.Page(_chat, title="Chat", icon=":material/chat:", default=True, url_path="chat"),
        "history": st.Page(_history, title="History", icon=":material/history:", url_path="history"),
        "documents": st.Page(_documents, title="Documents", icon=":material/description:", url_path="documents"),
        "model": st.Page(_model, title="Model", icon=":material/tune:", url_path="model"),
    }
    register_pages(app_pages)
    st.navigation(list(app_pages.values())).run()
    render_footer()


main()
