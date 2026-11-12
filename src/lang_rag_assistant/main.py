"""Streamlit entry point.

Run with: uv run streamlit run src/lang_rag_assistant/main.py
"""

from __future__ import annotations

import streamlit as st

from lang_rag_assistant.config.logging_config import get_logger
from lang_rag_assistant.errors import AppError
from lang_rag_assistant.ui import pages
from lang_rag_assistant.ui.components import render_sidebar
from lang_rag_assistant.ui.navigation import register_pages
from lang_rag_assistant.ui.services import get_services
from lang_rag_assistant.ui.theme import PAGE_ICON, apply_theme

logger = get_logger(__name__)

st.set_page_config(page_title="Lang RAG Assistant", page_icon=str(PAGE_ICON), layout="wide")
apply_theme()

try:
    services = get_services()
except AppError as exc:
    st.error(str(exc))
    st.stop()
except Exception:
    logger.exception("The app could not start")
    st.error("The app could not start. See the log for details.")
    st.stop()

app_pages = {
    "chat": st.Page(pages.chat_page, title="Chat", default=True),
    "history": st.Page(pages.history_page, title="History"),
    "documents": st.Page(pages.documents_page, title="Documents"),
    "model": st.Page(pages.model_page, title="Model and keys"),
}
register_pages(app_pages)
render_sidebar(services)
for warning in services.startup_warnings:
    st.sidebar.warning(warning)
navigation = st.navigation(list(app_pages.values()))
navigation.run()
