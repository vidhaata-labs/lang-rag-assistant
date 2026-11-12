"""Registry of page objects, so any page can switch to another page."""

from __future__ import annotations

import streamlit as st

_PAGES_KEY = "_app_pages"


def register_pages(pages: dict[str, st.Page]) -> None:
    """Store the page objects for the current run (called by the entry script)."""
    st.session_state[_PAGES_KEY] = pages


def page_ref(name: str) -> st.Page:
    """Return the registered page object for `name` (for example "chat")."""
    return st.session_state[_PAGES_KEY][name]
