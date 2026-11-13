"""Documents page: the library, and adding a PDF with its suggested questions."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

import streamlit as st

from lang_rag_assistant.config.logging_config import get_logger
from lang_rag_assistant.errors import DocumentError
from lang_rag_assistant.ingest import ingest_pdf
from lang_rag_assistant.ui.components import show_alert
from lang_rag_assistant.ui.pages.chat import PENDING_KEY
from lang_rag_assistant.ui.services import (
    Services,
    chat_model_for,
    navigate_to,
    selected_option,
)

logger = get_logger(__name__)

# The database accepts "bundled" and "uploaded" (see the documents table CHECK constraint).
UPLOAD_SOURCE_KIND = "uploaded"
SHOW_UPLOADER_KEY = "show_uploader"
UPLOAD_VERSION_KEY = "upload_version"
UPLOAD_ERROR_KEY = "upload_error"
GENERIC_FAILURE = "Something went wrong while adding this file. Try again, or choose another PDF."


def failure_message(exc: Exception) -> str:
    """User-facing text for a file that could not be added. Unexpected errors are not shown raw."""
    if isinstance(exc, DocumentError):
        return str(exc)
    return GENERIC_FAILURE


def dismiss_failed_upload(state: MutableMapping[str, Any]) -> None:
    """Forget the failed file and start a fresh, empty uploader."""
    state.pop(UPLOAD_ERROR_KEY, None)
    state[UPLOAD_VERSION_KEY] = state.get(UPLOAD_VERSION_KEY, 0) + 1


def _ask(question: str) -> None:
    st.session_state[PENDING_KEY] = question
    navigate_to("chat")


def render(services: Services) -> None:
    st.title("Documents")
    st.caption("Bundled documents ship with the app. Uploaded documents stay in this workspace.")
    for warning in services.startup_warnings:
        st.warning(warning)
    if st.button("Add a PDF", type="primary", key="add_pdf_toggle"):
        st.session_state[SHOW_UPLOADER_KEY] = not st.session_state.get(SHOW_UPLOADER_KEY, False)
    if st.session_state.get(SHOW_UPLOADER_KEY):
        _uploader(services)

    documents = services.repo.list_documents()
    if not documents:
        st.caption("No documents yet.")
        return
    columns = st.columns(3)
    for index, document in enumerate(documents):
        with columns[index % 3], st.container(border=True):
            st.markdown(f"**`{document.name}`**")
            st.badge("Bundled" if document.source_kind == "bundled" else "Uploaded")
            st.caption(
                f"{services.repo.page_count(document.id)} pages · {len(document.questions)} suggested questions"
            )
            with st.expander(f"Suggested questions · {len(document.questions)}"):
                if not document.questions:
                    st.caption("No suggested questions yet.")
                for position, question in enumerate(document.questions):
                    if st.button(question, key=f"ask_{document.id}_{position}", width="stretch"):
                        _ask(question)


def _uploader(services: Services) -> None:
    if st.session_state.get(UPLOAD_ERROR_KEY):
        failed = st.session_state[UPLOAD_ERROR_KEY]
        show_alert("error", failed["title"], failed["message"])
        st.caption(failed["file"])
        st.button("Remove file", key="remove_failed", on_click=dismiss_failed_upload, args=(st.session_state,))
        st.button(
            "Choose another PDF",
            type="primary",
            key="choose_another",
            on_click=dismiss_failed_upload,
            args=(st.session_state,),
        )
        return

    version = st.session_state.get(UPLOAD_VERSION_KEY, 0)
    uploaded = st.file_uploader(
        "PDF file",
        type=["pdf"],
        key=f"pdf_upload_{version}",
        label_visibility="collapsed",
    )
    if uploaded is None:
        return
    st.caption(f"{uploaded.name} · {uploaded.size / 1_000_000:.1f} MB")
    if st.button("Add this PDF", type="primary", key="add_pdf_confirm"):
        _add(services, uploaded)


def _add(services: Services, uploaded: Any) -> None:
    option = selected_option(services)
    with st.status(f"Indexing {uploaded.name}", expanded=True) as status:
        try:
            result = ingest_pdf(
                repo=services.repo,
                kb=services.kb,
                chat_model=chat_model_for(option) if option else None,
                provider=option.provider if option else "",
                model_name=option.model if option else "",
                source=uploaded.getvalue(),
                name=uploaded.name,
                source_kind=UPLOAD_SOURCE_KIND,
            )
        except Exception as exc:  # any failure is shown as an alert, never as a traceback
            logger.exception("Could not add %s", uploaded.name)
            title = "Couldn't read this file" if isinstance(exc, DocumentError) else "Couldn't add this file"
            status.update(label=title, state="error")
            st.session_state[UPLOAD_ERROR_KEY] = {
                "title": title,
                "message": failure_message(exc),
                "file": f"{uploaded.name} · {uploaded.size / 1_000:.0f} KB",
            }
            st.rerun()
        status.update(label="Indexed", state="complete")

    if result.already_indexed:
        st.info("This file is already in your library.", icon=":material/info:")
        return
    st.toast(f"Added {uploaded.name}")
    if result.warning:
        show_alert("warning", "Questions not generated", result.warning)
