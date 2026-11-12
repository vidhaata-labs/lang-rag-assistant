"""The four app pages, laid out as in the approved Doop wireframes."""

from __future__ import annotations

import html
from datetime import UTC, datetime
from typing import Any

import streamlit as st

from lang_rag_assistant.agent import StepEvent
from lang_rag_assistant.chat_service import cited_numbers
from lang_rag_assistant.config.model_config import ChatModelOption, create_embeddings
from lang_rag_assistant.config.settings import EMBEDDING_MODELS
from lang_rag_assistant.database import ConversationSummary
from lang_rag_assistant.errors import AppError, DocumentError, ProviderError
from lang_rag_assistant.ingest import ingest_pdf, reembed_all
from lang_rag_assistant.suggestions import generate_document_questions
from lang_rag_assistant.ui.components import (
    ERROR_TITLES,
    citation_chips,
    document_chips,
    error_card,
    no_key_banner,
    passage_list,
    render_sidebar,
)
from lang_rag_assistant.ui.navigation import page_ref
from lang_rag_assistant.ui.services import (
    CHAT_OPTION_KEY,
    EMBEDDING_SETTING,
    Services,
    chat_model_for,
    get_services,
    selected_option,
)
from lang_rag_assistant.ui.theme import page_header

PENDING_KEY = "pending_question"
STARTER_QUESTIONS = 4
REGENERATE_PASSAGES = 8
RECENT_DAYS = 7


def _ask(text: str) -> None:
    """Button callback: queue a question to be sent on the next run."""
    st.session_state[PENDING_KEY] = text


def _followup_buttons(questions: list[str], key_prefix: str) -> None:
    for index, text in enumerate(questions):
        st.button(text, key=f"{key_prefix}-{index}", on_click=_ask, args=(text,))


def _user_message(key: str, text: str) -> None:
    with st.container(key=key, width="content"):
        st.markdown(text)


def _history_messages(services: Services, conversation_id: int) -> None:
    for message in services.repo.list_messages(conversation_id):
        if message.role == "user":
            _user_message(f"usermsg-{message.id}", message.content)
            continue
        with st.container(key=f"botmsg-{message.id}"):
            st.markdown(message.content)
            citation_chips([(s.number, s.document_name, s.page_number) for s in message.sources])
            _followup_buttons(message.followups, key_prefix=f"hist-{message.id}")


def _empty_state(services: Services) -> None:
    st.html('<h2 class="lrag-hero">What would you like to know from <em>these</em> documents?</h2>')
    st.html(
        '<p class="lrag-lede">Answers are written from the documents and cite the passages they came from.</p>'
    )
    starters = [
        (doc.name, question)
        for doc in services.repo.list_documents()
        for question in doc.questions
    ][:STARTER_QUESTIONS]
    columns = st.columns(2)
    for index, (doc_name, question) in enumerate(starters):
        with columns[index % 2], st.container(key=f"starter-{index}", border=True):
            st.html(f'<div class="lrag-mono">{html.escape(doc_name)}</div>')
            st.button(question, key=f"suggest-{index}", on_click=_ask, args=(question,), width="stretch")


def _run_turn(services: Services, option: ChatModelOption, question: str) -> None:
    conversation_id = st.session_state.get("conversation_id")
    if conversation_id is None:
        conversation_id = services.repo.create_conversation("New conversation")
        st.session_state["conversation_id"] = conversation_id

    _user_message("usermsg-new", question)
    with st.container(key="botmsg-new"):
        status = st.status("Working on your answer", expanded=True)

        def on_step(event: StepEvent) -> None:
            suffix = f" · {event.detail}" if event.detail else ""
            status.write(f"{event.label}{suffix}")

        try:
            result = services.chat.run_turn(
                conversation_id=conversation_id,
                question=question,
                option=option,
                chat_model=chat_model_for(option),
                on_step=on_step,
            )
        except ProviderError as exc:
            status.update(label="Could not answer", state="error", expanded=True)
            error_card(ERROR_TITLES[exc.kind], exc.user_message)
            return
        except AppError as exc:
            status.update(label="Could not answer", state="error", expanded=True)
            error_card("Something went wrong", str(exc))
            return

        status.update(label="Answer ready", state="complete", expanded=False)
        st.markdown(result.answer)
        numbered = [n for n in cited_numbers(result.answer) if n <= result.retrieved_count]
        cited = list(zip(numbered, result.sources))
        citation_chips([(number, chunk.document_name, chunk.page_number) for number, chunk in cited])
        st.caption(f"Read {result.retrieved_count} passages · {len(cited)} cited")
        with st.expander(f"Cited passages · {len(cited)}"):
            passage_list(cited)
        _followup_buttons(result.followups, key_prefix=f"new-{conversation_id}")


def chat_page() -> None:
    """Chat with the documents: header, thread, and composer, per the Chat frames."""
    services = get_services()
    render_sidebar(services, "chat")
    option = selected_option(services)
    if option is None:
        no_key_banner()

    conversation_id = st.session_state.get("conversation_id")
    placeholder = "Ask a follow-up…" if conversation_id is not None else "Ask a question about your documents…"
    typed = st.chat_input(placeholder, disabled=option is None, key="composer")
    question = typed or st.session_state.pop(PENDING_KEY, None)

    if conversation_id is None:
        page_header("Ask the documents")
    else:
        conversation = next(
            (s for s in services.repo.list_conversations() if s.id == conversation_id), None
        )
        page_header(html.escape(conversation.title) if conversation else "Ask the documents")

    with st.container(key="thread"):
        if conversation_id is None and question is None:
            _empty_state(services)
        elif conversation_id is not None:
            _history_messages(services, conversation_id)
        if question and option is not None:
            _run_turn(services, option, question)


def _recency_label(updated_at: str) -> str:
    moment = datetime.fromisoformat(updated_at)
    return "Last 7 days" if (datetime.now(UTC) - moment).days < RECENT_DAYS else "Earlier"


def _short_date(updated_at: str) -> str:
    moment = datetime.fromisoformat(updated_at)
    return f"{moment.day} {moment.strftime('%b')}"


def history_page() -> None:
    """Search and resume past conversations, grouped by recency, per the History frame."""
    services = get_services()
    render_sidebar(services, "history")

    head_left, head_right = st.columns([3, 2], vertical_alignment="bottom")
    with head_left:
        page_header("History")
    with head_right:
        query = st.text_input(
            "Search conversations and answers",
            placeholder="Search conversations and answers",
            label_visibility="collapsed",
            key="history-search",
        )
    summaries: list[ConversationSummary] = (
        services.repo.search_conversations(query) if query.strip() else services.repo.list_conversations()
    )
    if not summaries:
        st.caption("No conversations match.")

    current_group = None
    for summary in summaries:
        group = _recency_label(summary.updated_at)
        if group != current_group:
            st.html(f'<div class="lrag-label">{group}</div>')
            current_group = group
        with st.container(key=f"history-{summary.id}", border=True, horizontal=True, vertical_alignment="center"):
            with st.container(width="stretch"):
                st.markdown(f"**{html.escape(summary.title)}**")
                st.caption(summary.snippet)
                document_chips(summary.documents)
                st.html(
                    f'<div class="lrag-mono">{summary.message_count} messages · '
                    f"last active {_short_date(summary.updated_at)}</div>"
                )
            if st.button("Resume", key=f"resume-{summary.id}", type="primary"):
                st.session_state["conversation_id"] = summary.id
                st.switch_page(page_ref("chat"))


def documents_page() -> None:
    """Add PDFs, list indexed documents, and manage suggested questions, per the Documents frames."""
    services = get_services()
    render_sidebar(services, "documents")
    option = selected_option(services)

    with st.container(key="page-head", horizontal=True, vertical_alignment="bottom"):
        with st.container(width="stretch"):
            page_header("Documents", "PDFs the assistant can answer from. Bundled files are always available.")
        with st.popover("Add a PDF", type="primary", key="add-pdf"):
            uploaded = st.file_uploader("Add a PDF", type=["pdf"], key="pdf-upload")
            if uploaded is not None and st.button("Add this PDF", key="pdf-add", type="primary"):
                _add_pdf(services, option, uploaded)

    for warning in services.startup_warnings:
        st.warning(warning)

    header = st.columns([4, 2, 2, 2, 2])
    for column, label in zip(header, ("Name", "Source", "Passages", "Status", "")):
        column.html(f'<div class="lrag-label">{label}</div>')

    for doc in services.repo.list_documents():
        row = st.columns([4, 2, 2, 2, 2], vertical_alignment="center")
        row[0].markdown(f"**{html.escape(doc.name)}**")
        tag = "Bundled" if doc.source_kind == "bundled" else "Uploaded"
        row[1].html(f'<span class="lrag-chip">{tag}</span>')
        row[2].html(f'<div class="lrag-mono">{doc.passage_count}</div>')
        row[3].html('<div class="lrag-key-set">● Indexed</div>')
        if doc.source_kind == "uploaded":
            if row[4].button("Remove", key=f"remove-{doc.id}", type="tertiary"):
                services.repo.delete_document(doc.id)
                services.kb.refresh()
                st.rerun()
        else:
            row[4].html('<div class="lrag-muted">Remove unavailable</div>')
        with st.expander(f"Suggested questions · {len(doc.questions)}"):
            for question in doc.questions:
                st.markdown(f"- {html.escape(question)}")
            if st.button("Regenerate suggestions", key=f"regen-{doc.id}", disabled=option is None):
                try:
                    questions = generate_document_questions(
                        chat_model_for(option),
                        services.repo.chunks_for_document(doc.id)[:REGENERATE_PASSAGES],
                    )
                except ProviderError as exc:
                    error_card(ERROR_TITLES[exc.kind], exc.user_message)
                else:
                    services.repo.set_suggestions(doc.id, questions)
                    st.rerun()


def _add_pdf(services: Services, option: ChatModelOption | None, uploaded: Any) -> None:
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
                source_kind="uploaded",
            )
        except DocumentError as exc:
            status.update(label="Could not add this PDF", state="error")
            error_card("This PDF cannot be used", str(exc))
            return
        status.update(label="Document ready", state="complete")
    if result.already_indexed:
        st.info(f"{result.name} is already indexed.")
    else:
        st.success(f"Indexed {result.passage_count} passages from {result.name}.")
    if result.warning:
        st.warning(result.warning)


def model_page() -> None:
    """Choose the chat model (only those with a key) and the embedding model, per the Model frame."""
    services = get_services()
    render_sidebar(services, "model")
    page_header("Model and keys")

    enabled = services.enabled_options
    if not enabled:
        no_key_banner()

    with st.container(key="model-panel", border=True):
        current = selected_option(services)
        chat_choice = st.selectbox(
            "Chat model",
            options=enabled,
            index=enabled.index(current) if current in enabled else None,
            format_func=lambda o: f"{o.model} · {o.provider}",
            disabled=not enabled,
            help="Writes the answers. Only models whose API key is set can be chosen.",
        )
        st.html('<div class="lrag-label">API keys · read from the environment file</div>')
        for option in services.options:
            status = (
                '<span class="lrag-key-set">Key set</span>'
                if option.has_key
                else '<span class="lrag-key-missing">Key missing</span>'
            )
            st.html(f"<div>{html.escape(option.provider)} · {html.escape(option.model)} · {status}</div>")

        models = list(EMBEDDING_MODELS)
        embedding_choice = st.selectbox(
            "Embedding model",
            options=models,
            index=models.index(services.embedding_model),
            help="Turns passages into vectors for search. Changing it re-embeds every document.",
        )

        if st.button("Use this model", disabled=not enabled, key="use-model", type="primary"):
            if chat_choice is not None:
                st.session_state[CHAT_OPTION_KEY] = chat_choice.key
            if embedding_choice != services.embedding_model:
                with st.spinner("Re-embedding documents…"):
                    reembed_all(
                        repo=services.repo,
                        embeddings=create_embeddings(embedding_choice),
                        embedding_model=embedding_choice,
                    )
                    services.repo.set_setting(EMBEDDING_SETTING, embedding_choice)
                st.cache_resource.clear()
                st.rerun()
            st.success("Saved.")
