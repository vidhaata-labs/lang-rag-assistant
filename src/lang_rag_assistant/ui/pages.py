"""The four app pages: Chat, History, Documents, and Model and keys."""

from __future__ import annotations

import html

import streamlit as st

from lang_rag_assistant.agent import StepEvent
from lang_rag_assistant.chat_service import cited_numbers
from lang_rag_assistant.config.model_config import ChatModelOption, create_embeddings
from lang_rag_assistant.config.settings import EMBEDDING_MODELS
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

STARTER_QUESTIONS = 4
REGENERATE_PASSAGES = 8
PENDING_KEY = "pending_question"


def _followup_buttons(questions: list[str], key_prefix: str) -> None:
    for index, text in enumerate(questions):
        if st.button(text, key=f"{key_prefix}-{index}"):
            st.session_state[PENDING_KEY] = text


def _history_messages(services: Services, conversation_id: int) -> None:
    for message in services.repo.list_messages(conversation_id):
        with st.chat_message(message.role):
            st.markdown(message.content)
            if message.role == "assistant":
                citation_chips([(s.number, s.document_name, s.page_number) for s in message.sources])
                _followup_buttons(message.followups, key_prefix=f"hist-{message.id}")


def _empty_state(services: Services) -> None:
    st.markdown("<h2>What would you like to know from these documents?</h2>", unsafe_allow_html=True)
    st.caption("Answers are written from the documents and cite the passages they came from.")
    starters = [q for doc in services.repo.list_documents() for q in doc.questions][:STARTER_QUESTIONS]
    for index, text in enumerate(starters):
        if st.button(text, key=f"starter-{index}"):
            st.session_state[PENDING_KEY] = text


def _run_turn(services: Services, option: ChatModelOption, question: str) -> None:
    conversation_id = st.session_state.get("conversation_id")
    if conversation_id is None:
        conversation_id = services.repo.create_conversation("New conversation")
        st.session_state["conversation_id"] = conversation_id

    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
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
    """Chat with the documents: open conversation, starter questions, composer."""
    services = get_services()
    option = selected_option(services)
    if option is None:
        no_key_banner()

    conversation_id = st.session_state.get("conversation_id")
    if conversation_id is None:
        _empty_state(services)
    else:
        _history_messages(services, conversation_id)

    typed = st.chat_input("Ask a question about your documents…", disabled=option is None)
    question = typed or st.session_state.pop(PENDING_KEY, None)
    if question and option is not None:
        _run_turn(services, option, question)


def history_page() -> None:
    """Search and resume past conversations."""
    services = get_services()
    st.markdown("<h1>History</h1>", unsafe_allow_html=True)
    query = st.text_input("Search conversations and answers", placeholder="Search conversations and answers")
    summaries = services.repo.search_conversations(query) if query.strip() else services.repo.list_conversations()
    if not summaries:
        st.caption("No conversations match.")
    for summary in summaries:
        with st.container(border=True):
            left, right = st.columns([5, 1])
            with left:
                st.markdown(f"**{html.escape(summary.title)}**")
                st.caption(summary.snippet)
                document_chips(summary.documents)
                st.markdown(
                    f'<div class="lrag-mono">{summary.message_count} messages · '
                    f"last active {summary.updated_at[:10]}</div>",
                    unsafe_allow_html=True,
                )
            with right:
                if st.button("Resume", key=f"resume-{summary.id}"):
                    st.session_state["conversation_id"] = summary.id
                    st.switch_page(page_ref("chat"))


def documents_page() -> None:
    """Add PDFs, list indexed documents, and manage their suggested questions."""
    services = get_services()
    option = selected_option(services)
    st.markdown("<h1>Documents</h1>", unsafe_allow_html=True)
    st.caption("PDFs the assistant can answer from. Bundled files are always available.")

    uploaded = st.file_uploader("Add a PDF", type=["pdf"], key="pdf-upload")
    if uploaded is not None and st.button("Add this PDF", key="pdf-add"):
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
            else:
                status.update(label="Document ready", state="complete")
                if result.already_indexed:
                    st.info(f"{result.name} is already indexed.")
                else:
                    st.success(f"Indexed {result.passage_count} passages from {result.name}.")
                if result.warning:
                    st.warning(result.warning)

    for doc in services.repo.list_documents():
        with st.container(border=True):
            left, middle, right = st.columns([5, 2, 2])
            tag = "Bundled" if doc.source_kind == "bundled" else "Uploaded"
            left.markdown(f"**{html.escape(doc.name)}** · {tag}")
            middle.markdown(f'<div class="lrag-mono">{doc.passage_count} passages</div>', unsafe_allow_html=True)
            if doc.source_kind == "uploaded":
                if right.button("Remove", key=f"remove-{doc.id}"):
                    services.repo.delete_document(doc.id)
                    services.kb.refresh()
                    st.rerun()
            else:
                right.caption("Remove unavailable")
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


def model_page() -> None:
    """Choose the chat model (only those with a key) and the embedding model."""
    services = get_services()
    st.markdown("<h1>Model and keys</h1>", unsafe_allow_html=True)
    enabled = services.enabled_options
    if not enabled:
        no_key_banner()

    current = selected_option(services)
    chat_choice = st.selectbox(
        "Chat model",
        options=enabled,
        index=enabled.index(current) if current in enabled else None,
        format_func=lambda o: f"{o.model} · {o.provider}",
        disabled=not enabled,
        help="Writes the answers. Only models whose API key is set can be chosen.",
    )

    st.markdown('<div class="lrag-label">API keys · read from the environment file</div>', unsafe_allow_html=True)
    for option in services.options:
        status = (
            '<span class="lrag-key-set">Key set</span>'
            if option.has_key
            else '<span class="lrag-key-missing">Key missing</span>'
        )
        st.markdown(f"{html.escape(option.provider)} · {html.escape(option.model)} · {status}", unsafe_allow_html=True)

    models = list(EMBEDDING_MODELS)
    embedding_choice = st.selectbox(
        "Embedding model",
        options=models,
        index=models.index(services.embedding_model),
        help="Turns passages into vectors for search. Changing it re-embeds every document.",
    )

    if st.button("Use this model", disabled=not enabled, key="use-model"):
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
