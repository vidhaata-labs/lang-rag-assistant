"""Chat page: ask questions, read answers with their cited passages, and recover from errors."""

from __future__ import annotations

import streamlit as st

from lang_rag_assistant.agent import StepEvent
from lang_rag_assistant.config.logging_config import get_logger
from lang_rag_assistant.config.model_config import ChatModelOption
from lang_rag_assistant.database import MessageRecord
from lang_rag_assistant.errors import AppError, ProviderError, ProviderErrorKind
from lang_rag_assistant.ui.components import (
    Passage,
    render_sources,
    render_suggestions,
    render_user_message,
    show_alert,
    show_provider_error,
)
from lang_rag_assistant.ui.services import (
    Services,
    chat_model_for,
    navigate_to,
    option_help,
    option_label,
    provider_line,
    save_selection,
    selected_option,
)

CONVERSATION_KEY = "conversation_id"
PENDING_KEY = "pending_question"
ERROR_KEY = "turn_error"
EMPTY_STATE_QUESTIONS = 3
NEW_CONVERSATION_TITLE = "New conversation"
logger = get_logger(__name__)
RETRYABLE_KINDS = {ProviderErrorKind.RATE_LIMIT, ProviderErrorKind.UNAVAILABLE, ProviderErrorKind.MODEL_ERROR}


def render(services: Services) -> None:
    option = selected_option(services)
    _sidebar(services, option)
    st.title("Ask your documents")
    st.caption("Answers cite the passage they came from.")
    if option is None:
        _no_model(services)

    conversation_id = st.session_state.get(CONVERSATION_KEY)
    messages = services.repo.list_messages(conversation_id) if conversation_id is not None else []
    if messages:
        _transcript(services, messages)
    else:
        _empty_state(services)

    pending = st.session_state.pop(PENDING_KEY, None)
    typed = st.chat_input(
        "Ask a question about your documents",
        disabled=option is None,
        submit_mode="stop",
    )
    question = (pending or typed or "").strip()
    if option is not None and question:
        _answer(services, option, question)
    else:
        _turn_error_actions()


def _queue(question: str) -> None:
    """Callback: ask `question` on the next run."""
    st.session_state[PENDING_KEY] = question


def _retry(question: str) -> None:
    """Callback: clear the failed turn and ask `question` again."""
    st.session_state.pop(ERROR_KEY, None)
    _queue(question)


def _start_new_chat() -> None:
    """Callback: leave the current conversation so the next question starts a new one."""
    for key in (CONVERSATION_KEY, ERROR_KEY, PENDING_KEY):
        st.session_state.pop(key, None)


def _on_model_change(services: Services) -> None:
    label = st.session_state["chat_model_pick"]
    save_selection(services, next(o for o in services.enabled if option_label(o) == label))


def _sidebar(services: Services, option: ChatModelOption | None) -> None:
    with st.sidebar:
        enabled = services.enabled
        if not enabled:
            st.selectbox("Chat model", ["No models available"], disabled=True, key="chat_model_empty")
        else:
            st.selectbox(
                "Chat model",
                [option_label(o) for o in enabled],
                index=enabled.index(option) if option in enabled else None,
                help=option_help(option) if option else None,
                key="chat_model_pick",
                on_change=_on_model_change,
                args=(services,),
            )
            if option is not None:
                st.html(provider_line(option))
        st.button(
            "New chat",
            key="new_chat",
            icon=":material/add:",
            width="stretch",
            on_click=_start_new_chat,
        )


def _no_model(services: Services) -> None:
    if not services.enabled:
        show_alert(
            "info",
            "No model configured",
            "Add an API key for Google GenAI, Groq, OpenAI, Requesty or NVIDIA in .env, restart the app, "
            "then choose a model on the Model page.",
        )
    else:
        show_alert(
            "info",
            "Choose a model",
            "Your saved model is no longer available. Choose one on the Model page.",
        )
    if st.button("Open Model page", type="primary", key="open_model"):
        navigate_to("model")


def _empty_state(services: Services) -> None:
    questions = [
        question
        for document in services.repo.list_documents()
        for question in document.questions
    ][:EMPTY_STATE_QUESTIONS]
    if questions:
        st.caption("Try asking")
        render_suggestions(questions, key="suggest_empty", on_pick=_queue)


def _passages(services: Services, message: MessageRecord) -> list[Passage]:
    """The cited passages of a stored answer, with their text."""
    if not message.sources:
        return []
    texts = {chunk.chunk_id: chunk.text for chunk in services.repo.load_chunks(services.embedding_model)}
    return [
        Passage(source.document_name, source.page_number, texts.get(source.chunk_id, ""))
        for source in message.sources
    ]


def _transcript(services: Services, messages: list[MessageRecord]) -> None:
    for message in messages:
        if message.role == "user":
            render_user_message(message.content)
            continue
        with st.chat_message("assistant"):
            st.markdown(message.content)
            if message.stopped:
                st.caption("Stopped")
            render_sources(_passages(services, message))
    last = messages[-1]
    if last.role == "assistant" and last.followups:
        render_suggestions(last.followups, key="followups_last", on_pick=_queue)


def _turn_error_actions() -> None:
    """A failed turn stays visible with its recovery actions until the user acts on it."""
    state = st.session_state.get(ERROR_KEY)
    if state is None:
        return
    err: AppError = state["error"]
    with st.chat_message("assistant"):
        if isinstance(err, ProviderError):
            show_provider_error(err)
        else:
            show_alert("error", "Couldn't answer", err.user_message)
        retryable = not isinstance(err, ProviderError) or err.kind in RETRYABLE_KINDS
        if retryable:
            st.button("Try again", key="retry_turn", on_click=_retry, args=(state["question"],))
        if isinstance(err, ProviderError) and st.button("Change model", key="change_model"):
            navigate_to("model")


def _answer(services: Services, option: ChatModelOption, question: str) -> None:
    """Run one turn, showing its steps, its answer and the passages it cites."""
    st.session_state.pop(ERROR_KEY, None)
    conversation_id = st.session_state.get(CONVERSATION_KEY)
    if conversation_id is None:
        conversation_id = services.repo.create_conversation(NEW_CONVERSATION_TITLE)
        st.session_state[CONVERSATION_KEY] = conversation_id
    render_user_message(question)
    with st.chat_message("assistant"):
        status = st.status("Searching documents", expanded=False)
        placeholder = st.empty()
        streamed: list[str] = []

        def on_step(event: StepEvent) -> None:
            status.update(label=event.label)

        def on_text(piece: str) -> None:
            # Tokens stream into the answer as they arrive. Pressing Stop interrupts here,
            # and the core saves what has streamed so far as a stopped answer.
            streamed.append(piece)
            placeholder.markdown("".join(streamed))

        try:
            result = services.chat.run_turn(
                conversation_id=conversation_id,
                question=question,
                option=option,
                chat_model=chat_model_for(option),
                on_step=on_step,
                on_text=on_text,
            )
        except AppError as err:
            status.update(label="Could not answer", state="error")
            st.session_state[ERROR_KEY] = {"question": question, "error": err}
            # Rerun so the error is drawn with its actions on the same path as later runs.
            st.rerun()
        except Exception:  # never show an exception in the chat; the details go to the log
            logger.exception("The answer failed")
            status.update(label="Could not answer", state="error")
            st.session_state[ERROR_KEY] = {"question": question, "error": AppError()}
            st.rerun()
        status.update(label="Done", state="complete")
        placeholder.markdown(result.answer)
        render_sources([Passage(c.document_name, c.page_number, c.text) for c in result.sources])
        render_suggestions(result.followups, key="followups_turn", on_pick=_queue)
