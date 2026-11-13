"""One chat turn: retrieve, run the agent, persist the messages, and suggest follow-ups."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)

from lang_rag_assistant.agent import (
    StepEvent,
    build_agent,
    extract_final_answer,
    map_stream_update,
)
from lang_rag_assistant.config.logging_config import get_logger
from lang_rag_assistant.config.settings import Settings
from lang_rag_assistant.database import Repository, SourceRecord
from lang_rag_assistant.errors import (
    AppError,
    ProviderError,
    ToolFailureError,
    classify_provider_error,
)
from lang_rag_assistant.retrieval import KnowledgeBase, RetrievedChunk
from lang_rag_assistant.suggestions import generate_followups
from lang_rag_assistant.tools import make_search_tool
from lang_rag_assistant.utils.text_utils import extract_text_content

logger = get_logger(__name__)

HISTORY_MESSAGES = 6
TITLE_LENGTH = 60
NO_ANSWER_TEXT = "I could not write an answer from the documents."
_CITATION = re.compile(r"\[(\d+)\]")


@dataclass(frozen=True)
class TurnResult:
    """What one chat turn produced."""

    answer: str
    sources: list[RetrievedChunk]
    followups: list[str]
    retrieved_count: int


def cited_numbers(text: str) -> list[int]:
    """Return the distinct [n] references in `text`, in order of first appearance."""
    numbers: list[int] = []
    for match in _CITATION.finditer(text):
        number = int(match.group(1))
        if number not in numbers:
            numbers.append(number)
    return numbers


def _messages_in(update: Mapping[str, Any]) -> list[BaseMessage]:
    collected: list[BaseMessage] = []
    for node_output in update.values():
        if isinstance(node_output, Mapping):
            collected.extend(node_output.get("messages") or [])
    return collected


class _ModelFailureRecorder(BaseCallbackHandler):
    """Records the first model failure in the run, from the main agent or any subagent.

    Subagent errors are otherwise turned into tool text that the model would copy into the
    answer, so the turn checks this record after every streamed step and stops at once.
    """

    def __init__(self, provider: str, model: str) -> None:
        super().__init__()
        self._provider = provider
        self._model = model
        self.error: ProviderError | None = None

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        if self.error is None:
            self.error = classify_provider_error(error, self._provider, self._model)


def _has_tool_failure(update: Mapping[str, Any]) -> bool:
    return any(
        isinstance(message, ToolMessage) and message.status == "error"
        for node_output in update.values()
        if isinstance(node_output, Mapping)
        for message in node_output.get("messages") or []
    )


def _split_item(item: Any) -> tuple[str, Any]:
    """A stream item is (mode, data) when several modes are streamed, else a bare update."""
    if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str):
        return item
    return "updates", item


def _has_ai_message(update: Mapping[str, Any]) -> bool:
    return any(
        isinstance(message, AIMessage)
        for node_output in update.values()
        if isinstance(node_output, Mapping)
        for message in node_output.get("messages") or []
    )


def _token_text(data: Any) -> str:
    """The answer text in one streamed token from the top-level model.

    Tool-call fragments, other messages, and tokens from a subagent (whose namespace is
    nested under a tool call, so it contains "|") give no answer text.
    """
    chunk, metadata = data if isinstance(data, tuple) else (data, {})
    namespace = str((metadata or {}).get("langgraph_checkpoint_ns", ""))
    if "|" in namespace:
        return ""
    if isinstance(chunk, AIMessageChunk) and not chunk.tool_call_chunks:
        return extract_text_content(chunk.content)
    return ""


class ChatService:
    """Runs chat turns against the agent and stores their messages."""

    def __init__(self, repo: Repository, kb: KnowledgeBase, settings: Settings) -> None:
        self._repo = repo
        self._kb = kb
        self._settings = settings

    def _history(self, conversation_id: int) -> list[BaseMessage]:
        history: list[BaseMessage] = []
        for message in self._repo.list_messages(conversation_id)[-HISTORY_MESSAGES:]:
            if message.role == "user":
                history.append(HumanMessage(content=message.content))
            else:
                history.append(AIMessage(content=message.content))
        return history

    def run_turn(
        self,
        *,
        conversation_id: int,
        question: str,
        option: Any,
        chat_model: Any,
        on_step: Callable[[StepEvent], None],
        on_text: Callable[[str], None] | None = None,
    ) -> TurnResult:
        """Answer `question` in a conversation, persisting both messages.

        The user message is stored before the model is called, so a failed turn
        keeps the question. Model failures are raised as ProviderError.

        Answer text streams to `on_text` token by token. If the caller is interrupted
        (the user pressed Stop), the text streamed so far is saved as a stopped answer,
        and the interruption is re-raised.
        """
        is_first_turn = not self._repo.list_messages(conversation_id)
        history = self._history(conversation_id)
        self._repo.add_message(conversation_id, "user", question)

        retrieved: list[RetrievedChunk] = []
        workspace = self._settings.data_dir / "workspaces" / str(conversation_id)
        messages: list[BaseMessage] = []
        streamed: list[str] = []  # answer text of the model call in progress
        recorder = _ModelFailureRecorder(option.provider, option.model)
        try:
            search_tool = make_search_tool(self._kb, workspace, retrieved)
            agent = build_agent(chat_model, search_tool, workspace, self._settings.max_concurrent_analysts)
            inputs = {"messages": [*history, HumanMessage(content=question)]}
            for item in agent.stream(inputs, stream_mode=["updates", "messages"], config={"callbacks": [recorder]}):
                if recorder.error is not None:
                    raise recorder.error
                mode, data = _split_item(item)
                if mode == "updates":
                    if _has_tool_failure(data):
                        raise ToolFailureError()
                    for event in map_stream_update(data):
                        on_step(event)
                    messages.extend(_messages_in(data))
                    if _has_ai_message(data):
                        # A model call finished. Its text is kept in `messages` if it is the
                        # answer, or dropped if it was narration before a tool call.
                        streamed.clear()
                else:
                    piece = _token_text(data)
                    if piece:
                        streamed.append(piece)
                        if on_text is not None:
                            on_text(piece)
        except (ProviderError, AppError):
            raise
        except Exception as exc:
            raise (recorder.error or classify_provider_error(exc, option.provider, option.model)) from exc
        except BaseException:
            # Interrupted, for example by the Stop button: keep what the user has seen.
            partial = "".join(streamed)
            self._save_stopped_answer(conversation_id, partial, retrieved, is_first_turn, question)
            raise

        answer = extract_final_answer(messages) or NO_ANSWER_TEXT
        sources = [
            (number, retrieved[number - 1])
            for number in cited_numbers(answer)
            if 1 <= number <= len(retrieved)
        ]

        followups: list[str] = []
        try:
            followups = generate_followups(chat_model, question, answer)
        except ProviderError as exc:
            logger.warning("Follow-up suggestions skipped: %s", exc.user_message)

        self._repo.add_message(
            conversation_id,
            "assistant",
            answer,
            sources=_source_records(sources),
            followups=followups,
        )
        if is_first_turn:
            self._repo.rename_conversation(conversation_id, question[:TITLE_LENGTH])

        return TurnResult(
            answer=answer,
            sources=[chunk for _, chunk in sources],
            followups=followups,
            retrieved_count=len(retrieved),
        )

    def _save_stopped_answer(
        self,
        conversation_id: int,
        partial: str,
        retrieved: list[RetrievedChunk],
        is_first_turn: bool,
        question: str,
    ) -> None:
        """Store an interrupted answer, marked stopped, with the passages it already cites."""
        sources = [
            (number, retrieved[number - 1])
            for number in cited_numbers(partial)
            if 1 <= number <= len(retrieved)
        ]
        self._repo.add_message(
            conversation_id,
            "assistant",
            partial,
            sources=_source_records(sources),
            followups=[],
            stopped=True,
        )
        if is_first_turn:
            self._repo.rename_conversation(conversation_id, question[:TITLE_LENGTH])


def _source_records(sources: list[tuple[int, RetrievedChunk]]) -> list[SourceRecord]:
    return [
        SourceRecord(chunk_id=chunk.chunk_id, document_name=chunk.document_name,
                     page_number=chunk.page_number, number=number)
        for number, chunk in sources
    ]
