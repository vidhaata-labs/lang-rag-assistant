"""One chat turn: retrieve, run the agent, persist the messages, and suggest follow-ups."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from lang_rag_assistant.agent import (
    StepEvent,
    build_agent,
    extract_final_answer,
    map_stream_update,
)
from lang_rag_assistant.config.logging_config import get_logger
from lang_rag_assistant.config.settings import Settings
from lang_rag_assistant.database import Repository, SourceRecord
from lang_rag_assistant.errors import ProviderError, classify_provider_error
from lang_rag_assistant.retrieval import KnowledgeBase, RetrievedChunk
from lang_rag_assistant.suggestions import generate_followups
from lang_rag_assistant.tools import make_search_tool

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
    ) -> TurnResult:
        """Answer `question` in a conversation, persisting both messages.

        The user message is stored before the model is called, so a failed turn
        keeps the question. Model failures are raised as ProviderError.
        """
        is_first_turn = not self._repo.list_messages(conversation_id)
        history = self._history(conversation_id)
        self._repo.add_message(conversation_id, "user", question)

        retrieved: list[RetrievedChunk] = []
        workspace = self._settings.data_dir / "workspaces" / str(conversation_id)
        messages: list[BaseMessage] = []
        try:
            search_tool = make_search_tool(self._kb, workspace, retrieved)
            agent = build_agent(chat_model, search_tool, workspace, self._settings.max_concurrent_analysts)
            inputs = {"messages": [*history, HumanMessage(content=question)]}
            for update in agent.stream(inputs, stream_mode="updates"):
                for event in map_stream_update(update):
                    on_step(event)
                messages.extend(_messages_in(update))
        except ProviderError:
            raise
        except Exception as exc:
            raise classify_provider_error(exc, option.provider, option.model) from exc

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
            sources=[
                SourceRecord(chunk_id=chunk.chunk_id, document_name=chunk.document_name,
                             page_number=chunk.page_number, number=number)
                for number, chunk in sources
            ],
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
