"""Deep agent construction, and the mapping from streamed updates to UI step events.

Importing this module has no side effects: nothing is built or invoked until
`build_agent` is called.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from deepagents.middleware.filesystem import FilesystemPermission
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from lang_rag_assistant.prompts import (
    CHUNK_ANALYST_INSTRUCTIONS,
    RAG_WORKFLOW_INSTRUCTIONS,
    SUBAGENT_DELEGATION_INSTRUCTIONS,
)
from lang_rag_assistant.utils.text_utils import extract_text_content

SEARCH_TOOL = "search_documentation"
TODO_TOOL = "write_todos"
TASK_TOOL = "task"
_SAVED_COUNT = re.compile(r"Saved (\d+) documentation chunks")


@dataclass(frozen=True)
class StepEvent:
    """One visible step of a running answer, shown in the status panel."""

    key: str  # "plan" | "search" | "read"
    label: str
    detail: str


def build_agent(
    chat_model: Any,
    search_tool: Any,
    workspace_root: Path,
    max_concurrent_analysts: int,
) -> Any:
    """Build the deep agent that searches, reads passages, and answers.

    The agent can read the workspace but not write to it: passages are written
    by the search tool through the backend, not by the agent.
    """
    instructions = "\n\n".join(
        [
            RAG_WORKFLOW_INSTRUCTIONS,
            SUBAGENT_DELEGATION_INSTRUCTIONS.format(max_concurrent_analysts=max_concurrent_analysts),
        ]
    )
    return create_deep_agent(
        model=chat_model,
        tools=[search_tool],
        system_prompt=instructions,
        backend=FilesystemBackend(root_dir=workspace_root, virtual_mode=True),
        permissions=[FilesystemPermission(operations=["write"], paths=["/**"], mode="deny")],
        subagents=[
            {
                "name": "chunk-analyst",
                "description": (
                    "Analyzes one retrieved passage file. "
                    "Pass the user question and a single file path under /retrieved/."
                ),
                "system_prompt": CHUNK_ANALYST_INSTRUCTIONS,
                "model": chat_model,
            }
        ],
    )


def _call_event(name: str) -> StepEvent | None:
    if name == TODO_TOOL:
        return StepEvent(key="plan", label="Planned search queries", detail="")
    if name == SEARCH_TOOL:
        return StepEvent(key="search", label="Searching the documents", detail="")
    if name == TASK_TOOL:
        return StepEvent(key="read", label="Reading a passage", detail="")
    return None


def _search_detail(message: ToolMessage) -> str:
    match = _SAVED_COUNT.search(str(message.content))
    count = int(match.group(1)) if match else 0
    return f"{count} passage{'' if count == 1 else 's'} found"


def map_stream_update(update: Mapping[str, Any]) -> list[StepEvent]:
    """Translate one `stream_mode="updates"` chunk into UI step events."""
    events: list[StepEvent] = []
    for node_output in update.values():
        if not isinstance(node_output, Mapping):
            continue
        for message in node_output.get("messages") or []:
            if isinstance(message, AIMessage):
                for call in message.tool_calls:
                    event = _call_event(call["name"])
                    if event is not None:
                        events.append(event)
            elif isinstance(message, ToolMessage) and message.name == SEARCH_TOOL:
                events.append(
                    StepEvent(key="search", label="Searching the documents", detail=_search_detail(message))
                )
    return events


def extract_final_answer(messages: Sequence[BaseMessage]) -> str:
    """Return the text of the last assistant message that is not a tool call."""
    for message in reversed(messages):
        if isinstance(message, AIMessage) and not message.tool_calls:
            text = extract_text_content(message.content).strip()
            if text:
                return text
    return ""
