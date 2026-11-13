"""Generate suggested questions for documents and related follow-ups for answers."""

from __future__ import annotations

import re
from typing import Any

from lang_rag_assistant.config.logging_config import get_logger
from lang_rag_assistant.errors import classify_provider_error
from lang_rag_assistant.utils.text_utils import extract_text_content

logger = get_logger(__name__)

MAX_CONTEXT_CHARACTERS = 12000
_LEADING_MARKER = re.compile(r"^\s*(?:\d+[.)]|[-*])\s*")


def parse_numbered_lines(text: str, n: int) -> list[str]:
    """Return up to `n` lines that are questions, with list markers removed."""
    questions: list[str] = []
    for raw in text.splitlines():
        line = _LEADING_MARKER.sub("", raw).strip()
        if line.endswith("?"):
            questions.append(line)
        if len(questions) == n:
            break
    return questions


def _ask(chat_model: Any, prompt: str) -> str:
    """Invoke the model; provider failures become ProviderError."""
    provider = getattr(chat_model, "_llm_type", "unknown")
    model_name = getattr(chat_model, "model_name", None) or getattr(chat_model, "model", "unknown")
    try:
        message = chat_model.invoke(prompt)
    except Exception as exc:
        raise classify_provider_error(exc, provider, str(model_name)) from exc
    return extract_text_content(message.content)


def generate_document_questions(chat_model: Any, chunks: list[str], n: int = 6) -> list[str]:
    """Return up to `n` questions that the passages can answer."""
    context = "\n\n".join(chunks)[:MAX_CONTEXT_CHARACTERS]
    prompt = (
        f"Write exactly {n} short questions that the passages below can answer. "
        "One question per line. No numbering, no answers.\n\n"
        f"{context}"
    )
    return parse_numbered_lines(_ask(chat_model, prompt), n)


def generate_followups(chat_model: Any, question: str, answer: str, n: int = 2) -> list[str]:
    """Return up to `n` related questions that the same documents can answer."""
    prompt = (
        f"The user asked: {question}\n"
        f"The answer was: {answer}\n\n"
        f"Write exactly {n} related questions that the same documents can answer. "
        "One question per line. No numbering, no answers."
    )
    return parse_numbered_lines(_ask(chat_model, prompt), n)
