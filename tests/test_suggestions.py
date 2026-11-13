import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from lang_rag_assistant.errors import ProviderError
from lang_rag_assistant.suggestions import (
    generate_document_questions,
    generate_followups,
    parse_numbered_lines,
)


def test_parse_strips_numbering_and_limits_count():
    text = "1. What causes warming?\n2) How do glaciers melt?\n- Why do seas rise?\n\n4. Extra?"
    assert parse_numbered_lines(text, n=3) == [
        "What causes warming?", "How do glaciers melt?", "Why do seas rise?",
    ]


def test_parse_drops_blank_and_non_questions():
    assert parse_numbered_lines("Sure! Here you go.\n1. Is it hot?", n=4) == ["Is it hot?"]


def test_document_questions_come_from_model():
    model = FakeListChatModel(
        responses=["1. What is the main cause?\n2. How big is the effect?\n3. Who is affected?\n4. What helps?"]
    )
    assert len(generate_document_questions(model, ["chunk text"], n=4)) == 4


def test_model_failure_becomes_provider_error():
    class Boom(FakeListChatModel):
        def _call(self, *a, **k):
            raise RuntimeError("429 too many requests")

    with pytest.raises(ProviderError):
        generate_followups(Boom(responses=[]), "q", "a", n=2)


def test_document_questions_default_to_six():
    model = FakeListChatModel(responses=["\n".join(f"{i}. Question {i}?" for i in range(1, 9))])
    assert len(generate_document_questions(model, ["chunk text"])) == 6
