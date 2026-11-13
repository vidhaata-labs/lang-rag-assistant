import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage

from lang_rag_assistant.chat_service import ChatService, cited_numbers
from lang_rag_assistant.database import Repository
from lang_rag_assistant.errors import ProviderError, ProviderErrorKind
from lang_rag_assistant.retrieval import KnowledgeBase


class TimingOutModel(FakeListChatModel):
    """A chat model whose provider call always times out."""

    def bind_tools(self, tools, **kwargs):
        return self

    def _call(self, *args, **kwargs):
        raise RuntimeError("request timed out")

    def _stream(self, *args, **kwargs):
        # Streaming is how the app calls the model; a real timeout fails here too.
        raise RuntimeError("request timed out")
        yield  # pragma: no cover - makes this a generator, like the real streaming method


def test_cited_numbers_parses_bracket_references_in_order():
    assert cited_numbers("Fuel [1][2] and glaciers [2].") == [1, 2]


def test_cited_numbers_ignores_text_without_references():
    assert cited_numbers("No citations here.") == []


@pytest.fixture
def service(tmp_path):
    repo = Repository(tmp_path / "s.sqlite")
    emb = DeterministicFakeEmbedding(size=8)
    kb = KnowledgeBase(repo, emb, "all-MiniLM-L6-v2")
    settings = type("S", (), {"data_dir": tmp_path, "max_concurrent_analysts": 3})()
    yield ChatService(repo, kb, settings), repo
    repo.close()


def test_run_turn_persists_user_message_even_when_model_fails(service):
    svc, repo = service
    conv = repo.create_conversation("t")
    option = type("O", (), {"provider": "groq", "model": "llama-3.1-8b-instant"})()
    with pytest.raises(ProviderError) as info:
        svc.run_turn(conversation_id=conv, question="Why?", option=option,
                     chat_model=TimingOutModel(responses=["unused"]), on_step=lambda e: None)
    assert info.value.kind is ProviderErrorKind.UNAVAILABLE
    assert [m.role for m in repo.list_messages(conv)] == ["user"]


class StreamingModel(FakeListChatModel):
    """A chat model that streams its reply token by token and accepts tools."""

    def bind_tools(self, tools, **kwargs):
        return self


class _Stop(BaseException):
    """Stands in for Streamlit's Stop signal, which is not an Exception."""


def _option():
    return type("O", (), {"provider": "openai", "model": "gpt-4o-mini"})()


def test_streamed_tokens_reach_the_caller_and_the_answer_is_saved_whole(service):
    svc, repo = service
    conv = repo.create_conversation("t")
    pieces: list[str] = []
    svc.run_turn(conversation_id=conv, question="Why?", option=_option(),
                          chat_model=StreamingModel(responses=["Glaciers melt [1]."]),
                          on_step=lambda e: None, on_text=pieces.append)
    assert len(pieces) > 1
    assert "".join(pieces) == "Glaciers melt [1]."
    assert repo.list_messages(conv)[-1].content == "Glaciers melt [1]."
    assert repo.list_messages(conv)[-1].stopped is False


def test_stop_mid_answer_saves_the_partial_answer_marked_stopped(service):
    svc, repo = service
    conv = repo.create_conversation("t")
    seen: list[str] = []

    def on_text(piece: str) -> None:
        seen.append(piece)
        if len(seen) == 3:
            raise _Stop()

    with pytest.raises(_Stop):
        svc.run_turn(conversation_id=conv, question="Why?", option=_option(),
                     chat_model=StreamingModel(responses=["Glaciers melt quickly and seas rise."]),
                     on_step=lambda e: None, on_text=on_text)
    last = repo.list_messages(conv)[-1]
    assert last.role == "assistant"
    assert last.stopped is True
    assert last.content == "".join(seen)


def test_tokens_from_a_nested_agent_are_not_part_of_the_answer():
    from langchain_core.messages import AIMessageChunk

    from lang_rag_assistant.chat_service import _token_text

    top = (AIMessageChunk(content="Answer"), {"langgraph_checkpoint_ns": "model:1"})
    nested = (AIMessageChunk(content="Subagent notes"), {"langgraph_checkpoint_ns": "tools:2|model:3"})
    assert _token_text(top) == "Answer"
    assert _token_text(nested) == ""


class _ScriptedRun:
    """Stands in for the agent: runs a script that may report model failures through the callbacks."""

    def __init__(self, script):
        self._script = script

    def stream(self, inputs, stream_mode, config=None):
        yield from self._script(config)


def _tool_error(text):
    from langchain_core.messages import ToolMessage

    return ("updates", {"tools": {"messages": [ToolMessage(content=text, tool_call_id="1", status="error")]}})


def test_model_failure_inside_a_subagent_is_raised_and_its_log_never_reaches_the_answer(service, monkeypatch):
    from uuid import uuid4

    from langchain_core.exceptions import ModelRateLimitError

    from lang_rag_assistant import chat_service as cs

    svc, repo = service
    conv = repo.create_conversation("t")
    log = "429 RESOURCE_EXHAUSTED: quota exceeded. Traceback (most recent call last): ... very long log"

    def script(config):
        # The subagent's model call fails; the failure is reported through the callback.
        config["callbacks"][0].on_llm_error(ModelRateLimitError(log), run_id=uuid4(), parent_run_id=None)
        yield _tool_error(f"Error: {log}")  # the tool node turns the failure into text
        yield ("updates", {"model": {"messages": [AIMessage(content=f"The search failed: {log}")]}})

    monkeypatch.setattr(cs, "build_agent", lambda *a, **k: _ScriptedRun(script))
    with pytest.raises(ProviderError) as info:
        svc.run_turn(conversation_id=conv, question="Why?", option=_option(),
                     chat_model=StreamingModel(responses=["unused"]), on_step=lambda e: None)
    assert info.value.kind is ProviderErrorKind.RATE_LIMIT
    assert [m.role for m in repo.list_messages(conv)] == ["user"]


def test_tool_failure_gives_a_plain_message_not_the_tool_log(service, monkeypatch):
    from lang_rag_assistant import chat_service as cs
    from lang_rag_assistant.errors import AppError

    svc, repo = service
    conv = repo.create_conversation("t")

    def script(config):
        yield _tool_error("Error: Traceback (most recent call last): FileNotFoundError: /retrieved/x.md")

    monkeypatch.setattr(cs, "build_agent", lambda *a, **k: _ScriptedRun(script))
    with pytest.raises(AppError) as info:
        svc.run_turn(conversation_id=conv, question="Why?", option=_option(),
                     chat_model=StreamingModel(responses=["unused"]), on_step=lambda e: None)
    assert not isinstance(info.value, ProviderError)
    assert info.value.user_message == "Something went wrong while searching the documents. Try again."
    assert "Traceback" not in info.value.user_message
    assert [m.role for m in repo.list_messages(conv)] == ["user"]
