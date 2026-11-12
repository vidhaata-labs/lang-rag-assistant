import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel

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
