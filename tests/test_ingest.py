from pathlib import Path

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from lang_rag_assistant.database import Repository
from lang_rag_assistant.errors import DocumentError
from lang_rag_assistant.ingest import ingest_bundled, ingest_pdf, reembed_all
from lang_rag_assistant.retrieval import KnowledgeBase

BUNDLED_DIR = Path(__file__).resolve().parents[1] / "pdfs"
MODEL = "all-MiniLM-L6-v2"


@pytest.fixture
def env(tmp_path):
    repo = Repository(tmp_path / "i.sqlite")
    emb = DeterministicFakeEmbedding(size=16)
    kb = KnowledgeBase(repo, emb, MODEL)
    yield repo, emb, kb
    repo.close()


def _chat():
    return FakeListChatModel(responses=["1. Q one?\n2. Q two?\n3. Q three?\n4. Q four?"])


def test_bundled_pdfs_are_indexed_with_pages(env):
    repo, emb, kb = env
    results = ingest_bundled(repo=repo, kb=kb, chat_model=_chat(), provider="google_genai",
                             model_name="gemini-3.1-flash-lite", bundled_dir=BUNDLED_DIR)
    assert {r.name for r in results} == {"climate_change.pdf", "global_warming.pdf"}
    assert all(r.passage_count > 0 for r in results)
    assert all(len(r.questions) == 4 for r in results)
    assert all(c.page_number >= 1 for c in repo.load_chunks(MODEL))


def test_same_bytes_are_not_indexed_twice(env):
    repo, emb, kb = env
    pdf = (BUNDLED_DIR / "climate_change.pdf").read_bytes()
    first = ingest_pdf(repo=repo, kb=kb, chat_model=None, provider="google_genai", model_name="m",
                       source=pdf, name="copy.pdf", source_kind="uploaded")
    second = ingest_pdf(repo=repo, kb=kb, chat_model=None, provider="google_genai", model_name="m",
                        source=pdf, name="copy.pdf", source_kind="uploaded")
    assert first.already_indexed is False and second.already_indexed is True
    assert len(repo.list_documents()) == 1


def test_no_chat_model_leaves_questions_empty_with_warning(env):
    repo, emb, kb = env
    result = ingest_pdf(repo=repo, kb=kb, chat_model=None, provider="groq", model_name="m",
                        source=(BUNDLED_DIR / "global_warming.pdf").read_bytes(),
                        name="g.pdf", source_kind="uploaded")
    assert result.questions == [] and result.warning is not None


def test_invalid_pdf_stores_nothing(env):
    repo, emb, kb = env
    with pytest.raises(DocumentError):
        ingest_pdf(repo=repo, kb=kb, chat_model=None, provider="groq", model_name="m",
                   source=b"not a pdf", name="x.pdf", source_kind="uploaded")
    assert repo.list_documents() == []


def test_reembed_all_switches_model(env):
    repo, emb, kb = env
    ingest_pdf(repo=repo, kb=kb, chat_model=None, provider="groq", model_name="m",
               source=(BUNDLED_DIR / "global_warming.pdf").read_bytes(), name="g.pdf", source_kind="uploaded")
    count = reembed_all(repo=repo, embeddings=DeterministicFakeEmbedding(size=4), embedding_model="BGE-M3")
    assert count > 0
    assert repo.load_chunks("BGE-M3") and not repo.load_chunks(MODEL)
