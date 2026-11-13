import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from lang_rag_assistant.database import ChunkRecord, Repository
from lang_rag_assistant.retrieval import KnowledgeBase, cosine_similarity


@pytest.fixture
def emb():
    return DeterministicFakeEmbedding(size=8)


@pytest.fixture
def repo(tmp_path):
    r = Repository(tmp_path / "k.sqlite")
    yield r
    r.close()


def test_cosine_of_identical_vectors_is_one():
    assert cosine_similarity([1.0, 2.0], [1.0, 2.0]) == pytest.approx(1.0)


def test_cosine_with_zero_vector_is_zero():
    assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0


def test_search_returns_best_match_first(repo, emb):
    texts = ["fossil fuels cause warming", "glaciers melt into the ocean"]
    vectors = emb.embed_documents(texts)
    repo.add_document(name="a.pdf", source_kind="bundled", sha256="1",
                      chunks=[ChunkRecord(page_number=1, text=t) for t in texts],
                      vectors=vectors, embedding_model="all-MiniLM-L6-v2")
    kb = KnowledgeBase(repo, emb, "all-MiniLM-L6-v2")
    results = kb.search(texts[1], k=2)
    assert results[0].text == texts[1]
    assert results[0].score == pytest.approx(1.0, abs=1e-5)
    assert results[0].document_name == "a.pdf" and results[0].page_number == 1


def test_search_on_empty_store_returns_nothing(repo, emb):
    assert KnowledgeBase(repo, emb, "all-MiniLM-L6-v2").search("anything") == []


def test_knowledge_base_exposes_embeddings_and_model_name(repo, emb):
    kb = KnowledgeBase(repo, emb, "all-MiniLM-L6-v2")
    assert kb.embeddings is emb
    assert kb.embedding_model == "all-MiniLM-L6-v2"
