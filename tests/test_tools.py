from langchain_core.embeddings import DeterministicFakeEmbedding

from lang_rag_assistant.database import ChunkRecord, Repository
from lang_rag_assistant.retrieval import KnowledgeBase
from lang_rag_assistant.tools import make_search_tool


def test_search_writes_numbered_chunk_files_and_logs_them(tmp_path):
    repo = Repository(tmp_path / "t.sqlite")
    emb = DeterministicFakeEmbedding(size=8)
    texts = ["sea level rises with warming", "fossil fuels emit carbon"]
    repo.add_document(name="g.pdf", source_kind="bundled", sha256="s",
                      chunks=[ChunkRecord(page_number=4, text=t) for t in texts],
                      vectors=emb.embed_documents(texts), embedding_model="all-MiniLM-L6-v2")
    kb = KnowledgeBase(repo, emb, "all-MiniLM-L6-v2")
    retrieved = []
    tool = make_search_tool(kb, tmp_path / "ws", retrieved)
    output = tool.invoke({"query": "sea level rises with warming"})
    assert "/retrieved/" in output
    assert len(retrieved) == 2
    path = next(line for line in output.splitlines() if line.startswith("/retrieved/"))
    text = (tmp_path / "ws" / path.lstrip("/")).read_text(encoding="utf-8")
    assert text.startswith("# [1] g.pdf · p. 4")
    repo.close()


def test_empty_store_returns_no_match_message(tmp_path):
    repo = Repository(tmp_path / "e.sqlite")
    kb = KnowledgeBase(repo, DeterministicFakeEmbedding(size=8), "all-MiniLM-L6-v2")
    tool = make_search_tool(kb, tmp_path / "ws", [])
    assert tool.invoke({"query": "anything"}) == "No matching passages were found."
    repo.close()
