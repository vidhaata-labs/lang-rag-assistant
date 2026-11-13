import pytest

from lang_rag_assistant.database import ChunkRecord, Repository, SourceRecord


@pytest.fixture
def repo(tmp_path):
    r = Repository(tmp_path / "t.sqlite")
    yield r
    r.close()


def _add_doc(repo, sha="abc", kind="uploaded"):
    return repo.add_document(
        name="a.pdf", source_kind=kind, sha256=sha,
        chunks=[ChunkRecord(page_number=2, text="hello")],
        vectors=[[0.1, 0.2, 0.3]], embedding_model="all-MiniLM-L6-v2",
    )


def test_add_document_round_trips_chunks_and_vectors(repo):
    doc_id = _add_doc(repo)
    chunks = repo.load_chunks("all-MiniLM-L6-v2")
    assert len(chunks) == 1
    assert chunks[0].document_id == doc_id
    assert chunks[0].page_number == 2
    assert chunks[0].vector == pytest.approx([0.1, 0.2, 0.3], rel=1e-6)


def test_find_by_sha256_detects_duplicates(repo):
    _add_doc(repo, sha="same")
    assert repo.find_document_by_sha256("same") is not None
    assert repo.find_document_by_sha256("other") is None


def test_suggestions_are_stored_in_order(repo):
    doc_id = _add_doc(repo)
    repo.set_suggestions(doc_id, ["q1", "q2"])
    assert repo.list_documents()[0].questions == ["q1", "q2"]


def test_delete_document_removes_chunks(repo):
    doc_id = _add_doc(repo)
    repo.delete_document(doc_id)
    assert repo.load_chunks("all-MiniLM-L6-v2") == []


def test_messages_keep_sources_and_followups(repo):
    conv = repo.create_conversation("Causes")
    repo.add_message(conv, "user", "Why?")
    repo.add_message(conv, "assistant", "Because [1].",
                     sources=[SourceRecord(chunk_id=1, document_name="a.pdf", page_number=2, number=1)],
                     followups=["What next?"])
    messages = repo.list_messages(conv)
    assert [m.role for m in messages] == ["user", "assistant"]
    assert messages[1].sources[0].number == 1
    assert messages[1].followups == ["What next?"]


def test_search_matches_message_text(repo):
    conv = repo.create_conversation("Sea levels")
    repo.add_message(conv, "user", "How do glaciers affect the ocean?")
    assert [c.id for c in repo.search_conversations("glacier")] == [conv]
    assert repo.search_conversations("volcano") == []


def test_vectors_replaced_on_reembedding(repo):
    _add_doc(repo)
    chunk_id = repo.load_chunks("all-MiniLM-L6-v2")[0].chunk_id
    repo.replace_vectors("BGE-M3", {chunk_id: [1.0, 0.0]})
    assert repo.load_chunks("all-MiniLM-L6-v2") == []
    assert repo.load_chunks("BGE-M3")[0].vector == pytest.approx([1.0, 0.0])


def test_chunks_for_document_returns_texts_in_order(repo):
    repo.add_document(
        name="b.pdf", source_kind="bundled", sha256="order",
        chunks=[ChunkRecord(page_number=1, text="first"), ChunkRecord(page_number=2, text="second")],
        vectors=[[1.0], [2.0]], embedding_model="all-MiniLM-L6-v2",
    )
    doc_id = repo.find_document_by_sha256("order").id
    assert repo.chunks_for_document(doc_id) == ["first", "second"]


def test_stopped_flag_round_trips(tmp_path):
    repo = Repository(tmp_path / "f.sqlite")
    conv = repo.create_conversation("t")
    repo.add_message(conv, "assistant", "partial answer", stopped=True)
    assert repo.list_messages(conv)[-1].stopped is True


def test_database_from_before_the_stopped_flag_is_migrated(tmp_path):
    import sqlite3

    path = tmp_path / "old.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(
        "CREATE TABLE conversations (id INTEGER PRIMARY KEY, title TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);"
        "CREATE TABLE messages (id INTEGER PRIMARY KEY, conversation_id INTEGER NOT NULL, role TEXT NOT NULL,"
        " content TEXT NOT NULL, sources TEXT NOT NULL DEFAULT '[]', followups TEXT NOT NULL DEFAULT '[]', created_at TEXT NOT NULL);"
    )
    conn.commit()
    conn.close()
    repo = Repository(path)
    conv = repo.create_conversation("t")
    repo.add_message(conv, "assistant", "x", stopped=True)
    assert repo.list_messages(conv)[-1].stopped is True


def test_page_count_is_the_last_page_with_passages(tmp_path):
    from lang_rag_assistant.database import ChunkRecord

    repo = Repository(tmp_path / "p.sqlite")
    doc = repo.add_document(name="a.pdf", source_kind="bundled", sha256="p1",
                            chunks=[ChunkRecord(page_number=1, text="a"), ChunkRecord(page_number=4, text="b")],
                            vectors=[[1.0], [1.0]], embedding_model="m")
    assert repo.page_count(doc) == 4
