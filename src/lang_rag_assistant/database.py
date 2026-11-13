"""SQLite storage for documents, passages, suggested questions, conversations and settings."""

from __future__ import annotations

import json
import sqlite3
import threading
from array import array
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lang_rag_assistant.config.logging_config import get_logger
from lang_rag_assistant.errors import AppError

logger = get_logger(__name__)

SNIPPET_LENGTH = 160

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    source_kind TEXT NOT NULL CHECK (source_kind IN ('bundled', 'uploaded')),
    sha256 TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    page_number INTEGER NOT NULL,
    text TEXT NOT NULL,
    embedding BLOB,
    embedding_model TEXT
);
CREATE TABLE IF NOT EXISTS suggestions (
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    question TEXT NOT NULL,
    PRIMARY KEY (document_id, position)
);
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    sources TEXT NOT NULL DEFAULT '[]',
    followups TEXT NOT NULL DEFAULT '[]',
    stopped INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


@dataclass(frozen=True)
class ChunkRecord:
    """A passage to store, before it has an id or vector."""

    page_number: int
    text: str


@dataclass(frozen=True)
class StoredChunk:
    """A stored passage with its vector, for retrieval."""

    chunk_id: int
    document_id: int
    document_name: str
    source_kind: str
    page_number: int
    text: str
    vector: list[float]


@dataclass(frozen=True)
class DocumentRecord:
    """A stored document with its passage count and suggested questions."""

    id: int
    name: str
    source_kind: str
    sha256: str
    passage_count: int
    questions: list[str]


@dataclass(frozen=True)
class SourceRecord:
    """A passage cited by an assistant message. `number` is its [n] in that turn."""

    chunk_id: int
    document_name: str
    page_number: int
    number: int


@dataclass(frozen=True)
class MessageRecord:
    """A stored chat message."""

    id: int
    role: str
    content: str
    sources: list[SourceRecord] = field(default_factory=list)
    followups: list[str] = field(default_factory=list)
    stopped: bool = False


@dataclass(frozen=True)
class ConversationSummary:
    """A conversation row for the sidebar and the history page."""

    id: int
    title: str
    updated_at: str
    message_count: int
    documents: list[str]
    snippet: str


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _encode_vector(vector: Sequence[float]) -> bytes:
    return array("f", vector).tobytes()


def _decode_vector(blob: bytes) -> list[float]:
    values = array("f")
    values.frombytes(blob)
    return list(values)


def _like_pattern(query: str) -> str:
    escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def _snippet(text: str) -> str:
    return text if len(text) <= SNIPPET_LENGTH else text[: SNIPPET_LENGTH - 1] + "…"


class Repository:
    """Thread-safe SQLite repository. Each public method runs in one transaction."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        try:
            self._conn = sqlite3.connect(path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._conn.execute("PRAGMA journal_mode = WAL")
            self._conn.executescript(_SCHEMA)
            self._migrate()
        except sqlite3.Error as exc:
            raise AppError(f"Could not open the database at {path}: {exc}") from exc

    def _migrate(self) -> None:
        """Add columns that older databases lack. New databases already have them."""
        columns = {row["name"] for row in self._conn.execute("PRAGMA table_info(messages)")}
        if "stopped" not in columns:
            self._conn.execute("ALTER TABLE messages ADD COLUMN stopped INTEGER NOT NULL DEFAULT 0")
            self._conn.commit()

    def close(self) -> None:
        """Close the underlying connection."""
        with self._lock:
            self._conn.close()

    @contextmanager
    def _transaction(self) -> Generator[sqlite3.Connection]:
        with self._lock:
            try:
                yield self._conn
                self._conn.commit()
            except sqlite3.Error as exc:
                self._conn.rollback()
                logger.exception("Database operation failed")
                raise AppError(f"Database error: {exc}") from exc
            except BaseException:
                self._conn.rollback()
                raise

    # ----- documents -------------------------------------------------------

    def add_document(
        self,
        *,
        name: str,
        source_kind: str,
        sha256: str,
        chunks: list[ChunkRecord],
        vectors: list[list[float]],
        embedding_model: str,
    ) -> int:
        """Store a document with its passages and vectors; return the document id."""
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        with self._transaction() as conn:
            cursor = conn.execute(
                "INSERT INTO documents (name, source_kind, sha256, created_at) VALUES (?, ?, ?, ?)",
                (name, source_kind, sha256, _now()),
            )
            document_id = int(cursor.lastrowid or 0)
            conn.executemany(
                "INSERT INTO chunks (document_id, position, page_number, text, embedding, embedding_model)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (document_id, index, chunk.page_number, chunk.text,
                     _encode_vector(vector), embedding_model)
                    for index, (chunk, vector) in enumerate(zip(chunks, vectors))
                ],
            )
        return document_id

    def _document_from_row(self, conn: sqlite3.Connection, row: sqlite3.Row) -> DocumentRecord:
        count = conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE document_id = ?", (row["id"],)
        ).fetchone()[0]
        questions = [
            r["question"]
            for r in conn.execute(
                "SELECT question FROM suggestions WHERE document_id = ? ORDER BY position", (row["id"],)
            )
        ]
        return DocumentRecord(
            id=row["id"], name=row["name"], source_kind=row["source_kind"], sha256=row["sha256"],
            passage_count=count, questions=questions,
        )

    def find_document_by_sha256(self, sha256: str) -> DocumentRecord | None:
        """Return the document with this content hash, if stored."""
        with self._transaction() as conn:
            row = conn.execute("SELECT * FROM documents WHERE sha256 = ?", (sha256,)).fetchone()
            return self._document_from_row(conn, row) if row else None

    def list_documents(self) -> list[DocumentRecord]:
        """Return all documents, oldest first."""
        with self._transaction() as conn:
            rows = conn.execute("SELECT * FROM documents ORDER BY created_at, id").fetchall()
            return [self._document_from_row(conn, row) for row in rows]

    def delete_document(self, document_id: int) -> None:
        """Delete a document with its passages and suggestions."""
        with self._transaction() as conn:
            conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))

    def set_suggestions(self, document_id: int, questions: list[str]) -> None:
        """Replace the suggested questions for a document, keeping their order."""
        with self._transaction() as conn:
            conn.execute("DELETE FROM suggestions WHERE document_id = ?", (document_id,))
            conn.executemany(
                "INSERT INTO suggestions (document_id, position, question) VALUES (?, ?, ?)",
                [(document_id, index, question) for index, question in enumerate(questions)],
            )

    def chunks_for_document(self, document_id: int) -> list[str]:
        """Return the passage texts of a document in their stored order."""
        with self._transaction() as conn:
            rows = conn.execute(
                "SELECT text FROM chunks WHERE document_id = ? ORDER BY position", (document_id,)
            ).fetchall()
            return [row["text"] for row in rows]

    def load_chunks(self, embedding_model: str) -> list[StoredChunk]:
        """Return passages embedded with `embedding_model`, with their vectors."""
        with self._transaction() as conn:
            rows = conn.execute(
                "SELECT c.id, c.document_id, d.name, d.source_kind, c.page_number, c.text, c.embedding"
                " FROM chunks c JOIN documents d ON d.id = c.document_id"
                " WHERE c.embedding_model = ? ORDER BY c.id",
                (embedding_model,),
            ).fetchall()
            return [
                StoredChunk(
                    chunk_id=row["id"], document_id=row["document_id"], document_name=row["name"],
                    source_kind=row["source_kind"], page_number=row["page_number"], text=row["text"],
                    vector=_decode_vector(row["embedding"]),
                )
                for row in rows
            ]

    def chunks_for_reembedding(self) -> list[tuple[int, str]]:
        """Return (chunk_id, text) for every passage, for re-embedding."""
        with self._transaction() as conn:
            rows = conn.execute("SELECT id, text FROM chunks ORDER BY id").fetchall()
            return [(row["id"], row["text"]) for row in rows]

    def replace_vectors(self, embedding_model: str, vectors: dict[int, list[float]]) -> None:
        """Store new vectors for the given passages and mark them with `embedding_model`."""
        with self._transaction() as conn:
            conn.executemany(
                "UPDATE chunks SET embedding = ?, embedding_model = ? WHERE id = ?",
                [(_encode_vector(vector), embedding_model, chunk_id) for chunk_id, vector in vectors.items()],
            )

    # ----- settings --------------------------------------------------------

    def get_setting(self, key: str) -> str | None:
        """Return a stored setting value, or None."""
        with self._transaction() as conn:
            row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
            return row["value"] if row else None

    def set_setting(self, key: str, value: str) -> None:
        """Insert or replace a setting."""
        with self._transaction() as conn:
            conn.execute(
                "INSERT INTO settings (key, value) VALUES (?, ?)"
                " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    # ----- conversations ---------------------------------------------------

    def create_conversation(self, title: str) -> int:
        """Create an empty conversation and return its id."""
        now = _now()
        with self._transaction() as conn:
            cursor = conn.execute(
                "INSERT INTO conversations (title, created_at, updated_at) VALUES (?, ?, ?)",
                (title, now, now),
            )
            return int(cursor.lastrowid or 0)

    def rename_conversation(self, conversation_id: int, title: str) -> None:
        """Change a conversation's title."""
        with self._transaction() as conn:
            conn.execute("UPDATE conversations SET title = ? WHERE id = ?", (title, conversation_id))

    def _summary(self, conn: sqlite3.Connection, conversation_id: int, snippet: str | None) -> ConversationSummary:
        row = conn.execute("SELECT * FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
        count = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE conversation_id = ?", (conversation_id,)
        ).fetchone()[0]
        documents: list[str] = []
        for message in conn.execute(
            "SELECT sources FROM messages WHERE conversation_id = ? AND role = 'assistant'", (conversation_id,)
        ):
            for source in json.loads(message["sources"]):
                if source["document_name"] not in documents:
                    documents.append(source["document_name"])
        if snippet is None:
            last = conn.execute(
                "SELECT content FROM messages WHERE conversation_id = ? ORDER BY id DESC LIMIT 1",
                (conversation_id,),
            ).fetchone()
            snippet = _snippet(last["content"]) if last else ""
        return ConversationSummary(
            id=row["id"], title=row["title"], updated_at=row["updated_at"],
            message_count=count, documents=documents, snippet=snippet,
        )

    def list_conversations(self, limit: int | None = None) -> list[ConversationSummary]:
        """Return conversations, most recently updated first."""
        with self._transaction() as conn:
            sql = "SELECT id FROM conversations ORDER BY updated_at DESC, id DESC"
            params: tuple[Any, ...] = ()
            if limit is not None:
                sql += " LIMIT ?"
                params = (limit,)
            ids = [row["id"] for row in conn.execute(sql, params).fetchall()]
            return [self._summary(conn, conversation_id, None) for conversation_id in ids]

    def search_conversations(self, query: str) -> list[ConversationSummary]:
        """Return conversations with a message containing `query`, most recent first."""
        with self._transaction() as conn:
            rows = conn.execute(
                "SELECT m.conversation_id, m.content FROM messages m"
                " JOIN conversations c ON c.id = m.conversation_id"
                " WHERE m.content LIKE ? ESCAPE '\\'"
                " ORDER BY c.updated_at DESC, m.id DESC",
                (_like_pattern(query),),
            ).fetchall()
            seen: dict[int, str] = {}
            for row in rows:
                seen.setdefault(row["conversation_id"], row["content"])
            return [self._summary(conn, cid, _snippet(text)) for cid, text in seen.items()]

    def add_message(
        self,
        conversation_id: int,
        role: str,
        content: str,
        sources: list[SourceRecord] | None = None,
        followups: list[str] | None = None,
        stopped: bool = False,
    ) -> int:
        """Append a message to a conversation and return its id.

        `stopped` marks an answer the user stopped before it finished.
        """
        now = _now()
        with self._transaction() as conn:
            cursor = conn.execute(
                "INSERT INTO messages (conversation_id, role, content, sources, followups, stopped, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    conversation_id, role, content,
                    json.dumps([asdict(s) for s in sources or []]),
                    json.dumps(followups or []),
                    int(stopped),
                    now,
                ),
            )
            conn.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conversation_id))
            return int(cursor.lastrowid or 0)

    def list_messages(self, conversation_id: int) -> list[MessageRecord]:
        """Return a conversation's messages in order."""
        with self._transaction() as conn:
            rows = conn.execute(
                "SELECT * FROM messages WHERE conversation_id = ? ORDER BY id", (conversation_id,)
            ).fetchall()
            return [
                MessageRecord(
                    id=row["id"], role=row["role"], content=row["content"],
                    sources=[SourceRecord(**s) for s in json.loads(row["sources"])],
                    followups=json.loads(row["followups"]),
                    stopped=bool(row["stopped"]),
                )
                for row in rows
            ]

    def page_count(self, document_id: int) -> int:
        """The highest page that has passages in a document; 0 when it has none."""
        with self._transaction() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(page_number), 0) AS pages FROM chunks WHERE document_id = ?",
                (document_id,),
            ).fetchone()
            return int(row["pages"])
