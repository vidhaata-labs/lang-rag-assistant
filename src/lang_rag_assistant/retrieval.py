"""Similarity search over passages whose vectors are stored in the repository."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from lang_rag_assistant.config.logging_config import get_logger
from lang_rag_assistant.database import Repository, StoredChunk

logger = get_logger(__name__)


@dataclass(frozen=True)
class RetrievedChunk:
    """A passage returned by a search, with its cosine similarity score."""

    chunk_id: int
    document_name: str
    page_number: int
    text: str
    score: float


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two vectors; 0.0 when either has zero length."""
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


class KnowledgeBase:
    """Holds the passages embedded with one embedding model and ranks them against a query."""

    def __init__(self, repo: Repository, embeddings: Any, embedding_model: str) -> None:
        self._repo = repo
        self.embeddings = embeddings
        self.embedding_model = embedding_model
        self._chunks: list[StoredChunk] = []
        self.refresh()

    def refresh(self) -> None:
        """Reload passages and vectors from the repository."""
        self._chunks = self._repo.load_chunks(self.embedding_model)
        logger.info("Knowledge base loaded %d passages", len(self._chunks))

    def search(self, query: str, k: int = 4) -> list[RetrievedChunk]:
        """Return up to `k` passages most similar to `query`, best first."""
        if not self._chunks:
            return []
        query_vector = self.embeddings.embed_query(query)
        scored = sorted(
            ((cosine_similarity(query_vector, chunk.vector), chunk) for chunk in self._chunks),
            key=lambda pair: pair[0],
            reverse=True,
        )
        return [
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                document_name=chunk.document_name,
                page_number=chunk.page_number,
                text=chunk.text,
                score=score,
            )
            for score, chunk in scored[:k]
        ]
