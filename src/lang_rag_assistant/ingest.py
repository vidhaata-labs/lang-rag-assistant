"""Index PDFs: read, chunk, embed, store, and generate suggested questions."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from lang_rag_assistant.config.logging_config import get_logger
from lang_rag_assistant.database import ChunkRecord, Repository
from lang_rag_assistant.errors import DocumentError, ProviderError
from lang_rag_assistant.file_reader import read_pdf_pages
from lang_rag_assistant.retrieval import KnowledgeBase
from lang_rag_assistant.suggestions import generate_document_questions

logger = get_logger(__name__)

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
QUESTION_CONTEXT_PASSAGES = 8
NO_KEY_WARNING = "No model key is set, so suggested questions were skipped."


@dataclass(frozen=True)
class IngestResult:
    """Outcome of indexing one document."""

    document_id: int
    name: str
    passage_count: int
    already_indexed: bool
    questions: list[str]
    warning: str | None


def ingest_pdf(
    *,
    repo: Repository,
    kb: KnowledgeBase,
    chat_model: Any | None,
    provider: str,
    model_name: str,
    source: Path | bytes,
    name: str,
    source_kind: str,
) -> IngestResult:
    """Index one PDF unless identical bytes are already stored.

    Raises:
        DocumentError: the PDF cannot be read (nothing is stored).
    """
    pages = read_pdf_pages(source, name=name)
    content = source if isinstance(source, bytes) else source.read_bytes()
    digest = hashlib.sha256(content).hexdigest()

    existing = repo.find_document_by_sha256(digest)
    if existing is not None:
        logger.info("Skipping %s: identical content is already indexed", name)
        return IngestResult(existing.id, existing.name, existing.passage_count, True, existing.questions, None)

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    records = [
        ChunkRecord(page_number=page.page_number, text=piece)
        for page in pages
        for piece in splitter.split_text(page.text)
    ]
    texts = [record.text for record in records]
    logger.info("Indexing %s with %s:%s (%d passages)", name, provider, model_name, len(records))

    vectors = kb.embeddings.embed_documents(texts)
    document_id = repo.add_document(
        name=name,
        source_kind=source_kind,
        sha256=digest,
        chunks=records,
        vectors=vectors,
        embedding_model=kb.embedding_model,
    )
    kb.refresh()

    questions: list[str] = []
    warning: str | None = None
    if chat_model is None:
        warning = NO_KEY_WARNING
    else:
        try:
            questions = generate_document_questions(chat_model, texts[:QUESTION_CONTEXT_PASSAGES])
            repo.set_suggestions(document_id, questions)
        except ProviderError as exc:
            warning = exc.user_message

    return IngestResult(document_id, name, len(records), False, questions, warning)


def ingest_bundled(
    *,
    repo: Repository,
    kb: KnowledgeBase,
    chat_model: Any | None,
    provider: str,
    model_name: str,
    bundled_dir: Path,
) -> list[IngestResult]:
    """Index every PDF in `bundled_dir`; unreadable files are logged and skipped."""
    if not bundled_dir.is_dir():
        logger.warning("Bundled PDF folder not found: %s", bundled_dir)
        return []
    results: list[IngestResult] = []
    for path in sorted(bundled_dir.glob("*.pdf")):
        try:
            results.append(
                ingest_pdf(
                    repo=repo, kb=kb, chat_model=chat_model, provider=provider, model_name=model_name,
                    source=path, name=path.name, source_kind="bundled",
                )
            )
        except DocumentError as exc:
            logger.warning("Skipping bundled document %s: %s", path.name, exc)
    return results


def reembed_all(*, repo: Repository, embeddings: Any, embedding_model: str) -> int:
    """Recompute every passage vector with `embeddings`; return how many were updated."""
    rows = repo.chunks_for_reembedding()
    if not rows:
        return 0
    vectors = embeddings.embed_documents([text for _, text in rows])
    repo.replace_vectors(embedding_model, {chunk_id: vector for (chunk_id, _), vector in zip(rows, vectors)})
    logger.info("Re-embedded %d passages with %s", len(rows), embedding_model)
    return len(rows)
