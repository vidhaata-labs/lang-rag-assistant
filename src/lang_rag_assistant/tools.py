import uuid
from deepagents.backends import StateBackend
from langchain.tools import tool
from lang_rag_assistant.indexing import vector_store
from lang_rag_assistant.config.logging_config import get_logger

logger = get_logger(__name__)

backend = StateBackend()


@tool(parse_docstring=True)
def search_documentation(query: str) -> str:
    """Search the indexed documentation corpus and save matching chunks to the agent filesystem

    Args:
        query: Natural language search query.

    Returns:
        File paths where retrieved chunks were saved under /retrieved/
    """
    logger.info("Searching documentation for query: %s", query)
    retrieved_docs = vector_store.similarity_search(query, k=4)
    logger.info("Retrieved %d document chunks", len(retrieved_docs))

    batch_id = uuid.uuid4().hex[:8]
    uploads: list[tuple[str, bytes]] = []
    saved_paths: list[str] = []

    for index, doc in enumerate(retrieved_docs, start=1):
        path = f"/retrieved/{batch_id}/chunk_{index}.md"
        content = (
            f"# Source: {doc.metadata.get('source', 'unknown')}\n\n"
            f"{doc.page_content}"
        )
        uploads.append((path, content.encode("utf-8")))
        saved_paths.append(path)
        logger.debug("Prepared chunk %d: %s", index, path)

    logger.info("Uploading %d chunks to backend (batch_id=%s)", len(uploads), batch_id)
    backend.upload_files(uploads)

    result = (
        f"Saved {len(saved_paths)} documentation chunks:\n"
        + "\n".join(saved_paths)
    )
    logger.info("Search complete. Saved %d chunks", len(saved_paths))
    return result
