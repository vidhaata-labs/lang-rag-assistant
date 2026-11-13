"""Search tool: saves retrieved passages into the conversation workspace for the agent to read."""

from __future__ import annotations

import uuid
from pathlib import Path

from deepagents.backends import FilesystemBackend
from langchain_core.tools import BaseTool, tool

from lang_rag_assistant.config.logging_config import get_logger
from lang_rag_assistant.retrieval import KnowledgeBase, RetrievedChunk

logger = get_logger(__name__)

SEARCH_K = 4
NO_MATCH_MESSAGE = "No matching passages were found."


def make_search_tool(
    kb: KnowledgeBase,
    workspace_root: Path,
    retrieved: list[RetrievedChunk],
) -> BaseTool:
    """Return the `search_documentation` tool bound to a workspace and a per-turn log.

    Each passage is saved as `/retrieved/<batch>/chunk_<n>.md`, where `n` is its
    citation number. The header of each file is `# [n] <document> · p. <page>`.
    `retrieved` receives every saved passage, in citation order.
    """
    workspace_root.mkdir(parents=True, exist_ok=True)
    backend = FilesystemBackend(root_dir=workspace_root, virtual_mode=True)

    @tool("search_documentation", parse_docstring=True)
    def search_documentation(query: str) -> str:
        """Search the indexed documents and save matching passages for analysis.

        Args:
            query: Natural language search query.

        Returns:
            File paths where the retrieved passages were saved under /retrieved/.
        """
        hits = kb.search(query, k=SEARCH_K)
        if not hits:
            return NO_MATCH_MESSAGE

        batch = uuid.uuid4().hex[:8]
        first_number = len(retrieved) + 1
        uploads: list[tuple[str, bytes]] = []
        for offset, hit in enumerate(hits):
            number = first_number + offset
            header = f"# [{number}] {hit.document_name} · p. {hit.page_number}"
            uploads.append((f"/retrieved/{batch}/chunk_{number}.md", f"{header}\n\n{hit.text}".encode()))

        for response in backend.upload_files(uploads):
            if response.error:
                raise RuntimeError(f"Could not save retrieved passages: {response.error}")

        retrieved.extend(hits)
        logger.info("Saved %d passages for query (batch %s)", len(hits), batch)
        paths = [path for path, _ in uploads]
        return f"Saved {len(paths)} documentation chunks:\n" + "\n".join(paths)

    return search_documentation
