from pathlib import Path
from typing import List
from langchain_core.documents import Document
from pypdf import PdfReader
from lang_rag_assistant.config.logging_config import get_logger

logger = get_logger(__name__)


def read_pdf_text(pdf_path: str | Path) -> str:
    """Read and return the text contents of a PDF file"""
    path = Path(pdf_path)

    if not path.exists:
        logger.error("PDF not found: %s", pdf_path)
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    logger.debug("Reading PDF: %s", path)
    reader = PdfReader(str(pdf_path))
    text_parts = [page.extract_text() or "" for page in reader.pages]
    logger.debug("PDF %s has %d pages", path, len(reader.pages))

    return "\n".join(text_parts)


def read_pdfs(pdf_paths: List[str | Path]) -> List[Document]:
    """Read text from multiple PDFs and return a list of Documents"""
    docs: List[Document] = []

    for path in pdf_paths:
        path_obj = Path(path)
        text = read_pdf_text(path_obj)
        docs.append(Document(page_content=text, metadata={"source": str(path_obj)}))
        logger.info("Read PDF: %s (%d chars)", path_obj, len(text))

    logger.info("Read %d PDF document(s)", len(docs))
    return docs
