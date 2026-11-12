"""Read text from PDF files, page by page, with validation that raises DocumentError."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from lang_rag_assistant.config.logging_config import get_logger
from lang_rag_assistant.errors import DocumentError

logger = get_logger(__name__)

MAX_UPLOAD_BYTES = 20 * 1024 * 1024


@dataclass(frozen=True)
class PageText:
    """Extracted text of one PDF page."""

    page_number: int  # 1-based
    text: str


def _check_size(size: int, name: str) -> None:
    if size > MAX_UPLOAD_BYTES:
        raise DocumentError(f"{name} must be 20 MB or smaller.")


def read_pdf_pages(source: Path | bytes, *, name: str) -> list[PageText]:
    """Return the non-empty pages of a text-based PDF.

    Raises:
        DocumentError: the file is missing, too large, not a PDF, password
            protected, or has no selectable text.
    """
    if isinstance(source, bytes):
        _check_size(len(source), name)
        stream: BytesIO | Path = BytesIO(source)
    else:
        if not source.is_file():
            raise DocumentError(f"File not found: {name}.")
        _check_size(source.stat().st_size, name)
        stream = source

    try:
        reader = PdfReader(stream)
    except (PdfReadError, ValueError, OSError) as exc:
        raise DocumentError(f"{name} is not a readable PDF.") from exc

    try:
        if reader.is_encrypted and not reader.decrypt(""):
            raise DocumentError(f"{name} is password-protected.")
        pages: list[PageText] = []
        for index, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append(PageText(page_number=index, text=text))
    except PdfReadError as exc:
        raise DocumentError(f"{name} is not a readable PDF.") from exc

    if not pages:
        raise DocumentError(f"{name} has no selectable text (it may be a scanned PDF).")
    logger.debug("Read %d text pages from %s", len(pages), name)
    return pages
