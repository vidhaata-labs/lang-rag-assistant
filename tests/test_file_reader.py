from io import BytesIO
from pathlib import Path

import pytest
from pypdf import PdfWriter

from lang_rag_assistant.errors import DocumentError
from lang_rag_assistant.file_reader import MAX_UPLOAD_BYTES, read_pdf_pages

BUNDLED = Path(__file__).resolve().parents[1] / "pdfs" / "climate_change.pdf"


def _blank_pdf_bytes() -> bytes:
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buf = BytesIO()
    writer.write(buf)
    return buf.getvalue()


def test_bundled_pdf_yields_numbered_pages():
    pages = read_pdf_pages(BUNDLED, name="climate_change.pdf")
    assert pages and pages[0].page_number == 1
    assert any(p.text.strip() for p in pages)


def test_not_a_pdf_raises_document_error():
    with pytest.raises(DocumentError, match="not a readable PDF"):
        read_pdf_pages(b"hello world", name="notes.pdf")


def test_blank_pdf_raises_no_text_error():
    with pytest.raises(DocumentError, match="no selectable text"):
        read_pdf_pages(_blank_pdf_bytes(), name="scan.pdf")


def test_oversized_upload_raises():
    with pytest.raises(DocumentError, match="200 MB"):
        read_pdf_pages(b"%PDF" + b"0" * (MAX_UPLOAD_BYTES + 1), name="big.pdf")


def test_missing_path_raises_document_error(tmp_path):
    with pytest.raises(DocumentError, match="not found"):
        read_pdf_pages(tmp_path / "missing.pdf", name="missing.pdf")


def test_pdf_library_errors_become_document_errors(monkeypatch):
    from pypdf.errors import PyPdfError

    class _Failing:
        def __init__(self, *args, **kwargs):
            raise PyPdfError("decompression limit reached")

    monkeypatch.setattr("lang_rag_assistant.file_reader.PdfReader", _Failing)
    with pytest.raises(DocumentError):
        read_pdf_pages(b"%PDF-1.4 stub", name="huge.pdf")
