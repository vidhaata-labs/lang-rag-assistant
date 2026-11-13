from pathlib import Path

import pytest

from lang_rag_assistant.database import Repository

_LETTERS = "abcde"


def _vector(text: str) -> list[float]:
    """Deterministic 5-dimensional vector: how often each of a-e appears in `text`."""
    lowered = text.lower()
    return [float(lowered.count(letter)) for letter in _LETTERS]


class FakeEmbeddings:
    """Embeddings double: same text always gives the same vector, no model download."""

    def embed_query(self, text: str) -> list[float]:
        return _vector(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [_vector(text) for text in texts]


class BrokenEmbeddings:
    """Embeddings double whose every call fails, as a missing model would."""

    def embed_query(self, text: str) -> list[float]:
        raise RuntimeError("embedding model unavailable")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("embedding model unavailable")


class _RateLimitedChatModel:
    model_name = "gpt-4o-mini"

    def invoke(self, prompt: str):
        exc = Exception("rate limited")
        exc.status_code = 429
        raise exc


def make_pdf(*page_texts: str) -> bytes:
    """Return a minimal valid PDF with one page per text, using the Helvetica font."""
    objects: list[bytes] = []
    page_ids = [4 + 2 * i for i in range(len(page_texts))]
    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode())
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    for index, text in enumerate(page_texts):
        content = f"BT /F1 18 Tf 20 100 Td ({text}) Tj ET".encode()
        content_id = 5 + 2 * index
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 400 200] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>".encode()
        )
        objects.append(f"<< /Length {len(content)} >>\nstream\n".encode() + content + b"\nendstream")
    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_position = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += "".join(f"{offset:010d} 00000 n \n" for offset in offsets).encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_position}\n%%EOF\n"
    ).encode()
    return bytes(out)


@pytest.fixture
def tmp_db_path(tmp_path: Path) -> Path:
    """Path to a fresh SQLite file inside the test's temporary directory."""
    return tmp_path / "test.sqlite"


@pytest.fixture
def repo(tmp_db_path: Path) -> Repository:
    return Repository(tmp_db_path)


@pytest.fixture
def fake_embeddings() -> FakeEmbeddings:
    return FakeEmbeddings()


@pytest.fixture
def broken_embeddings() -> BrokenEmbeddings:
    return BrokenEmbeddings()


@pytest.fixture
def failing_chat_model() -> _RateLimitedChatModel:
    return _RateLimitedChatModel()


@pytest.fixture
def sample_pdf_bytes() -> bytes:
    return make_pdf("Oceans absorb most of the extra heat.", "Sea levels rise as water expands.")


@pytest.fixture
def bundled_pdf_dir(tmp_path: Path) -> Path:
    folder = tmp_path / "pdfs"
    folder.mkdir()
    (folder / "climate_change.pdf").write_bytes(make_pdf("Burning fossil fuels raises CO2."))
    (folder / "global_warming.pdf").write_bytes(make_pdf("Warming oceans expand and rise."))
    return folder
