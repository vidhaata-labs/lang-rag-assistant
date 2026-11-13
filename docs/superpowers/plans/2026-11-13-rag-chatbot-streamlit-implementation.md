# RAG Chatbot Streamlit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (native, inline) or superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the four-page Streamlit RAG chatbot (Chat, History, Documents, Model) that matches the approved Doop wireframes, with SQLite persistence, provider gating, and error handling, on top of a refactored `deepagents` RAG core.

**Architecture:** A service layer owns settings, errors, SQLite storage, embeddings, retrieval (cosine search over stored vectors), ingestion, suggestions, and the persisted chat turn. `agent.py` exposes a builder and a pure event mapper with no import-time side effects. `main.py` is only the Streamlit entry point; each page is a module with a `render(services)` function.

**Tech Stack:** Python 3.13, Streamlit 1.67, LangChain 1.x (`init_chat_model`, `langchain-huggingface`), `deepagents`, `pypdf`, `numpy`, SQLite (stdlib), pytest, ruff, ty.

**Spec:** `wireframes/UX.md` (flows and states, decisions in section 12) and the approved Doop canvas https://doop.design/c/3XoSXpnr0Q (screens). Executors read both.

**Starting point:** The current tree is a broken stub (`agent.py` runs at import, imports missing modules, `main.py` is a placeholder). The previous working implementation is in git history at `89af4a3^` (`src/lang_rag_assistant/...`, `tests/...`). Where a task says "restore", take that file with `git show 89af4a3^:<path>`, then apply the changes the task lists. Do not copy wholesale.

## Global Constraints

- Python `>=3.13`; Streamlit `>=1.67` (installed: 1.67.0).
- Use only native `st.*` components. No `use_container_width`; use `width="stretch"` or `width="content"`.
- Font: `font` and `headingFont` are Inter; `codeFont` is JetBrains Mono. All three use the Google Fonts URL form in `.streamlit/config.toml`, as in UX.md section 12.
- Chat composer: `st.chat_input(..., submit_mode="stop")` while an answer is running.
- Model picker lists only models whose provider has a configured key (`GOOGLE_API_KEY`, `GROQ_API_KEY`, `OPENAI_API_KEY`). One model per provider, so there is no separate provider picker.
- Upload limit: 200 MiB per PDF (`200 * 1024 * 1024` bytes), matching the wireframes.
- Suggested questions: up to 6 per document; up to 2 follow-ups per answer.
- Embedding default: `all-MiniLM-L6-v2` (local); options `BGE-M3`, `nomic-embed-text`, `all-MiniLM-L6-v2`.
- Chunking: 1000 characters, 200 overlap. Retrieval: top 4 passages per search.
- No git commits unless the user asks. Changes stay in the working tree.
- Run commands with `uv run` from the project root.

## Review Focus

1. Invalid, encrypted, or text-less PDF uploaded → an error alert, and nothing is saved. Tested in Tasks 5, 8, 15.
2. The same PDF uploaded twice → no duplicate document row. Tested in Task 8.
3. Saved model whose key was removed, or no keys at all → disabled picker with a warning, no crash. Tested in Tasks 4, 13.
4. Provider rate limit or outage during a turn → error alert, question kept, no half-written assistant message. Tested in Tasks 2, 11.
5. Stop during a streamed answer → partial answer saved and marked stopped. Tested in Tasks 6, 11.

---

### Task 1: Project baseline

**Files:**
- Modify: `pyproject.toml` (add dev group `pytest`, `ruff`, `ty`; add `[tool.pytest.ini_options] pythonpath = ["src"]`)
- Delete: `src/lang_rag_assistant/agent.py`, `src/lang_rag_assistant/tools.py`, `src/lang_rag_assistant/main.py` (rewritten in later tasks)
- Create: `tests/conftest.py`, `tests/test_package.py`

**Interfaces:**
- Consumes: nothing.
- Produces: an importable `lang_rag_assistant` package; a `tmp_db_path` pytest fixture returning a path to a temporary SQLite file.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_package.py
def test_package_imports():
    import lang_rag_assistant  # noqa: F401
```

- [ ] **Step 2: Run it**

Run: `uv run pytest tests/test_package.py -v`
Expected: PASS once the package imports; FAIL today only if the stub has a broken import.

- [ ] **Step 3: Set up tooling**

Add the dev group and pytest config to `pyproject.toml`. Run `uv sync`.

- [ ] **Step 4: Verify**

Run: `uv run pytest -q` and `uv run ruff check src tests`
Expected: both exit 0.

---

### Task 2: Errors and provider classification

**Files:**
- Create: `src/lang_rag_assistant/errors.py` (restore from `89af4a3^`)
- Test: `tests/test_errors.py`

**Interfaces:**
- Produces:
  - `class AppError(Exception)`, `class ConfigurationError(AppError)`, `class DocumentError(AppError)`
  - `class ProviderErrorKind(StrEnum)`: `RATE_LIMIT`, `QUOTA`, `UNAVAILABLE`, `KEY_REJECTED`, `UNKNOWN`
  - `class ProviderError(AppError)` with attributes `kind: ProviderErrorKind`, `provider: str`, `model: str`, `title: str`, `body: str`
  - `classify_provider_error(exc: BaseException, provider: str, model: str) -> ProviderError`

- [ ] **Step 1: Write the failing tests**

```python
from lang_rag_assistant.errors import ProviderErrorKind, classify_provider_error

class _Status(Exception):
    def __init__(self, status_code, message=""):
        super().__init__(message)
        self.status_code = status_code

def test_rate_limit_copy_matches_wireframe():
    err = classify_provider_error(_Status(429), "openai", "gpt-4o-mini")
    assert err.kind is ProviderErrorKind.RATE_LIMIT
    assert err.title == "Rate limit reached"
    assert "gpt-4o-mini" in err.body and "Model page" in err.body

def test_status_mapping():
    assert classify_provider_error(_Status(401), "groq", "m").kind is ProviderErrorKind.KEY_REJECTED
    assert classify_provider_error(_Status(503), "groq", "m").kind is ProviderErrorKind.UNAVAILABLE

def test_quota_from_message():
    exc = Exception("You exceeded your current quota: insufficient_quota")
    assert classify_provider_error(exc, "openai", "m").kind is ProviderErrorKind.QUOTA

def test_unknown_without_status_does_not_crash():
    assert classify_provider_error(ValueError("boom"), "openai", "m").kind is ProviderErrorKind.UNKNOWN
```

- [ ] **Step 2: Run them**

Run: `uv run pytest tests/test_errors.py -v`
Expected: FAIL (module not found) before restoring.

- [ ] **Step 3: Restore `errors.py`**

Restore from history. Add the `title` and `body` copy for each kind. Rate-limit copy comes from the wireframe alert: `"{provider} is limiting requests for {model} right now. Try again in a minute, or choose another model on the Model page."` Other kinds use the copy in UX.md section 6, to be finalized here.

- [ ] **Step 4: Run them to verify**

Run: `uv run pytest tests/test_errors.py -v`
Expected: 4 passed.

---

### Task 3: Settings and logging

**Files:**
- Create: `src/lang_rag_assistant/config/settings.py` (restore from `89af4a3^`, then edit)
- Modify: `src/lang_rag_assistant/config/logging_config.py` (remove the duplicated block in the stub)
- Test: `tests/test_settings.py`

**Interfaces:**
- Consumes: `ConfigurationError` (Task 2).
- Produces:
  - `@dataclass(frozen=True) class Settings`: `provider: str`, `google_model: str | None`, `groq_model: str | None`, `openai_model: str | None`, `keys: dict[str, str]` (only non-empty, stripped), `embedding_model: str`
  - `load_settings(env: Mapping[str, str] | None = None) -> Settings`
  - `configure_logging(level: str | None = None) -> None` (idempotent)
  - `get_logger(name: str) -> logging.Logger`
- Changes from the restored version: `ConfigurationError` on an unknown `EMBEDDING_MODEL`; `keys` uses the lower-cased provider name.

- [ ] **Step 1: Write the failing tests**

```python
from lang_rag_assistant.config.settings import load_settings
from lang_rag_assistant.errors import ConfigurationError
import logging, pytest

def test_defaults():
    s = load_settings({})
    assert s.provider == "google_genai"
    assert s.embedding_model == "all-MiniLM-L6-v2"
    assert s.keys == {}

def test_keeps_only_non_empty_keys():
    s = load_settings({"GROQ_API_KEY": "  ", "OPENAI_API_KEY": "sk-x"})
    assert s.keys == {"openai": "sk-x"}

def test_unknown_embedding_model_is_configuration_error():
    with pytest.raises(ConfigurationError):
        load_settings({"EMBEDDING_MODEL": "bogus"})

def test_configure_logging_is_idempotent():
    from lang_rag_assistant.config.logging_config import configure_logging
    configure_logging(); n = len(logging.getLogger().handlers)
    configure_logging(); assert len(logging.getLogger().handlers) == n
```

- [ ] **Step 2: Run them**

Run: `uv run pytest tests/test_settings.py -v`
Expected: FAIL before restore.

- [ ] **Step 3: Implement**

Restore `settings.py`, apply the two changes above. In `logging_config.py`, keep one copy of the handler setup; the stub duplicated it.

- [ ] **Step 4: Verify**

Run: `uv run pytest tests/test_settings.py -v`
Expected: 4 passed.

---

### Task 4: Model options and factories

**Files:**
- Create: `src/lang_rag_assistant/config/model_config.py` (restore from `89af4a3^`, then edit)
- Test: `tests/test_model_config.py`

**Interfaces:**
- Consumes: `Settings` (Task 3), `ConfigurationError` (Task 2).
- Produces:
  - `@dataclass(frozen=True) class ChatModelOption`: `provider: str`, `model: str`, `label: str` (for example `"OpenAI · gpt-4o-mini"`)
  - `DEFAULT_MODELS: dict[str, str]` with `google_genai`, `groq`, `openai` defaults
  - `EMBEDDING_MODELS: dict[str, str]` mapping the option name to the Hugging Face id
  - `list_chat_model_options(settings: Settings) -> list[ChatModelOption]`: only providers with a key; empty list when no keys
  - `resolve_selection(options: list[ChatModelOption], saved: tuple[str, str] | None) -> ChatModelOption | None`: the saved `(provider, model)` if still in `options`, else `None`
  - `create_chat_model(option: ChatModelOption) -> BaseChatModel`: calls `init_chat_model(model=..., model_provider=...)`
  - `create_embeddings(name: str) -> Embeddings`: `HuggingFaceEmbeddings(model_name=EMBEDDING_MODELS[name])`
- [ ] **Step 1: Write the failing tests**

```python
from lang_rag_assistant.config.settings import Settings
from lang_rag_assistant.config.model_config import list_chat_model_options, resolve_selection, create_chat_model, ChatModelOption

def _settings(keys):
    return Settings(provider="google_genai", google_model=None, groq_model=None,
                    openai_model=None, keys=keys, embedding_model="all-MiniLM-L6-v2")

def test_only_configured_providers_listed():
    opts = list_chat_model_options(_settings({"openai": "sk"}))
    assert {o.provider for o in opts} == {"openai"}

def test_no_keys_gives_no_options():
    assert list_chat_model_options(_settings({})) == []

def test_stale_saved_selection_resolves_to_none():
    opts = list_chat_model_options(_settings({"openai": "sk"}))
    assert resolve_selection(opts, ("groq", "llama-3.1-8b-instant")) is None

def test_create_chat_model_passes_provider(monkeypatch):
    calls = {}
    monkeypatch.setattr("lang_rag_assistant.config.model_config.init_chat_model",
                        lambda **kw: calls.update(kw) or object())
    create_chat_model(ChatModelOption("groq", "llama-3.1-8b-instant", "Groq"))
    assert calls == {"model": "llama-3.1-8b-instant", "model_provider": "groq"}
```

- [ ] **Step 2: Run them**

Run: `uv run pytest tests/test_model_config.py -v`
Expected: FAIL before implementation.

- [ ] **Step 3: Implement**

Restore from history, keep the `list_chat_model_options` gating, and add `resolve_selection`. Keep the `init_chat_model` import at module level so the test can patch it.

- [ ] **Step 4: Verify**

Run: `uv run pytest tests/test_model_config.py -v`
Expected: 4 passed.

---

### Task 5: PDF reader

**Files:**
- Create: `src/lang_rag_assistant/file_reader.py` (restore from `89af4a3^`, then edit)
- Test: `tests/test_file_reader.py`

**Interfaces:**
- Consumes: `DocumentError` (Task 2).
- Produces:
  - `MAX_UPLOAD_BYTES = 200 * 1024 * 1024`
  - `@dataclass(frozen=True) class PageText`: `number: int` (from 1), `text: str`
  - `read_pdf_pages(source: Path | bytes, *, name: str) -> list[PageText]`: raises `DocumentError` for invalid, encrypted, oversized, or text-less PDFs; raises `FileNotFoundError` for a missing path
- Fixes from the stub: `path.exists` becomes `path.exists()`; the size limit is 200 MiB.

- [ ] **Step 1: Write the failing tests**

```python
import io, pytest
from pypdf import PdfWriter
from lang_rag_assistant import file_reader
from lang_rag_assistant.errors import DocumentError
from lang_rag_assistant.file_reader import read_pdf_pages

def _blank_pdf(encrypt=None):
    w = PdfWriter(); w.add_blank_page(width=200, height=200)
    if encrypt: w.encrypt(encrypt)
    buf = io.BytesIO(); w.write(buf); return buf.getvalue()

def test_invalid_bytes_raise_document_error():
    with pytest.raises(DocumentError):
        read_pdf_pages(b"not a pdf", name="notes.pdf")

def test_encrypted_pdf_raises():
    with pytest.raises(DocumentError):
        read_pdf_pages(_blank_pdf(encrypt="secret"), name="locked.pdf")

def test_text_less_pdf_raises():
    with pytest.raises(DocumentError):
        read_pdf_pages(_blank_pdf(), name="scan.pdf")

def test_oversized_raises(monkeypatch):
    monkeypatch.setattr(file_reader, "MAX_UPLOAD_BYTES", 10)
    with pytest.raises(DocumentError):
        read_pdf_pages(b"x" * 11, name="big.pdf")

def test_missing_path_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_pdf_pages(tmp_path / "missing.pdf", name="missing.pdf")
```

- [ ] **Step 2: Run them**

Run: `uv run pytest tests/test_file_reader.py -v`
Expected: FAIL before implementation.

- [ ] **Step 3: Implement**

Restore from history. Wrap `pypdf` errors (`PdfReadError`, encryption) as `DocumentError`. Raise `DocumentError` when no page yields text. Use `path.exists()`.

- [ ] **Step 4: Verify**

Run: `uv run pytest tests/test_file_reader.py -v`
Expected: 5 passed.

---

### Task 6: SQLite repository

**Files:**
- Create: `src/lang_rag_assistant/database.py` (restore from `89af4a3^`, then edit)
- Test: `tests/test_database.py`

**Interfaces:**
- Consumes: nothing.
- Produces (method names as restored; the signatures below are the contract):
  - `Repository(db_path: Path)`; `transaction()` context manager that commits on success and rolls back on any exception
  - `find_document_by_sha256(sha256: str) -> DocumentRecord | None`
  - `add_document(name: str, source_kind: str, sha256: str) -> int`; `list_documents() -> list[DocumentRecord]`
  - `add_chunks(document_id: int, chunks: list[ChunkRecord]) -> None`; `all_chunks() -> list[StoredChunk]`
  - `set_suggestions(document_id: int, questions: list[str]) -> None`; `suggestions_for(document_id: int) -> list[str]`
  - `create_conversation(title: str, provider: str, model: str) -> int`
  - `append_message(conversation_id: int, role: str, text: str, *, stopped: bool = False) -> int`
  - `add_sources(message_id: int, sources: list[SourceRecord]) -> None`
  - `list_conversations(query: str | None = None) -> list[ConversationSummary]`: newest first; `query` matches title case-insensitively
  - `load_messages(conversation_id: int) -> list[MessageRecord]`
  - `get_setting(key: str) -> str | None`; `set_setting(key: str, value: str) -> None`
- Schema: tables `documents`, `chunks` (vector as BLOB), `suggestions`, `conversations`, `messages` (with `stopped` integer column), `message_sources`, `settings`. `sha256` unique on `documents`.

- [ ] **Step 1: Write the failing tests**

```python
from lang_rag_assistant.database import Repository

def test_transaction_rolls_back_on_error(tmp_db_path):
    repo = Repository(tmp_db_path)
    try:
        with repo.transaction():
            repo.add_document("a.pdf", "upload", "h1")
            raise RuntimeError("fail")
    except RuntimeError:
        pass
    assert repo.list_documents() == []

def test_find_by_sha256(tmp_db_path):
    repo = Repository(tmp_db_path)
    doc_id = repo.add_document("a.pdf", "upload", "h1")
    assert repo.find_document_by_sha256("h1").id == doc_id
    assert repo.find_document_by_sha256("missing") is None

def test_stopped_flag_persists(tmp_db_path):
    repo = Repository(tmp_db_path)
    cid = repo.create_conversation("Q", "openai", "gpt-4o-mini")
    repo.append_message(cid, "assistant", "partial", stopped=True)
    assert repo.load_messages(cid)[0].stopped is True

def test_conversation_search_matches_title(tmp_db_path):
    repo = Repository(tmp_db_path)
    repo.create_conversation("What causes climate change?", "openai", "m")
    repo.create_conversation("Oceans", "openai", "m")
    titles = [c.title for c in repo.list_conversations("climate")]
    assert titles == ["What causes climate change?"]

def test_setting_round_trip(tmp_db_path):
    repo = Repository(tmp_db_path)
    repo.set_setting("selected_model", "openai|gpt-4o-mini")
    assert repo.get_setting("selected_model") == "openai|gpt-4o-mini"
```

`tmp_db_path` is the fixture from Task 1.

- [ ] **Step 2: Run them**

Run: `uv run pytest tests/test_database.py -v`
Expected: FAIL before implementation.

- [ ] **Step 3: Implement**

Restore from history. Add the `transaction()` context manager (one connection, `BEGIN` / `COMMIT` / `ROLLBACK`), the `stopped` column, and the `settings` table if missing. Keep `sqlite3.Row` access.

- [ ] **Step 4: Verify**

Run: `uv run pytest tests/test_database.py -v`
Expected: 5 passed.

---

### Task 7: Knowledge base retrieval

**Files:**
- Create: `src/lang_rag_assistant/retrieval.py` (restore from `89af4a3^`)
- Test: `tests/test_retrieval.py`

**Interfaces:**
- Consumes: `Repository.all_chunks()` (Task 6); an embeddings object with `embed_query(text: str) -> list[float]`.
- Produces:
  - `@dataclass(frozen=True) class RetrievedChunk`: `document_name: str`, `page: int`, `text: str`, `score: float`
  - `cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float` (a zero vector gives `0.0`)
  - `KnowledgeBase(repo: Repository, embeddings: Embeddings)`; `search(query: str, k: int = 4) -> list[RetrievedChunk]`, sorted by score, descending; `[]` when nothing is indexed

- [ ] **Step 1: Write the failing tests**

```python
import pytest
from lang_rag_assistant.retrieval import cosine_similarity

def test_identical_vectors_score_one():
    assert cosine_similarity([1.0, 2.0], [1.0, 2.0]) == pytest.approx(1.0)

def test_zero_vector_scores_zero():
    assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0

def test_search_on_empty_index_returns_empty(tmp_db_path, fake_embeddings):
    from lang_rag_assistant.database import Repository
    from lang_rag_assistant.retrieval import KnowledgeBase
    assert KnowledgeBase(Repository(tmp_db_path), fake_embeddings).search("anything") == []
```

`fake_embeddings` is a deterministic test double added to `tests/conftest.py` in this task: `embed_query` maps text to a small vector (for example, the character count of each letter a-e); `embed_documents` maps each text the same way.

- [ ] **Step 2: Run them**

Run: `uv run pytest tests/test_retrieval.py -v`
Expected: FAIL before implementation.

- [ ] **Step 3: Implement**

Restore from history. Keep the pure-Python cosine; it is fast enough for the bundled corpus. Add the `k` cap.

- [ ] **Step 4: Verify**

Run: `uv run pytest tests/test_retrieval.py -v`
Expected: 3 passed.

---

### Task 8: Ingestion

**Files:**
- Create: `src/lang_rag_assistant/ingest.py` (restore from `89af4a3^`, then edit)
- Test: `tests/test_ingest.py`

**Interfaces:**
- Consumes: `read_pdf_pages` (Task 5), `Repository` and its `transaction()` (Task 6), `generate_document_questions` (Task 9), `DocumentError` and `ProviderError` (Task 2).
- Produces:
  - `@dataclass(frozen=True) class IngestResult`: `name: str`, `status: str` (`"added"` or `"duplicate"`), `document_id: int | None`, `warning: str | None`
  - `ingest_pdf(repo, embeddings, embedding_model: str, *, name: str, content: bytes, source_kind: str, chat_model: Any | None) -> IngestResult`
  - `ingest_bundled(repo, embeddings, embedding_model: str, chat_model: Any | None, pdf_dir: Path) -> list[IngestResult]`
  - `reembed_all(*, repo, embeddings, embedding_model: str) -> int`
- Behaviour:
  - Duplicate `sha256` returns `status="duplicate"` and writes nothing.
  - Chunks: 1000 characters, 200 overlap.
  - One `repo.transaction()` covers document, chunks, and embeddings. Any failure rolls back everything.
  - Suggested questions run after the commit. A `ProviderError` there sets `warning` and leaves the document in place.

- [ ] **Step 1: Write the failing tests**

```python
def test_duplicate_upload_is_not_added(tmp_db_path, fake_embeddings, sample_pdf_bytes):
    repo = Repository(tmp_db_path)
    first = ingest_pdf(repo, fake_embeddings, "all-MiniLM-L6-v2", name="a.pdf",
                       content=sample_pdf_bytes, source_kind="upload", chat_model=None)
    second = ingest_pdf(repo, fake_embeddings, "all-MiniLM-L6-v2", name="a.pdf",
                        content=sample_pdf_bytes, source_kind="upload", chat_model=None)
    assert first.status == "added" and second.status == "duplicate"
    assert len(repo.list_documents()) == 1

def test_bundled_ingest_is_idempotent(tmp_db_path, fake_embeddings, bundled_pdf_dir):
    repo = Repository(tmp_db_path)
    ingest_bundled(repo, fake_embeddings, "all-MiniLM-L6-v2", None, bundled_pdf_dir)
    ingest_bundled(repo, fake_embeddings, "all-MiniLM-L6-v2", None, bundled_pdf_dir)
    assert len(repo.list_documents()) == len(list(bundled_pdf_dir.glob("*.pdf")))

def test_question_failure_keeps_document(tmp_db_path, fake_embeddings, sample_pdf_bytes, failing_chat_model):
    repo = Repository(tmp_db_path)
    result = ingest_pdf(repo, fake_embeddings, "all-MiniLM-L6-v2", name="a.pdf",
                        content=sample_pdf_bytes, source_kind="upload", chat_model=failing_chat_model)
    assert result.status == "added" and result.warning
    assert repo.suggestions_for(result.document_id) == []

def test_embedding_failure_leaves_no_rows(tmp_db_path, sample_pdf_bytes, broken_embeddings):
    repo = Repository(tmp_db_path)
    with pytest.raises(Exception):
        ingest_pdf(repo, broken_embeddings, "all-MiniLM-L6-v2", name="a.pdf",
                   content=sample_pdf_bytes, source_kind="upload", chat_model=None)
    assert repo.list_documents() == []

def test_reembed_replaces_vectors(tmp_db_path, fake_embeddings, sample_pdf_bytes):
    repo = Repository(tmp_db_path)
    ingest_pdf(repo, fake_embeddings, "all-MiniLM-L6-v2", name="a.pdf",
               content=sample_pdf_bytes, source_kind="upload", chat_model=None)
    assert reembed_all(repo=repo, embeddings=fake_embeddings, embedding_model="BGE-M3") > 0
```

The fixtures `sample_pdf_bytes`, `bundled_pdf_dir`, `failing_chat_model`, and `broken_embeddings` go in `tests/conftest.py`. `sample_pdf_bytes` builds a one-page PDF with text (for example with `reportlab` or by writing a text stream with `pypdf`). Use `pypdf` to avoid a new dependency.

- [ ] **Step 2: Run them**

Run: `uv run pytest tests/test_ingest.py -v`
Expected: FAIL before implementation.

- [ ] **Step 3: Implement**

Restore from history. Wrap the write path in `repo.transaction()`. Keep the question step outside the transaction and catch `ProviderError` there.

- [ ] **Step 4: Verify**

Run: `uv run pytest tests/test_ingest.py -v`
Expected: 5 passed.

---

### Task 9: Suggested questions

**Files:**
- Create: `src/lang_rag_assistant/suggestions.py` (restore from `89af4a3^`)
- Test: `tests/test_suggestions.py`

**Interfaces:**
- Consumes: `classify_provider_error` (Task 2).
- Produces:
  - `parse_numbered_lines(text: str, n: int) -> list[str]`: strips `1.`, `1)`, `-`, `*` markers, drops empty lines, returns at most `n`
  - `generate_document_questions(chat_model: Any, chunks: list[str], n: int = 6) -> list[str]`
  - `generate_followups(chat_model: Any, question: str, answer: str, n: int = 2) -> list[str]`
  - Model failures raise `ProviderError` (through `classify_provider_error`). `generate_followups` lets the caller decide whether to swallow it.

- [ ] **Step 1: Write the failing tests**

```python
from lang_rag_assistant.suggestions import parse_numbered_lines

def test_strips_markers_and_caps_count():
    text = "1. What is X?\n2) How does Y work?\n- Why Z?\n\n* Extra?"
    assert parse_numbered_lines(text, 3) == ["What is X?", "How does Y work?", "Why Z?"]

def test_empty_text_gives_no_questions():
    assert parse_numbered_lines("", 6) == []
```

- [ ] **Step 2: Run them**

Run: `uv run pytest tests/test_suggestions.py -v`
Expected: FAIL before implementation.

- [ ] **Step 3: Implement**

Restore from history. Change the default `n` for documents to 6.

- [ ] **Step 4: Verify**

Run: `uv run pytest tests/test_suggestions.py -v`
Expected: 2 passed.

---

### Task 10: Search tool and agent

**Files:**
- Create: `src/lang_rag_assistant/tools.py`, `src/lang_rag_assistant/prompts.py`, `src/lang_rag_assistant/agent.py` (restore from `89af4a3^`, then edit)
- Test: `tests/test_tools.py`, `tests/test_agent.py`

**Interfaces:**
- Consumes: `KnowledgeBase` (Task 7), `create_chat_model` (Task 4), `ChatModelOption` (Task 4).
- Produces:
  - `make_search_tool(knowledge_base: KnowledgeBase, k: int = 4) -> BaseTool`: returns numbered passages `[1]`…`[k]` with `Source: <name>, page <p>`; returns `NO_MATCH_MESSAGE` when empty
  - `@dataclass(frozen=True) class StepEvent`: `kind: str` (`"search"`, `"read"`, `"plan"`, `"answer"`), `label: str`, `detail: str`
  - `build_agent(chat_model: Any, knowledge_base: KnowledgeBase) -> CompiledStateGraph`: no side effects at import
  - `map_stream_update(update: Mapping[str, Any]) -> list[StepEvent]`
  - `extract_final_answer(messages: Sequence[BaseMessage]) -> str`
- Change from the restored version: `agent.py` builds the agent inside `build_agent`. The stub ran `agent.invoke` at import time; remove that.

- [ ] **Step 1: Write the failing tests**

```python
def test_importing_agent_has_no_side_effects(monkeypatch):
    import importlib
    called = []
    monkeypatch.setattr("lang_rag_assistant.config.model_config.create_chat_model",
                        lambda *a, **k: called.append(1))
    import lang_rag_assistant.agent as agent
    importlib.reload(agent)
    assert called == []

def test_search_tool_returns_numbered_passages(fake_knowledge_base):
    tool = make_search_tool(fake_knowledge_base, k=2)
    out = tool.invoke({"query": "climate"})
    assert out.startswith("[1]") and "Source:" in out and "page" in out

def test_search_tool_no_match_message(empty_knowledge_base):
    out = make_search_tool(empty_knowledge_base).invoke({"query": "x"})
    assert out == NO_MATCH_MESSAGE

def test_map_search_call_to_step_event():
    update = {"model": {"messages": [AIMessage(content="", tool_calls=[
        {"name": "search_documentation", "args": {"query": "q"}, "id": "1"}])]}}
    events = map_stream_update(update)
    assert events[0].kind == "search" and events[0].label == "Searching documents"
```

`fake_knowledge_base` and `empty_knowledge_base` are fixtures in `tests/conftest.py`, built on `KnowledgeBase` with `fake_embeddings` and a seeded or empty `Repository`.

- [ ] **Step 2: Run them**

Run: `uv run pytest tests/test_tools.py tests/test_agent.py -v`
Expected: FAIL before implementation.

- [ ] **Step 3: Implement**

Restore the three modules. Remove the import-time agent run and the import-time model call. Wrap the graph construction in `build_agent`. Keep the step labels aligned with the wireframes: "Searching documents", "Reading N passages".

- [ ] **Step 4: Verify**

Run: `uv run pytest tests/test_tools.py tests/test_agent.py -v`
Expected: 4 passed.

---

### Task 11: Chat service (one persisted turn)

**Files:**
- Create: `src/lang_rag_assistant/chat_service.py` (restore from `89af4a3^`, then add stop and error handling)
- Test: `tests/test_chat_service.py`

**Interfaces:**
- Consumes: `Repository` (Task 6), `build_agent` and `map_stream_update` and `extract_final_answer` and `StepEvent` (Task 10), `generate_followups` (Task 9), `classify_provider_error` and `ProviderError` (Task 2), `ChatModelOption` (Task 4), `RetrievedChunk` (Task 7).
- Produces:
  - `@dataclass(frozen=True) class TurnResult`: `conversation_id: int`, `answer: str`, `sources: list[RetrievedChunk]`, `followups: list[str]`, `stopped: bool`, `error: ProviderError | None`
  - `class ChatService(repo: Repository, knowledge_base: KnowledgeBase)` with
    `run_turn(conversation_id: int | None, option: ChatModelOption, chat_model: Any, question: str, *, on_event: Callable[[StepEvent], None], should_stop: Callable[[], bool] | None = None) -> TurnResult`
- Behaviour:
  - A blank question raises `ValueError` before any write.
  - The user message is saved first. A new conversation is created when `conversation_id` is `None`, titled with the first 60 characters of the question.
  - Each streamed chunk goes to `on_event` as a `StepEvent`. Text accumulates in a buffer.
  - `should_stop()` returning `True` ends the stream. The partial text is saved as an assistant message with `stopped=True`, and `TurnResult.stopped` is `True`.
  - A `ProviderError` (raised during streaming) is caught. No assistant message is saved, and `TurnResult.error` is set. The user message stays.
  - Follow-up generation runs after a successful answer. A `ProviderError` there gives `followups=[]` and does not change the answer.

- [ ] **Step 1: Write the failing tests**

```python
import pytest
from lang_rag_assistant.chat_service import ChatService
from lang_rag_assistant.errors import ProviderError

def test_turn_persists_question_and_answer(service, option, chat_model):
    result = service.run_turn(None, option, chat_model, "What causes climate change?",
                              on_event=lambda e: None)
    roles = [m.role for m in service.repo.load_messages(result.conversation_id)]
    assert roles == ["user", "assistant"] and result.error is None

def test_blank_question_rejected_before_writing(service, option, chat_model):
    with pytest.raises(ValueError):
        service.run_turn(None, option, chat_model, "   ", on_event=lambda e: None)
    assert service.repo.list_conversations() == []

def test_rate_limit_mid_stream_keeps_question_no_assistant(service, option, rate_limited_model):
    result = service.run_turn(None, option, rate_limited_model, "Q", on_event=lambda e: None)
    assert isinstance(result.error, ProviderError)
    roles = [m.role for m in service.repo.load_messages(result.conversation_id)]
    assert roles == ["user"]

def test_stop_saves_partial_answer_marked_stopped(service, option, slow_model):
    stop = iter([False, True])
    result = service.run_turn(None, option, slow_model, "Q", on_event=lambda e: None,
                              should_stop=lambda: next(stop))
    assert result.stopped is True
    last = service.repo.load_messages(result.conversation_id)[-1]
    assert last.role == "assistant" and last.stopped is True

def test_followup_failure_returns_empty_followups(service, option, followup_failing_model):
    result = service.run_turn(None, option, followup_failing_model, "Q", on_event=lambda e: None)
    assert result.error is None and result.followups == []
```

`service`, `option`, `chat_model`, `rate_limited_model`, `slow_model`, and `followup_failing_model` are fixtures in `tests/conftest.py`. The chat models are fakes that stream fixed chunks, raise a provider-style exception, or raise on the follow-up call.

- [ ] **Step 2: Run them**

Run: `uv run pytest tests/test_chat_service.py -v`
Expected: FAIL before implementation.

- [ ] **Step 3: Implement**

Restore `chat_service.py`, then add the buffer and the stop check inside the stream loop. Wrap the stream in `try/except ProviderError` (the classification happens in the agent call path) and `try/finally` for the partial save. Persist the assistant message and its sources in one `repo.transaction()`.

- [ ] **Step 4: Verify**

Run: `uv run pytest tests/test_chat_service.py -v`
Expected: 5 passed.

---

### Task 12: UI components

**Files:**
- Create: `src/lang_rag_assistant/ui/components.py`
- Test: `tests/test_ui_components.py`

**Interfaces:**
- Consumes: `ProviderError` (Task 2), `ConversationSummary` (Task 6).
- Produces:
  - `show_alert(kind: str, title: str, body: str) -> None`: `kind` is `"error"`, `"info"`, or `"warning"`; calls `st.error` / `st.info` / `st.warning` with `title=`
  - `show_provider_error(err: ProviderError) -> None`: `show_alert("error", err.title, err.body)`
  - `render_user_message(text: str) -> None`: `st.chat_message("user")`
  - `render_assistant_message(text: str) -> None`: `st.chat_message("assistant")`
  - `render_sources(sources: list[RetrievedChunk]) -> None`: `st.expander(f"Cited passages · {n}")`
  - `render_suggestions(questions: list[str], *, key: str, on_pick: Callable[[str], None]) -> None`: `st.pills`
  - `format_recency(updated_at: datetime, now: datetime) -> str`: `"Updated 2 hours ago"`, `"Updated yesterday"`, `"Updated 3 days ago"` (the wireframe labels)

- [ ] **Step 1: Write the failing tests**

```python
from datetime import datetime, timedelta
from lang_rag_assistant.ui.components import format_recency

def test_recency_labels():
    now = datetime(2026, 11, 13, 12, 0)
    assert format_recency(now - timedelta(hours=2), now) == "Updated 2 hours ago"
    assert format_recency(now - timedelta(days=1), now) == "Updated yesterday"
    assert format_recency(now - timedelta(days=3), now) == "Updated 3 days ago"
```

- [ ] **Step 2: Run it**

Run: `uv run pytest tests/test_ui_components.py -v`
Expected: FAIL before implementation.

- [ ] **Step 3: Implement**

Each function is a thin wrapper over one native command. No custom HTML and no CSS. `format_recency` is a pure function.

- [ ] **Step 4: Verify**

Run: `uv run pytest tests/test_ui_components.py -v`
Expected: 1 passed.

---

### Task 13: Services container and the four pages

**Files:**
- Create: `src/lang_rag_assistant/ui/services.py`, `src/lang_rag_assistant/ui/pages/chat.py`, `src/lang_rag_assistant/ui/pages/history.py`, `src/lang_rag_assistant/ui/pages/documents.py`, `src/lang_rag_assistant/ui/pages/model.py`
- Test: `tests/test_app.py` (AppTest, in Task 15)

**Interfaces:**
- Consumes: everything from Tasks 2–12.
- Produces:
  - `@dataclass(frozen=True) class Services`: `settings`, `repo`, `embeddings`, `knowledge_base`, `chat_service`, `options: list[ChatModelOption]`, `embedding_model: str`
  - `get_services() -> Services`: `@st.cache_resource`; builds the repository, embeddings, knowledge base, and service once per process, then runs `ingest_bundled` once
  - each page module exposes `render(services: Services) -> None`

**Page behaviour (from UX.md):**
- **Chat (flows 2, 4, 5, 6):** the sidebar shows `st.selectbox` over `services.options` (disabled when empty). With no options, show `show_alert("info", "No model configured", ...)`, an `st.button` "Open Model page" (`st.switch_page`), and a disabled `st.chat_input`. Empty state: `render_suggestions` with up to 3 document questions. On submit: `st.status` shows steps through `on_event`; `st.chat_input(submit_mode="stop")`; `should_stop` reads a session flag set by the stop action; `TurnResult.error` goes to `show_provider_error` with "Try again" and "Change model" buttons.
- **History (flow 7):** `st.text_input` search calling `list_conversations(query)`; each row is a bordered container with `format_recency` and `st.button("Resume")`. Resume stores the id in `st.session_state` and switches to Chat. Empty state uses `show_alert("info", "No conversations yet", ...)` and a "Go to Chat" button.
- **Documents (flow 8, 9):** "Add a PDF" reveals `st.file_uploader(type=["pdf"])`. "Add this PDF" calls `ingest_pdf` inside `st.status` with `st.progress`. `DocumentError` shows `show_alert("error", "Couldn't read this file", ...)`. A `duplicate` result shows `st.info` "This file is already in your library." The library lists each document as a bordered container with a badge (`Bundled` or `Uploaded`) and an `st.expander` "Suggested questions · N".
- **Model (flow 3):** `st.selectbox` for the chat model (one entry per configured provider, labelled `model · Provider`) and `st.selectbox` for the embedding model. A key table with `st.table` (Provider, Default model, Key: `Configured` / `Missing key`). "Use this model" stores `(provider, model)` in settings. Embedding change opens `st.dialog` to confirm, then calls `reembed_all`. A stale saved selection (see `resolve_selection`) shows `show_alert("warning", "No API key for <provider>", ...)` and disables the button.

- [ ] **Step 1: Write the failing tests**

Covered by Task 15 (AppTest). For this task, add one unit test for `get_services` with no keys:

```python
def test_services_without_keys_have_no_options(monkeypatch, tmp_path):
    from lang_rag_assistant.ui import services
    monkeypatch.setattr(services, "load_settings", lambda: load_settings({}))
    svc = services.build_services(db_path=tmp_path / "x.sqlite", embeddings=fake_embeddings_value)
    assert svc.options == []
```

Add `build_services(db_path, embeddings)` as the unwrapped builder. `get_services` calls it under `st.cache_resource`.

- [ ] **Step 2: Run it**

Run: `uv run pytest tests/test_app.py -k services -v`
Expected: FAIL before implementation.

- [ ] **Step 3: Implement**

Build the four pages from the wireframes using only the components in Task 12 and native `st.*` widgets. Keep each page file under about 150 lines. Keys are unique per page (prefix with the page name).

- [ ] **Step 4: Verify**

Run: `uv run pytest tests/test_app.py -k services -v`
Expected: 1 passed.

---

### Task 14: Entry point, navigation, and theme config

**Files:**
- Rewrite: `src/lang_rag_assistant/main.py`
- Modify: `.streamlit/config.toml`
- Test: covered by Task 15

**Interfaces:**
- Consumes: `get_services` (Task 13), the four page `render` functions.
- Produces: `main()` run by `streamlit run src/lang_rag_assistant/main.py`.

**Behaviour:**
- `st.set_page_config(page_title="Lang RAG Assistant", page_icon=<app_icon.png path>, layout="wide")`.
- `st.logo(<app_icon.png path>)`.
- `st.navigation([st.Page(chat, title="Chat", icon=":material/chat:", default=True), st.Page(history, title="History", icon=":material/history:"), st.Page(documents, title="Documents", icon=":material/description:"), st.Page(model, title="Model", icon=":material/tune:")]).run()`.
- Page bodies are functions that call `render(get_services())`, so each page stays a script-like entry.
- `.streamlit/config.toml`: keep `primaryColor`, `backgroundColor`, `secondaryBackgroundColor`, `textColor`; set `font`, `headingFont` and `codeFont` to the URL values in UX.md section 12.

- [ ] **Step 1: Write the failing check**

Run: `uv run streamlit run src/lang_rag_assistant/main.py --server.headless=true --server.port=8765` in the background, then `curl -s http://127.0.0.1:8765/_stcore/health`.
Expected before the rewrite: the app errors on import.

- [ ] **Step 2: Implement**

Rewrite `main.py` as above. Update `config.toml`.

- [ ] **Step 3: Verify**

Health check returns `ok`. The Agent API (Task 15) loads the app with `status: "ready"`.

---

### Task 15: Verification and parity

**Files:**
- Create: `tests/test_app.py` (AppTest flows)
- Modify: `README.md`, `.env.example` (document `EMBEDDING_MODEL` choices and the 200 MiB limit)

**Interfaces:**
- Consumes: the full app.
- Produces: a passing test suite and a parity note.

- [ ] **Step 1: Write AppTest flows**

```python
from streamlit.testing.v1 import AppTest

def test_no_keys_shows_no_model_alert(monkeypatch, tmp_path):
    monkeypatch.setenv("LANG_RAG_DB_PATH", str(tmp_path / "t.sqlite"))
    for k in ("GOOGLE_API_KEY", "GROQ_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    at = AppTest.from_file("src/lang_rag_assistant/main.py").run()
    assert any("No model configured" in e.value for e in at.info)
    assert at.chat_input and at.chat_input[0].disabled

def test_history_empty_state(monkeypatch, tmp_path):
    monkeypatch.setenv("LANG_RAG_DB_PATH", str(tmp_path / "t.sqlite"))
    at = AppTest.from_file("src/lang_rag_assistant/main.py").run()
    at.switch_page("history.py")  # or click the sidebar Page by title
    at.run()
    assert any("No conversations yet" in e.value for e in at.info)
```

`LANG_RAG_DB_PATH` is added in Task 3's settings, with the default at `data/lang_rag.sqlite`.

- [ ] **Step 2: Run the suite**

Run: `uv run pytest -q`
Expected: all tests pass.

- [ ] **Step 3: Static checks**

Run: `uv run ruff check src tests` and `uv run ty check src`
Expected: both exit 0 (or report only pre-existing items the executor records).

- [ ] **Step 4: Live check with the Agent API**

Start `uv run streamlit run src/lang_rag_assistant/main.py --server.headless=true --server.address=127.0.0.1 --server.port=8765 --server.enableAgentApi=true`. POST `/_stcore/agent/interact` with `{}`. Expected: `status: "ready"`, no `exception` element, no new warnings in console output. Stop the server.

- [ ] **Step 5: Visual parity**

Screenshot each of the four pages in both states (no keys and with a key), and compare against the Doop frames listed in UX.md section 11. Record any difference in the summary.

- [ ] **Step 6: Docs**

Update `README.md` (setup, run command, providers, the 200 MiB limit) and `.env.example`.

---

## Plan self-review

- **Spec coverage:** Every flow in UX.md sections 2–10 maps to a page behaviour in Task 13 or a service in Tasks 2–11. Section 12 decisions appear in Global Constraints and Tasks 3, 4, 5, 8, 11, 13, 14.
- **Type consistency:** `ProviderError` (Task 2) is raised by `classify_provider_error` and caught in Tasks 8, 9, 11. `RetrievedChunk` (Task 7) is used by Tasks 10, 11, 12. `StepEvent` (Task 10) is used by Tasks 11, 13. `ChatModelOption` (Task 4) is used by Tasks 10, 11, 13.
- **Review Focus:** Line 1 is tested in Tasks 5, 8, 15. Line 2 in Task 8. Line 3 in Tasks 4, 13. Line 4 in Tasks 2, 11. Line 5 in Tasks 6, 11.
- **Gaps noted:** the Doop wireframe for the empty state shows three suggestions; Task 13 uses up to three document questions. Scanned PDFs (no text) raise `DocumentError` (Task 5); this is a new decision, not in UX.md, and should be called out at approval.
