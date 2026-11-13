# Lang RAG Assistant

![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)
![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)
![Tests: 100 passing](https://img.shields.io/badge/tests-100%20passing-brightgreen.svg)
![Code style: ruff](https://img.shields.io/badge/code_style-ruff-ff69b4.svg)
![Type check: ty](https://img.shields.io/badge/type_check-ty-8957e5.svg)
![Package manager: uv](https://img.shields.io/badge/package_manager-uv-3478f6.svg)
![Framework: Streamlit](https://img.shields.io/badge/framework-Streamlit-ff4b4b.svg)

A Streamlit chatbot that answers questions from PDF documents. Every answer cites the passages it came from. Built on LangChain and deepagents, with a local embedding model and SQLite storage.

## Pages

- **Chat**: ask a question and read the answer with its cited passages. Follow-up questions are suggested under each answer. Stop cancels a running answer and keeps what has streamed.
- **History**: search saved conversations and resume one.
- **Documents**: the bundled PDFs and any PDFs you add. Each document has suggested questions.
- **Model**: choose the chat model and the embedding model. Only models whose provider has an API key are listed.

## Project Structure

```
lang-rag-assistant/
├── pyproject.toml                  # Project config, runtime and dev dependencies
├── .env.example                    # Template for environment variables
├── .env                            # Your local keys (gitignored, create from .env.example)
├── .streamlit/config.toml          # Theme, fonts and upload limit
├── pdfs/                           # Bundled PDFs, indexed on first start
├── data/                           # SQLite database and conversation workspaces (created at runtime)
├── logs/                           # Rotating log files (created at runtime)
├── src/lang_rag_assistant/
│   ├── main.py                     # Streamlit entry point and navigation
│   ├── agent.py                    # Deep agent builder, chunk-analyst subagent, step events
│   ├── prompts.py                  # Workflow and subagent instructions
│   ├── tools.py                    # search_documentation tool
│   ├── chat_service.py             # One persisted chat turn, streaming and Stop
│   ├── retrieval.py                # KnowledgeBase: cosine search over stored passages
│   ├── ingest.py                   # PDF indexing, bundled PDFs, re-embedding
│   ├── file_reader.py              # PDF page extraction and size limit
│   ├── suggestions.py              # Suggested questions and follow-ups
│   ├── database.py                 # SQLite repository
│   ├── errors.py                   # Provider error classification and plain messages
│   ├── config/                     # Settings, model options, logging
│   └── ui/                         # Services, components, theme and the four pages
└── tests/                          # Unit and AppTest UI tests
```

## Architecture

### Component diagram

The UI calls application services. The chat service runs the deep agent, which calls the retrieval layer and the chat model. Storage holds the documents, the vectors and the conversations.

```mermaid
flowchart LR
    subgraph UI["Streamlit UI"]
        PAGES["Pages: Chat, History, Documents, Model"]
        SVC["services.py: cached services"]
    end

    subgraph APP["Application services"]
        CS["ChatService: one persisted turn"]
        ING["Ingest, PDF reader and suggestions"]
        MC["Model config: options and factory"]
    end

    subgraph AGENT["Agent (deepagents)"]
        AG["Deep agent with chunk-analyst subagent"]
        TOOL["search_documentation tool"]
    end

    KB["KnowledgeBase: cosine search"]
    DB[("SQLite: documents, chunks, conversations")]
    WS["Conversation workspaces: retrieved passages"]
    LLM["Chat models: Google, Groq, OpenAI"]
    EMB["Embedding model: local, sentence-transformers"]
    CFG[".env read by settings.py"]

    CFG --> SVC
    PAGES --> SVC
    SVC --> CS
    SVC --> KB
    CS --> AG
    CS --> DB
    AG --> TOOL
    AG --> LLM
    TOOL --> KB
    TOOL --> WS
    KB --> DB
    KB --> EMB
    ING --> EMB
    ING --> DB
    MC --> LLM

    linkStyle default stroke:#64748B,stroke-width:1.5px

    classDef ui fill:#E0F2FE,stroke:#0369A1,color:#0C4A6E
    classDef app fill:#EDE9FE,stroke:#6D28D9,color:#2E1065
    classDef agent fill:#FEF3C7,stroke:#B45309,color:#451A03
    classDef ret fill:#DCFCE7,stroke:#15803D,color:#14532D
    classDef store fill:#E5E7EB,stroke:#4B5563,color:#111827
    classDef ext fill:#FCE7F3,stroke:#BE185D,color:#500724
    classDef cfg fill:#F5F5F4,stroke:#78716C,color:#1C1917
    class PAGES,SVC ui
    class CS,ING,MC app
    class AG,TOOL agent
    class KB ret
    class DB,WS store
    class LLM,EMB ext
    class CFG cfg
    style UI fill:#F1F5F9,stroke:#94A3B8,color:#0F172A
    style APP fill:#F1F5F9,stroke:#94A3B8,color:#0F172A
    style AGENT fill:#F1F5F9,stroke:#94A3B8,color:#0F172A
```


### Flow chart

Two paths share the database. Indexing runs at startup and on each upload. A chat turn runs when you send a question.

```mermaid
flowchart TD
    subgraph IDX["Indexing"]
        A1["PDF from pdfs/ or an upload"] --> A2["Read each page with pypdf; reject over 200 MB"]
        A2 --> A3{"SHA-256 already stored?"}
        A3 -- yes --> A4["Skip: return the existing document"]
        A3 -- no --> A5["Split each page: 1000 characters, 200 overlap"]
        A5 --> A6["Embed passages with the active embedding model"]
        A6 --> A7[("SQLite: document, chunks with vectors and page numbers")]
        A7 --> A8["KnowledgeBase refresh: load vectors for that model"]
        A8 --> A9["Suggest 6 questions from the first 8 passages"]
        A9 --> A10[("SQLite: suggestions")]
    end

    subgraph TURN["Chat turn"]
        B1["User question"] --> B2[("Save the user message")]
        B2 --> B3["Load the last 6 messages as history"]
        B3 --> B4["Deep agent plans search queries with write_todos"]
        B4 --> B5["search_documentation(query)"]
        B5 --> B6["Embed the query with the same embedding model"]
        B6 --> B7["Cosine similarity against every stored passage; keep top 4"]
        B7 --> B8["Write the 4 passages to the conversation workspace"]
        B8 --> B9["task(): one chunk-analyst per file, in parallel, 3 by default"]
        B9 --> B10["Each analyst reads its file and returns facts with a citation number"]
        B10 --> B11["Agent writes the answer with citation numbers, or searches again"]
        B11 --> B12["Tokens stream to the UI"]
        B12 --> B13{"Stop pressed?"}
        B13 -- yes --> B14[("Save the streamed text as a Stopped answer")]
        B13 -- no --> B15["Map citation numbers to passages"]
        B15 --> B16["Generate 2 follow-up questions"]
        B16 --> B17[("Save the assistant message with sources and follow-ups")]
        B17 --> B18["UI: answer, cited passages, follow-up pills"]
    end

    A7 -. "passages searched" .-> B7
    B9 -. "failures: classified and shown as plain messages" .-> B18

    linkStyle default stroke:#64748B,stroke-width:1.5px

    style IDX fill:#F1F5F9,stroke:#94A3B8,color:#0F172A
    style TURN fill:#F1F5F9,stroke:#94A3B8,color:#0F172A

    classDef step fill:#F8FAFC,stroke:#475569,color:#0F172A
    classDef store fill:#E2E8F0,stroke:#334155,color:#0F172A
    classDef decision fill:#FEF9C3,stroke:#A16207,color:#422006
    class A1,A2,A4,A5,A6,A8,A9,B1,B3,B4,B5,B6,B7,B8,B9,B10,B11,B12,B15,B16,B18 step
    class A7,A10,B2,B14,B17 store
    class A3,B13 decision
```

### Flow summary

1. **Index**: each PDF is read page by page, split into 1000-character passages with 200 characters of overlap, and embedded locally. Identical files are stored once, found by SHA-256.
2. **Ask**: a question is saved, and the last six messages are added as history.
3. **Retrieve**: the agent calls `search_documentation`. The query is embedded, scored by cosine similarity against every stored passage of the active embedding model, and the top four are written to the conversation's workspace.
4. **Analyse**: one chunk-analyst subagent per passage file reads it in parallel, up to `MAX_CONCURRENT_ANALYSTS` at a time.
5. **Answer**: the agent writes the answer with `[n]` citations. Tokens stream to the screen. Stop saves the text streamed so far.
6. **Persist**: the cited passages and two follow-up questions are saved with the message.
7. **Errors**: provider and tool failures are classified (rate limit, quota, rejected key, unavailable, model error). The UI shows a plain message and never the exception text.

## Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) (recommended package manager)
- At least one API key: Google GenAI, Groq or OpenAI

## Setup

```bash
uv sync
cp .env.example .env   # then add at least one API key
```

Edit `.env` and set:

| Variable | Description | Required |
|----------|-------------|----------|
| `LLM_PROVIDER` | Model preselected in the app: `google_genai`, `groq` or `openai` | No (default: `google_genai`) |
| `GOOGLE_MODEL` | Gemini model name | No (default: `gemini-3.1-flash-lite`) |
| `GROQ_MODEL` | Groq model name | No (default: `llama-3.1-8b-instant`) |
| `OPENAI_MODEL` | OpenAI model name | No (default: `gpt-4o-mini`) |
| `GOOGLE_API_KEY` | Google Gemini API key (from https://makersuite.google.com/app/apikey) | At least one key |
| `GROQ_API_KEY` | Groq API key (from https://console.groq.com/keys) | At least one key |
| `OPENAI_API_KEY` | OpenAI API key (from https://platform.openai.com/api-keys) | At least one key |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2`, `nomic-embed-text` or `BGE-M3` | No (default: `all-MiniLM-L6-v2`) |
| `MAX_CONCURRENT_ANALYSTS` | Parallel chunk-analyst subagents, 2 to 5 | No (default: `3`) |
| `LOG_LEVEL` | `DEBUG`, `INFO`, `WARNING`, `ERROR` or `CRITICAL` | No (default: `INFO`) |

A model can be chosen in the app only when its key is set. Uploads are limited to 200 MB per PDF.

## Running the App

```bash
uv run streamlit run src/lang_rag_assistant/main.py
```

The first start downloads the embedding model from Hugging Face, then indexes the PDFs in `pdfs/`. Suggested questions need a configured API key. Open the URL that Streamlit prints, usually http://localhost:8501.

## Running Tests

```bash
uv run pytest          # 100 tests
uv run ruff check src tests
uv run ty check src    # type check
```

The test badge reflects the last local run. The repository has no CI workflow yet.

## Configuration

Settings are read from `.env` by `config/settings.py`. Passages are chunked at 1000 characters with 200 characters of overlap. Each search returns the four closest passages.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

### Summary

| Aspect | Details |
|--------|---------|
| **License** | MIT License |
| **Copyright** | 2026 Vidhaata Labs (Karthik Prabhu K) |
| **Permissions** | Commercial use, distribution, modification, private use |
| **Limitations** | No warranty, no liability |
| **Conditions** | Include copyright and license notice in copies |

The software is provided "as is", without warranty of any kind. The authors or copyright holders are not liable for any claims, damages, or other liability arising from the use of this software.
