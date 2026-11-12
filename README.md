# Lang RAG Assistant

A document Q&A chatbot built on LangChain and Deep Agents. Ask questions about PDFs; answers cite the passages they came from.

## Setup

1. Install dependencies: `uv sync`
2. Copy `.env.example` to `.env` and set at least one API key (`GOOGLE_API_KEY`, `GROQ_API_KEY` or `OPENAI_API_KEY`). A model can be chosen in the app only when its key is set.

## Run

```bash
uv run streamlit run src/lang_rag_assistant/main.py
```

The app has four pages:

- **Chat**: ask questions, see the agent's steps, cited passages, and related follow-up questions.
- **History**: search and resume past conversations.
- **Documents**: add PDFs (text-based, up to 20 MB), review suggested questions, remove uploaded documents.
- **Model and keys**: choose the chat model and the embedding model.

Bundled documents live in `pdfs/` and are indexed on first start.

## Test

```bash
uv run pytest
```

## Data

Conversations, documents, passages and vectors are stored in `data/lang_rag.sqlite`. Logs are written to `logs/`. Both folders are git-ignored.

## Configuration

See `.env.example` for every setting: provider and model names, embedding model, concurrent analysts (2 to 5), and log level.
