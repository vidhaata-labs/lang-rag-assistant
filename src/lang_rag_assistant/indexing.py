from pathlib import Path
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from lang_rag_assistant.file_reader import read_pdfs
from lang_rag_assistant.config.logging_config import get_logger
from lang_rag_assistant.config.model_config import create_embeddings

logger = get_logger(__name__)

embeddings = create_embeddings()

vector_store = InMemoryVectorStore(embeddings)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DOC_PATHS = [
    PROJECT_ROOT / "pdfs" / "climate_change.pdf",
    PROJECT_ROOT / "pdfs" / "global_warming.pdf"
]

logger.info("Reading PDFs from %s", PROJECT_ROOT / "pdfs")
docs = read_pdfs([str(p) for p in DOC_PATHS])
logger.info("Read %d document(s)", len(docs))

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
doc_chunks = text_splitter.split_documents(docs)

logger.info("Split documentation into %d chunks", len(doc_chunks))

vector_store.add_documents(documents=doc_chunks)

logger.info("Indexed %d chunks into vector store", len(doc_chunks))
