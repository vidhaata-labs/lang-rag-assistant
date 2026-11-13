from typing import Any

from langchain.chat_models import init_chat_model

from lang_rag_assistant.config.logging_config import get_logger

logger = get_logger(__name__)

# Provider-specific model defaults
DEFAULT_MODELS = {
    "google_genai": "gemini-3.1-flash-lite",
    "groq": "llama-3.1-8b-instant",
    "openai": "gpt-4o-mini",
}

# Map of provider to the env var name holding its model choice
MODEL_ENV_VARS = {
    "google_genai": "GOOGLE_MODEL",
    "groq": "GROQ_MODEL",
    "openai": "OPENAI_MODEL",
}


def get_llm_provider() -> str:
    """Return the LLM provider from env, defaulting to google_genai."""
    import os

    provider = os.getenv("LLM_PROVIDER", "google_genai").strip().lower()
    logger.debug("LLM_PROVIDER read from env: %s", provider)
    valid = list(DEFAULT_MODELS.keys())
    if provider not in valid:
        logger.warning(
            "Unknown LLM_PROVIDER '%s'. Falling back to '%s'. Valid: %s",
            provider,
            DEFAULT_MODELS.keys(),
            valid,
        )
        provider = "google_genai"
    logger.info("Using LLM provider: %s", provider)
    return provider


def get_model_name(provider: str | None = None) -> str:
    """Return the model name for the given provider from env or default."""
    import os

    provider = provider or get_llm_provider()
    env_var = MODEL_ENV_VARS.get(provider)
    if env_var:
        model = os.getenv(env_var)
        if model:
            logger.info("Using model from env var %s: %s", env_var, model.strip())
            return model.strip()
    default = DEFAULT_MODELS.get(provider, DEFAULT_MODELS["google_genai"])
    logger.info("Using default model for provider %s: %s", provider, default)
    return default


def create_llm(provider: str | None = None, model: str | None = None) -> Any:
    """Create a chat model for the configured provider.

    Provider and model can be overridden via arguments; otherwise they are
    read from environment variables (LLM_PROVIDER, <PROVIDER>_MODEL).
    """
    provider = provider or get_llm_provider()
    model = model or get_model_name(provider)

    logger.info("Creating LLM model: provider=%s, model=%s", provider, model)
    return init_chat_model(model=model, model_provider=provider)


def get_available_providers() -> list[str]:
    """Return the list of supported LLM providers."""
    return list(DEFAULT_MODELS.keys())


# Embedding model defaults (native/local via sentence-transformers).
EMBEDDING_MODELS = {
    "BGE-M3": "BAAI/bge-m3",
    "nomic-embed-text": "nomic-ai/nomic-embed-text-v1.5",
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
}


def get_embedding_model_name() -> str:
    """Return the embedding model name from the EMBEDDING_MODEL env var."""
    import os

    name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    logger.debug("EMBEDDING_MODEL read from env: %s", name)
    if name not in EMBEDDING_MODELS:
        raise ValueError(
            f"Unsupported EMBEDDING_MODEL '{name}'. "
            f"Valid options: {list(EMBEDDING_MODELS)}"
        )
    logger.info("Using embedding model: %s (%s)", name, EMBEDDING_MODELS[name])
    return EMBEDDING_MODELS[name]


def create_embeddings():
    """Create HuggingFace embeddings for the configured embedding model."""
    from langchain_huggingface import HuggingFaceEmbeddings

    model_name = get_embedding_model_name()
    logger.info("Creating embeddings model: %s", model_name)
    return HuggingFaceEmbeddings(model_name=model_name)
