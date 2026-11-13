import pytest

from lang_rag_assistant.config.settings import load_settings
from lang_rag_assistant.errors import ConfigurationError


def test_defaults_use_google_and_groq_llama():
    s = load_settings(env={})
    assert s.llm_provider == "google_genai"
    assert s.chat_model_names["groq"] == "llama-3.1-8b-instant"
    assert s.chat_model_names["google_genai"] == "gemini-3.1-flash-lite"
    assert s.embedding_model == "all-MiniLM-L6-v2"
    assert s.max_concurrent_analysts == 3
    assert s.log_level == "INFO"


def test_analysts_out_of_range_raises():
    with pytest.raises(ConfigurationError):
        load_settings(env={"MAX_CONCURRENT_ANALYSTS": "9"})


def test_analysts_not_integer_raises():
    with pytest.raises(ConfigurationError):
        load_settings(env={"MAX_CONCURRENT_ANALYSTS": "three"})


def test_unknown_embedding_raises():
    with pytest.raises(ConfigurationError):
        load_settings(env={"EMBEDDING_MODEL": "nope"})


def test_unknown_provider_raises():
    with pytest.raises(ConfigurationError):
        load_settings(env={"LLM_PROVIDER": "anthropic"})
