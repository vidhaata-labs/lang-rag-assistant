from lang_rag_assistant.config.model_config import (
    REQUESTY_BASE_URL,
    create_chat_model,
    list_chat_model_options,
)
from lang_rag_assistant.config.settings import load_settings


def test_options_list_one_per_provider_with_key_flags():
    env = {"GOOGLE_API_KEY": "x"}
    options = list_chat_model_options(load_settings(env=env), env)
    by_provider = {o.provider: o for o in options}
    assert set(by_provider) == {"google_genai", "groq", "openai", "requesty", "nvidia"}
    assert by_provider["google_genai"].has_key is True
    assert by_provider["groq"].has_key is False
    assert by_provider["openai"].has_key is False
    assert by_provider["requesty"].has_key is False
    assert by_provider["nvidia"].has_key is False


def test_empty_key_counts_as_missing():
    env = {"GROQ_API_KEY": "   "}
    options = list_chat_model_options(load_settings(env=env), env)
    assert next(o for o in options if o.provider == "groq").has_key is False


def test_option_key_is_provider_and_model():
    env = {"OPENAI_API_KEY": "x"}
    option = next(o for o in list_chat_model_options(load_settings(env=env), env) if o.provider == "openai")
    assert option.key == "openai:gpt-4o-mini"


def test_requesty_default_model_and_key_flag():
    env = {"REQUESTY_API_KEY": "sk-requesty"}
    option = next(o for o in list_chat_model_options(load_settings(env=env), env) if o.provider == "requesty")
    assert option.model == "nemotron-3-ultra-550b-a55b"
    assert option.has_key is True
    assert option.key == "requesty:nemotron-3-ultra-550b-a55b"
    assert option.api_key_env == "REQUESTY_API_KEY"


def test_requesty_chat_model_uses_openai_router(monkeypatch):
    captured: dict = {}

    def fake_init(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("langchain.chat_models.init_chat_model", fake_init)
    monkeypatch.setenv("REQUESTY_API_KEY", "sk-requesty")
    env = {"REQUESTY_API_KEY": "sk-requesty"}
    option = next(o for o in list_chat_model_options(load_settings(env=env), env) if o.provider == "requesty")
    create_chat_model(option)
    assert captured == {
        "model": "nemotron-3-ultra-550b-a55b",
        "model_provider": "openai",
        "max_retries": 1,
        "api_key": "sk-requesty",
        "base_url": REQUESTY_BASE_URL,
    }
