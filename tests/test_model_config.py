from lang_rag_assistant.config.model_config import list_chat_model_options
from lang_rag_assistant.config.settings import load_settings


def test_options_list_one_per_provider_with_key_flags():
    env = {"GOOGLE_API_KEY": "x"}
    options = list_chat_model_options(load_settings(env=env), env)
    by_provider = {o.provider: o for o in options}
    assert set(by_provider) == {"google_genai", "groq", "openai"}
    assert by_provider["google_genai"].has_key is True
    assert by_provider["groq"].has_key is False
    assert by_provider["openai"].has_key is False


def test_empty_key_counts_as_missing():
    env = {"GROQ_API_KEY": "   "}
    options = list_chat_model_options(load_settings(env=env), env)
    assert next(o for o in options if o.provider == "groq").has_key is False


def test_option_key_is_provider_and_model():
    env = {"OPENAI_API_KEY": "x"}
    option = next(o for o in list_chat_model_options(load_settings(env=env), env) if o.provider == "openai")
    assert option.key == "openai:gpt-4o-mini"
