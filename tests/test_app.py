import dataclasses

import pytest
import streamlit as st
from conftest import make_pdf
from streamlit.testing.v1 import AppTest

from lang_rag_assistant.config.settings import load_settings
from lang_rag_assistant.database import Repository
from lang_rag_assistant.errors import ProviderError, ProviderErrorKind
from lang_rag_assistant.ui import services as svc

TIMEOUT = 120
PROVIDER_ENV = ("GOOGLE_API_KEY", "GROQ_API_KEY", "OPENAI_API_KEY")
OPENAI_LABEL = "gpt-4o-mini · OpenAI"


class _FailingModel:
    """Chat model double whose every call fails, so no network is used."""

    def invoke(self, prompt):
        exc = Exception("rate limited")
        exc.status_code = 429
        raise exc


def _failing_chat(kind: ProviderErrorKind):
    """A ChatService stand-in whose every turn fails with `kind`."""

    class _Chat:
        def __init__(self, repo, kb, settings):
            pass

        def run_turn(self, **kwargs):
            raise ProviderError(kind, "openai", "gpt-4o-mini")

    return _Chat


@pytest.fixture
def app_env(tmp_path, monkeypatch, fake_embeddings):
    """The app with no API keys, a temporary data folder, one bundled PDF and fake embeddings."""
    st.cache_resource.clear()
    for var in PROVIDER_ENV:
        monkeypatch.delenv(var, raising=False)
    pdfs = tmp_path / "pdfs"
    pdfs.mkdir()
    (pdfs / "climate_change.pdf").write_bytes(make_pdf("Burning fossil fuels raises CO2."))
    settings = dataclasses.replace(load_settings({}), data_dir=tmp_path, bundled_pdf_dir=pdfs)
    monkeypatch.setattr(svc, "load_settings", lambda: settings)
    monkeypatch.setattr(svc, "create_embeddings", lambda name: fake_embeddings)
    yield settings
    st.cache_resource.clear()


@pytest.fixture
def openai_env(app_env, monkeypatch):
    """As app_env, with an OpenAI key set and a model double that never reaches the network."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(svc, "create_chat_model", lambda option: _FailingModel())
    return app_env


def _repo(settings) -> Repository:
    return Repository(settings.data_dir / "lang_rag.sqlite")


# Page harnesses: AppTest runs each function as a script, so each one renders one page.


def _chat_page() -> None:
    from lang_rag_assistant.ui.pages import chat
    from lang_rag_assistant.ui.services import get_services

    chat.render(get_services())


def _history_page() -> None:
    from lang_rag_assistant.ui.pages import history
    from lang_rag_assistant.ui.services import get_services

    history.render(get_services())


def _documents_page() -> None:
    from lang_rag_assistant.ui.pages import documents
    from lang_rag_assistant.ui.services import get_services

    documents.render(get_services())


def _model_page() -> None:
    from lang_rag_assistant.ui.pages import model
    from lang_rag_assistant.ui.services import get_services

    model.render(get_services())


def test_chat_without_keys_shows_no_model_and_disables_input(app_env):
    at = AppTest.from_function(_chat_page, default_timeout=TIMEOUT).run()
    assert not at.exception
    assert any("Add an API key" in alert.value for alert in at.info)
    assert at.chat_input[0].disabled


def test_history_page_empty_state(app_env):
    at = AppTest.from_function(_history_page, default_timeout=TIMEOUT).run()
    assert not at.exception
    assert any("Ask a question on the Chat page" in alert.value for alert in at.info)


def test_documents_page_lists_bundled_documents(app_env):
    at = AppTest.from_function(_documents_page, default_timeout=TIMEOUT).run()
    assert not at.exception
    assert "climate_change.pdf" in " ".join(md.value for md in at.markdown)


def test_model_page_without_keys_disables_use_button(app_env):
    at = AppTest.from_function(_model_page, default_timeout=TIMEOUT).run()
    assert not at.exception
    assert next(b for b in at.button if b.key == "use_model").disabled


def test_stale_saved_model_lets_user_pick_a_configured_one(openai_env):
    _repo(openai_env).set_setting(svc.CHAT_OPTION_SETTING, "groq:llama-3.1-8b-instant")
    at = AppTest.from_function(_model_page, default_timeout=TIMEOUT).run()
    assert not at.exception
    assert next(b for b in at.button if b.key == "use_model").disabled
    at.selectbox(key="model_chat_pick").set_value(OPENAI_LABEL).run()
    assert not next(b for b in at.button if b.key == "use_model").disabled


def test_rate_limit_offers_try_again_and_change_model(openai_env, monkeypatch):
    from lang_rag_assistant.ui.pages import chat as chat_page

    monkeypatch.setattr(svc, "ChatService", _failing_chat(ProviderErrorKind.RATE_LIMIT))
    monkeypatch.setattr(chat_page, "chat_model_for", lambda option: _FailingModel())
    navigated: list[str] = []
    monkeypatch.setattr(chat_page, "navigate_to", lambda name: navigated.append(name))

    at = AppTest.from_function(_chat_page, default_timeout=TIMEOUT).run()
    at.chat_input[0].set_value("What causes climate change?").run()
    assert not at.exception
    assert any("too many requests" in e.value for e in at.error)
    assert any(b.key == "retry_turn" for b in at.button)
    at.button(key="change_model").click().run()
    assert navigated == ["model"]


def test_auth_failure_offers_change_model_but_not_try_again(openai_env, monkeypatch):
    from lang_rag_assistant.ui.pages import chat as chat_page

    monkeypatch.setattr(svc, "ChatService", _failing_chat(ProviderErrorKind.AUTH_FAILED))
    monkeypatch.setattr(chat_page, "chat_model_for", lambda option: _FailingModel())

    at = AppTest.from_function(_chat_page, default_timeout=TIMEOUT).run()
    at.chat_input[0].set_value("What causes climate change?").run()
    assert any(b.key == "change_model" for b in at.button)
    assert not any(b.key == "retry_turn" for b in at.button)


def test_suggested_question_opens_chat_with_it(app_env, monkeypatch):
    from lang_rag_assistant.ui.pages import documents as documents_page

    AppTest.from_function(_documents_page, default_timeout=TIMEOUT).run()
    repo = _repo(app_env)
    document = repo.list_documents()[0]
    repo.set_suggestions(document.id, ["What are the causes?"])
    at = AppTest.from_function(_documents_page, default_timeout=TIMEOUT).run()

    navigated: list[str] = []
    monkeypatch.setattr(documents_page, "navigate_to", lambda name: navigated.append(name))
    at.button(key=f"ask_{document.id}_0").click().run()
    assert navigated == ["chat"]
    assert at.session_state["pending_question"] == "What are the causes?"


def test_new_chat_leaves_the_current_conversation(app_env):
    at = AppTest.from_function(_chat_page, default_timeout=TIMEOUT).run()
    at.session_state["conversation_id"] = 1
    at.button(key="new_chat").click().run()
    assert not at.exception
    assert "conversation_id" not in at.session_state


def test_resumed_conversation_shows_stored_answer(app_env):
    repo = _repo(app_env)
    conversation_id = repo.create_conversation("What causes climate change?")
    repo.add_message(conversation_id, "user", "What causes climate change?")
    repo.add_message(conversation_id, "assistant", "Fossil fuels raise CO2 levels.", followups=["How do oceans respond?"])
    at = AppTest.from_function(_chat_page, default_timeout=TIMEOUT)
    at.session_state["conversation_id"] = conversation_id
    at.run()
    assert not at.exception
    assert any("Fossil fuels raise CO2 levels." in m.value for m in at.markdown)


def test_stopped_answer_shows_its_stopped_caption(app_env):
    repo = _repo(app_env)
    conversation_id = repo.create_conversation("Why?")
    repo.add_message(conversation_id, "user", "Why?")
    repo.add_message(conversation_id, "assistant", "Glaciers melt", stopped=True)
    at = AppTest.from_function(_chat_page, default_timeout=TIMEOUT)
    at.session_state["conversation_id"] = conversation_id
    at.run()
    assert not at.exception
    assert any(c.value == "Stopped" for c in at.caption)


def _raising_chat(error):
    """A ChatService stand-in whose every turn raises `error`."""

    class _Chat:
        def __init__(self, repo, kb, settings):
            pass

        def run_turn(self, **kwargs):
            raise error

    return _Chat


def test_tool_failure_shows_a_plain_message_and_no_internal_detail(openai_env, monkeypatch):
    from lang_rag_assistant.errors import ToolFailureError
    from lang_rag_assistant.ui.pages import chat as chat_page

    monkeypatch.setattr(svc, "ChatService", _raising_chat(ToolFailureError()))
    monkeypatch.setattr(chat_page, "chat_model_for", lambda option: _FailingModel())
    at = AppTest.from_function(_chat_page, default_timeout=TIMEOUT).run()
    at.chat_input[0].set_value("What causes climate change?").run()
    assert not at.exception
    assert any("Something went wrong while searching the documents" in e.value for e in at.error)


def test_unexpected_failure_never_shows_the_exception_text(openai_env, monkeypatch):
    from lang_rag_assistant.ui.pages import chat as chat_page

    monkeypatch.setattr(svc, "ChatService", _raising_chat(RuntimeError("disk I/O error at /secret/path")))
    monkeypatch.setattr(chat_page, "chat_model_for", lambda option: _FailingModel())
    at = AppTest.from_function(_chat_page, default_timeout=TIMEOUT).run()
    at.chat_input[0].set_value("What causes climate change?").run()
    assert not at.exception
    shown = " ".join(e.value for e in at.error) + " ".join(m.value for m in at.markdown)
    assert "/secret/path" not in shown and "disk I/O" not in shown
    assert any("Something went wrong" in e.value for e in at.error)
