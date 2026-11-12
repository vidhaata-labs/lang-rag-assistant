from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

MAIN = Path(__file__).resolve().parents[1] / "src" / "lang_rag_assistant" / "main.py"


@pytest.fixture
def no_keys(monkeypatch):
    # An empty value counts as "not set" and stops the project .env from supplying a key.
    for var in ("GOOGLE_API_KEY", "GROQ_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.setenv(var, "")


def test_app_without_keys_disables_send_and_shows_warning(no_keys):
    at = AppTest.from_file(str(MAIN), default_timeout=180).run()
    assert not at.exception
    assert any("No model has an API key" in w.value for w in at.warning)
    assert at.chat_input and at.chat_input[0].disabled
