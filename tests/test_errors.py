from lang_rag_assistant.errors import ProviderErrorKind, classify_provider_error


class FakeAPIError(Exception):
    def __init__(self, status_code: int | None, message: str = "", name: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        if name:
            self.__class__ = type(name, (FakeAPIError,), {})


def _classify(exc):
    return classify_provider_error(exc, "groq", "llama-3.1-8b-instant")


def test_429_is_rate_limit():
    assert _classify(FakeAPIError(429, "Too many requests")).kind is ProviderErrorKind.RATE_LIMIT


def test_429_with_quota_text_is_quota():
    err = _classify(FakeAPIError(429, "insufficient_quota: you exceeded your current quota"))
    assert err.kind is ProviderErrorKind.QUOTA_EXCEEDED


def test_401_and_403_are_auth_failed():
    assert _classify(FakeAPIError(401)).kind is ProviderErrorKind.AUTH_FAILED
    assert _classify(FakeAPIError(403)).kind is ProviderErrorKind.AUTH_FAILED


def test_5xx_is_unavailable():
    assert _classify(FakeAPIError(503)).kind is ProviderErrorKind.UNAVAILABLE


def test_timeout_by_class_name_is_unavailable():
    class APITimeoutError(Exception):
        pass

    assert _classify(APITimeoutError("timed out")).kind is ProviderErrorKind.UNAVAILABLE


def test_user_message_is_exact_copy():
    err = _classify(FakeAPIError(429, "Too many requests"))
    assert err.user_message == (
        "The model is receiving too many requests right now. Wait a moment and try again."
    )


def test_provider_and_model_are_kept():
    err = _classify(FakeAPIError(401))
    assert (err.provider, err.model) == ("groq", "llama-3.1-8b-instant")
