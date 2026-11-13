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


from langchain_core.exceptions import (
    ModelAuthenticationError,
    ModelConnectionError,
    ModelRateLimitError,
)


def test_typed_rate_limit_without_status_is_rate_limit():
    err = classify_provider_error(ModelRateLimitError("slow down"), "google_genai", "gemini")
    assert err.kind is ProviderErrorKind.RATE_LIMIT


def test_typed_rate_limit_with_billing_text_stays_rate_limit():
    msg = "You exceeded your current quota, please check your plan and billing details."
    assert classify_provider_error(ModelRateLimitError(msg), "google_genai", "g").kind is ProviderErrorKind.RATE_LIMIT


def test_openai_insufficient_quota_code_is_quota():
    exc = Exception("over budget")
    exc.status_code = 429
    exc.code = "insufficient_quota"
    assert classify_provider_error(exc, "openai", "gpt-4o-mini").kind is ProviderErrorKind.QUOTA_EXCEEDED


def test_typed_authentication_error_is_auth_failed():
    assert classify_provider_error(ModelAuthenticationError("bad key"), "groq", "m").kind is ProviderErrorKind.AUTH_FAILED


def test_typed_connection_error_is_unavailable():
    assert classify_provider_error(ModelConnectionError("reset"), "groq", "m").kind is ProviderErrorKind.UNAVAILABLE


def test_resource_exhausted_text_is_rate_limit_without_a_quota_mention():
    assert classify_provider_error(Exception("429 RESOURCE_EXHAUSTED: slow down"), "google_genai", "g").kind is ProviderErrorKind.RATE_LIMIT


def test_google_invalid_api_key_message_is_auth_failed():
    exc = Exception("400 INVALID_ARGUMENT. API key not valid. Please pass a valid API key.")
    assert classify_provider_error(exc, "google_genai", "gemini").kind is ProviderErrorKind.AUTH_FAILED


def test_openai_no_credits_is_quota():
    msg = "Error code: 429 - You have no credits remaining. Add credits to continue using the API at https://platform.openai.com/settings/organization/billing/."
    # The real OpenAI error is langchain's typed rate-limit error, with no status attribute.
    from langchain_core.exceptions import ModelRateLimitError

    assert classify_provider_error(ModelRateLimitError(msg), "openai", "gpt-4o-mini").kind is ProviderErrorKind.QUOTA_EXCEEDED


def test_groq_per_minute_limit_with_billing_link_is_rate_limit():
    msg = ("Error code: 429 - Rate limit reached for model `qwen/qwen3.8-27b` on input tokens per minute (ITPM). "
           "Please try again in 20.8s. Need more tokens? Upgrade at https://console.groq.com/settings/billing")
    assert classify_provider_error(FakeAPIError(429, msg), "groq", "qwen").kind is ProviderErrorKind.RATE_LIMIT


def test_model_tool_call_mistake_is_model_error_not_unavailable():
    msg = "tool call validation failed: attempted to call tool 'search_device_documentation' which was not in request.tools"
    assert classify_provider_error(Exception(msg), "groq", "qwen").kind is ProviderErrorKind.MODEL_ERROR
