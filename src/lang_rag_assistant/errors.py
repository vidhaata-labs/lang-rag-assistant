"""Application exception types shared across layers, and provider error classification."""

from __future__ import annotations

from enum import StrEnum

from langchain_core.exceptions import (
    ModelAuthenticationError,
    ModelInvalidRequestError,
    ModelRateLimitError,
)

from lang_rag_assistant.config.logging_config import get_logger

logger = get_logger(__name__)


class AppError(Exception):
    """Base class for errors the user can be shown a message for.

    `user_message` is the only text shown to the user. The exception's own text is
    for the log, and may hold internal detail, so it is never shown.
    """

    @property
    def user_message(self) -> str:
        return "Something went wrong. Try again."


class ToolFailureError(AppError):
    """A tool the agent relies on, such as document search, failed while answering."""

    @property
    def user_message(self) -> str:
        return "Something went wrong while searching the documents. Try again."


class ConfigurationError(AppError):
    """A required setting is missing or has an invalid value."""


class DocumentError(AppError):
    """An uploaded or bundled document cannot be read or indexed."""


class ProviderErrorKind(StrEnum):
    """Kinds of failure returned by a model provider, each with its own message."""

    RATE_LIMIT = "rate_limit"
    QUOTA_EXCEEDED = "quota_exceeded"
    AUTH_FAILED = "auth_failed"
    UNAVAILABLE = "unavailable"
    MODEL_ERROR = "model_error"


_USER_MESSAGES: dict[ProviderErrorKind, str] = {
    ProviderErrorKind.RATE_LIMIT: (
        "The model is receiving too many requests right now. Wait a moment and try again."
    ),
    ProviderErrorKind.QUOTA_EXCEEDED: (
        "This key has reached its usage limit. Check billing with the provider "
        "or choose another model."
    ),
    ProviderErrorKind.AUTH_FAILED: (
        "The API key was rejected. Check the key in the environment file and restart the app."
    ),
    ProviderErrorKind.MODEL_ERROR: (
        "The model could not complete this request. Try again, or choose another model on the Model page."
    ),
    ProviderErrorKind.UNAVAILABLE: (
        "Couldn't reach the model provider. Check your connection and try again."
    ),
}


class ProviderError(AppError):
    """A model call failed for a reason the user can act on."""

    def __init__(
        self,
        kind: ProviderErrorKind,
        provider: str,
        model: str,
        original: BaseException | None = None,
    ) -> None:
        super().__init__(f"{provider}:{model} failed: {kind.value}")
        self.kind = kind
        self.provider = provider
        self.model = model
        self.__cause__ = original

    @property
    def user_message(self) -> str:
        """Exact copy shown to the user for this kind of failure."""
        return _USER_MESSAGES[self.kind]


def _status_code(exc: BaseException) -> int | None:
    """Return the HTTP status carried by a provider exception, if any."""
    for attribute in ("status_code", "code"):
        value = getattr(exc, attribute, None)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _model_mistake(exc: BaseException, status: int | None, text: str) -> bool:
    """The model sent a request the provider cannot run, such as a tool call it was never offered."""
    return "tool call validation failed" in text or isinstance(exc, ModelInvalidRequestError) or status == 400


def _rejects_api_key(text: str) -> bool:
    """Providers report a bad key as a 400 or 403 with a message naming the key, not as a 401."""
    return ("api key" in text or "api_key" in text) and any(
        word in text for word in ("not valid", "invalid", "expired", "incorrect", "permission_denied")
    )


def classify_provider_error(exc: BaseException, provider: str, model: str) -> ProviderError:
    """Map a provider exception to a ProviderError with the matching kind.

    An `insufficient_quota` code or a "no credits" message (OpenAI's exhausted budget) is QUOTA_EXCEEDED.
    langchain_core's typed errors are checked next: authentication is AUTH_FAILED
    and a rate limit is RATE_LIMIT, even when the provider gives no status code
    (Gemini's per-minute limit has none). 401 and 403 are AUTH_FAILED. A 429 is
    QUOTA_EXCEEDED when the text says quota, otherwise RATE_LIMIT. A billing link alone is not a quota signal.
    Everything else (5xx, timeouts, connection failures, unknown errors) is UNAVAILABLE.
    """
    status = _status_code(exc)
    text = str(exc).lower()
    if getattr(exc, "code", None) == "insufficient_quota" or "insufficient_quota" in text or "no credits" in text:
        kind = ProviderErrorKind.QUOTA_EXCEEDED
    elif status in (401, 403) or isinstance(exc, ModelAuthenticationError) or _rejects_api_key(text):
        kind = ProviderErrorKind.AUTH_FAILED
    elif isinstance(exc, ModelRateLimitError) or "resource_exhausted" in text and "quota" not in text:
        kind = ProviderErrorKind.RATE_LIMIT
    elif _model_mistake(exc, status, text):
        kind = ProviderErrorKind.MODEL_ERROR
    elif status == 429:
        kind = (
            ProviderErrorKind.QUOTA_EXCEEDED
            if "quota" in text
            else ProviderErrorKind.RATE_LIMIT
        )
    else:
        kind = ProviderErrorKind.UNAVAILABLE
    logger.error("Provider %s:%s failed (%s)", provider, model, kind.value, exc_info=exc)
    return ProviderError(kind, provider, model, exc)
