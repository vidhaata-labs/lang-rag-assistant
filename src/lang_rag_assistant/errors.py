"""Application exception types shared across layers, and provider error classification."""

from __future__ import annotations

from enum import StrEnum

from lang_rag_assistant.config.logging_config import get_logger

logger = get_logger(__name__)


class AppError(Exception):
    """Base class for errors the user can be shown a message for."""


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


def classify_provider_error(exc: BaseException, provider: str, model: str) -> ProviderError:
    """Map a provider exception to a ProviderError with the matching kind.

    401 and 403 are AUTH_FAILED. 429 is QUOTA_EXCEEDED when the text says
    quota or billing, otherwise RATE_LIMIT. Everything else (5xx, timeouts,
    connection failures, unknown errors) is UNAVAILABLE.
    """
    status = _status_code(exc)
    text = str(exc).lower()
    if status in (401, 403):
        kind = ProviderErrorKind.AUTH_FAILED
    elif status == 429:
        kind = (
            ProviderErrorKind.QUOTA_EXCEEDED
            if "quota" in text or "billing" in text
            else ProviderErrorKind.RATE_LIMIT
        )
    else:
        kind = ProviderErrorKind.UNAVAILABLE
    logger.error("Provider %s:%s failed (%s)", provider, model, kind.value, exc_info=exc)
    return ProviderError(kind, provider, model, exc)
