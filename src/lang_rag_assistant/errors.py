"""Application exception types shared across layers."""


class AppError(Exception):
    """Base class for errors the user can be shown a message for."""


class ConfigurationError(AppError):
    """A required setting is missing or has an invalid value."""


class DocumentError(AppError):
    """An uploaded or bundled document cannot be read or indexed."""
