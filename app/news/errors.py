"""Custom domain exceptions for the news & economic-events layer."""


class NewsError(Exception):
    """Base exception for all news/economic-events errors."""


class ProviderError(NewsError):
    """Raised when an economic-calendar provider fails."""


class NormalizationError(NewsError):
    """Raised when raw provider data cannot be normalized."""


class ValidationError(NewsError):
    """Raised when economic-event data fails validation."""


class StorageError(NewsError):
    """Raised when the event repository read/write operations fail."""


class CalendarError(NewsError):
    """Raised when calendar queries receive invalid inputs."""
