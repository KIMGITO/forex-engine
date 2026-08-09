"""Custom domain exceptions for market data operations."""


class MarketDataError(Exception):
    """Base exception for all market data errors."""


class ValidationError(MarketDataError):
    """Raised when market data fails structural or financial validation rules."""


class ProviderError(MarketDataError):
    """Raised when data retrieval or adapter operations fail."""


class StorageError(MarketDataError):
    """Raised when repository read/write operations fail."""


# ── Provider-specific failure modes (all under ProviderError) ────────────────

class AuthenticationError(ProviderError):
    """Raised when provider authentication fails (401)."""


class RateLimitError(ProviderError):
    """Raised when the provider enforces a rate limit (429)."""


class NetworkError(ProviderError):
    """Raised on transport/network-level failures."""


class MalformedResponseError(ProviderError):
    """Raised when the provider returns an unparseable/inconsistent response."""


class UnavailableSymbolError(ProviderError):
    """Raised when the requested symbol is not offered by the provider."""


class UnavailableTimeframeError(ProviderError):
    """Raised when the requested timeframe is not offered by the provider."""


class ProviderServerError(ProviderError):
    """Raised on provider 5xx server errors."""


class RetryExhaustedError(ProviderError):
    """Raised when retries are exhausted without a successful response."""
