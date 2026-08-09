"""Custom domain exceptions for market data operations."""


class MarketDataError(Exception):
    """Base exception for all market data errors."""


class ValidationError(MarketDataError):
    """Raised when market data fails structural or financial validation rules."""


class ProviderError(MarketDataError):
    """Raised when data retrieval or adapter operations fail."""


class StorageError(MarketDataError):
    """Raised when repository read/write operations fail."""