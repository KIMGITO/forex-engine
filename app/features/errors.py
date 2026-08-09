"""Custom domain exceptions for the quantitative feature engine."""


class FeatureError(Exception):
    """Base exception for all feature-engine errors."""


class UnknownFeatureError(FeatureError):
    """Raised when a requested feature name is not registered with the engine."""


class InsufficientDataError(FeatureError):
    """Raised when a feature cannot be computed due to insufficient lookback data."""