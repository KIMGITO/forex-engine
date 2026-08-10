"""Domain exceptions for the strategy and signal layer."""


class StrategyError(Exception):
    """Base exception for all strategy layer errors."""


class SignalValidationError(StrategyError):
    """Raised when a signal violates structural rules."""


class StrategyConfigurationError(StrategyError):
    """Raised when a strategy is configured with invalid parameters."""


class StrategyExecutionError(StrategyError):
    """Raised when a strategy evaluation fails."""