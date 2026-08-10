"""Domain exceptions for the multi-timeframe engine."""


class MtfError(Exception):
    """Base exception for all MTF engine errors."""


class MtfAlignmentError(MtfError):
    """Raised when timeframe alignment invariants are violated."""


class MtfConfigurationError(MtfError):
    """Raised when MTF configuration is invalid."""