"""Custom domain exceptions for the market structure & liquidity engine."""


class MarketStructureError(Exception):
    """Base exception for all market-structure engine errors."""


class SwingDetectionError(MarketStructureError):
    """Raised when swing detection fails due to invalid inputs or parameters."""


class StructureError(MarketStructureError):
    """Raised when structural analysis encounters invalid inputs."""


class LiquidityError(MarketStructureError):
    """Raised when liquidity-zone or sweep detection fails."""


class DisplacementError(MarketStructureError):
    """Raised when displacement computation fails."""


class RangeError(MarketStructureError):
    """Raised when range/consolidation detection fails."""