"""Market structure & liquidity analysis engine.

Converts validated OHLC data into measurable market-structure information:
swings, structure, breaks, potential liquidity zones, sweeps, displacement, and
ranges. This is a research/analysis layer — it does **not** generate trading
signals, entries, or risk decisions.
"""

from app.market_structure.engine import MarketStructureConfig, MarketStructureEngine
from app.market_structure.errors import (
    DisplacementError,
    LiquidityError,
    MarketStructureError,
    RangeError,
    StructureError,
    SwingDetectionError,
)
from app.market_structure.models import (
    BreakEvent,
    BreakType,
    DisplacementClass,
    DisplacementEvent,
    LiquidityZone,
    MarketStructureResult,
    RangeEvent,
    StructurePoint,
    StructureType,
    SweepEvent,
    SweepType,
    Swing,
    SwingType,
)

__all__ = [
    "BreakEvent",
    "BreakType",
    "DisplacementClass",
    "DisplacementError",
    "DisplacementEvent",
    "LiquidityError",
    "LiquidityZone",
    "MarketStructureConfig",
    "MarketStructureEngine",
    "MarketStructureError",
    "MarketStructureResult",
    "RangeError",
    "RangeEvent",
    "StructureError",
    "StructurePoint",
    "StructureType",
    "SweepEvent",
    "SweepType",
    "Swing",
    "SwingDetectionError",
    "SwingType",
]