"""Strategy, signal, and research engine.

Signals are RESEARCH INFORMATION, not orders. This layer is consumable by the
backtester (Step 8), future paper-trading simulation, and a future API service
(FastAPI/Supabase/React) without redesign.
"""

from app.strategy.base import SignalToOrderAdapter, Strategy
from app.strategy.config import StrategyConfig
from app.strategy.context import StrategyContext
from app.strategy.engine import (
    HistoricalSignalScanner,
    SignalScanResult,
    StrategyComparison,
)
from app.strategy.models import (
    Setup,
    Signal,
    SignalDirection,
    SignalReason,
    SignalStatus,
    SignalStrength,
)
from app.strategy.strategies import LiquidityReversalStrategy, TrendStructureStrategy

__all__ = [
    "HistoricalSignalScanner",
    "LiquidityReversalStrategy",
    "Setup",
    "Signal",
    "SignalDirection",
    "SignalReason",
    "SignalScanResult",
    "SignalStatus",
    "SignalStrength",
    "SignalToOrderAdapter",
    "Strategy",
    "StrategyComparison",
    "StrategyConfig",
    "StrategyContext",
    "TrendStructureStrategy",
]
