"""Research-grade event-driven backtesting engine.

Deterministic, causal, historical simulation. This is NOT a trading strategy
and does NOT execute trades; it provides the environment for future strategies.
"""

from app.backtest.config import BacktestConfig
from app.backtest.engine import EventBacktester, NoOpStrategy, Strategy
from app.backtest.models import (
    BacktestResult,
    FillPolicy,
    OrderIntent,
    OrderSide,
    OrderType,
    PerformanceMetrics,
)

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "EventBacktester",
    "FillPolicy",
    "NoOpStrategy",
    "OrderIntent",
    "OrderSide",
    "OrderType",
    "PerformanceMetrics",
    "Strategy",
]
