"""Standalone equity/drawdown/benchmark utilities.

The engine computes ``PerformanceMetrics`` during a run; this module provides
reusable curves and the BUY_AND_HOLD benchmark comparison for reporting.
"""

from collections.abc import Sequence

from app.backtest.models import EquityPoint

__all__ = ["buy_and_hold_curve", "drawdown_curve", "drawdown_pct"]


def drawdown_pct(equity_curve: list[EquityPoint]) -> list[float]:
    """Per-point drawdown as a fraction of the running peak (0..1)."""
    out: list[float] = []
    peak = -1.0
    for pt in equity_curve:
        peak = max(peak, pt.equity)
        dd = (peak - pt.equity) / peak if peak > 0 else 0.0
        out.append(max(dd, 0.0))
    return out


def drawdown_curve(equity_curve: list[EquityPoint]) -> list[EquityPoint]:
    """Return an equity-style curve of drawdown levels per timestamp."""
    dd = drawdown_pct(equity_curve)
    out: list[EquityPoint] = []
    for pt, d in zip(equity_curve, dd):
        out.append(
            EquityPoint(
                timestamp=pt.timestamp,
                equity=pt.equity * (1.0 - d),
                balance=pt.balance,
                unrealized_pnl=pt.unrealized_pnl,
            )
        )
    return out


def buy_and_hold_curve(
    equity_curve: list[EquityPoint],
    initial_balance: float,
    closes: Sequence[float],
) -> list[EquityPoint]:
    """Buy-and-hold benchmark equity per bar: initial * close[i] / close[0].

    ``closes`` must be aligned 1:1 with ``equity_curve`` timestamps.

    Documented: buy-and-hold is NOT an appropriate benchmark for every Forex
    strategy; it is provided for context only.
    """
    if not equity_curve:
        return []
    if len(closes) != len(equity_curve):
        raise ValueError("closes length must match equity_curve length")
    first = closes[0]
    if first <= 0:
        raise ValueError("first close must be > 0")

    out: list[EquityPoint] = []
    for pt, close in zip(equity_curve, closes):
        eq = initial_balance * (close / first)
        out.append(
            EquityPoint(
                timestamp=pt.timestamp,
                equity=eq,
                balance=pt.balance,
                unrealized_pnl=0.0,
            )
        )
    return out