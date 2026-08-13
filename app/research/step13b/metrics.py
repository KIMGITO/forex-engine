"""Extended metric calculations for Step 13B.

Beyond the core backtest metrics, Step 13B computes grouped/segmented metrics:
* monthly returns
* yearly returns
* per-regime performance
* per-symbol performance
* per-timeframe performance
* per-session performance
* consecutive win/loss streaks
* average drawdown
* average trades/month
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.research.step13b.models import WindowMetrics


def _consecutive_streaks(results: list[str]) -> tuple[int, int]:
    """Return (max_consecutive_losses, max_consecutive_wins)."""
    max_loss = 0
    max_win = 0
    cur_loss = 0
    cur_win = 0
    for r in results:
        if r == "win":
            cur_win += 1
            cur_loss = 0
        elif r == "loss":
            cur_loss += 1
            cur_win = 0
        else:
            cur_loss = 0
            cur_win = 0
        max_win = max(max_win, cur_win)
        max_loss = max(max_loss, cur_loss)
    return max_loss, max_win


def _r_multiple_from_trade(trade: Any, entry: float, stop_loss: float) -> float:
    """Compute R multiple for a backtest Trade."""
    risk_dist = abs(entry - stop_loss)
    if risk_dist <= 0:
        return 0.0
    return trade.net_pnl / (risk_dist * trade.quantity)


def compute_window_metrics(
    *,
    window_index: int,
    phase: str,
    symbol: str,
    timeframe: str,
    param_set: str,
    trades: list[Any],
    equity_curve: list[Any],
    bars: int,
    risk_metrics: dict[str, int] | None = None,
) -> WindowMetrics:
    """Compute comprehensive metrics from backtest result components.

    Parameters
    ----------
    trades : list[Trade]
        Completed backtest trades.
    equity_curve : list[EquityPoint]
        Equity observations over the phase.
    bars : int
        Number of bars in the phase.
    risk_metrics : optional dict
        Risk engine counters (approved/rejected/limitations).
    """
    n = len(trades)

    net_pnl = sum(t.net_pnl for t in trades) if trades else 0.0
    net_return = (
        (equity_curve[-1].equity - equity_curve[0].equity) / equity_curve[0].equity
        if equity_curve and len(equity_curve) >= 2 and equity_curve[0].equity > 0
        else 0.0
    )

    wins = [t for t in trades if t.net_pnl > 0]
    losses = [t for t in trades if t.net_pnl < 0]
    breakeven = [t for t in trades if t.net_pnl == 0]

    win_rate = len(wins) / n if n else None
    avg_win = sum(t.net_pnl for t in wins) / len(wins) if wins else None
    avg_loss = sum(t.net_pnl for t in losses) / len(losses) if losses else None

    gross_profit = sum(t.net_pnl for t in wins)
    gross_loss = abs(sum(t.net_pnl for t in losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else None
    expectancy = (net_pnl / n) if n else None

    # Expectancy in R (risk-normalized)
    r_values: list[float] = []
    for t in trades:
        risk_dist = abs(t.entry_price - t.stop_loss) if t.stop_loss else 0.0
        if risk_dist > 0:
            r_values.append(t.net_pnl / (risk_dist * t.quantity))
    expectancy_r = sum(r_values) / len(r_values) if r_values else None

    # Drawdown from equity curve
    max_dd, dd_values = _drawdown_stats(equity_curve)
    avg_dd = float(np.mean(dd_values)) if dd_values else 0.0

    # Sharpe / Sortino from per-bar returns
    sharpe, sortino = _sharpe_sortino(equity_curve)

    # Streaks
    results = []
    for t in trades:
        if t.net_pnl > 0:
            results.append("win")
        elif t.net_pnl < 0:
            results.append("loss")
        else:
            results.append("breakeven")
    max_losses, max_wins = _consecutive_streaks(results)

    # Trades per month
    if equity_curve and len(equity_curve) >= 2:
        start_ts = equity_curve[0].timestamp
        end_ts = equity_curve[-1].timestamp
        months = max((end_ts - start_ts).total_seconds() / (30.44 * 86400.0), 1e-9)
        avg_trades_month = n / months if months > 0 else 0.0
    else:
        avg_trades_month = 0.0

    rm = risk_metrics or {}
    return WindowMetrics(
        window_index=window_index,
        phase=phase,
        symbol=symbol,
        timeframe=timeframe,
        param_set=param_set,
        net_return=float(net_return),
        net_profit=float(net_pnl),
        profit_factor=float(profit_factor) if profit_factor is not None else None,
        expectancy=float(expectancy) if expectancy is not None else None,
        expectancy_r=float(expectancy_r) if expectancy_r is not None else None,
        win_rate=float(win_rate) if win_rate is not None else None,
        average_win=float(avg_win) if avg_win is not None else None,
        average_loss=float(avg_loss) if avg_loss is not None else None,
        max_drawdown=float(max_dd),
        average_drawdown=float(avg_dd),
        sharpe=float(sharpe) if sharpe is not None else None,
        sortino=float(sortino) if sortino is not None else None,
        trade_count=n,
        average_trades_per_month=float(avg_trades_month),
        max_consecutive_losses=int(max_losses),
        max_consecutive_wins=int(max_wins),
        bars_in_phase=int(bars),
        risk_rejected_count=int(rm.get("risk_rejected_count", 0)),
        risk_approved_count=int(rm.get("risk_approved_count", 0)),
        daily_loss_breaches=int(rm.get("daily_loss_breaches", 0)),
        drawdown_limit_breaches=int(rm.get("drawdown_limit_breaches", 0)),
        position_limit_breaches=int(rm.get("position_limit_breaches", 0)),
        exposure_breaches=int(rm.get("exposure_breaches", 0)),
    )


def _drawdown_stats(equity_curve: list[Any]) -> tuple[float, list[float]]:
    """Compute max drawdown and per-point drawdown values."""
    if not equity_curve or len(equity_curve) < 2:
        return 0.0, []
    peak = -1.0
    max_dd = 0.0
    dd_values: list[float] = []
    for pt in equity_curve:
        peak = max(peak, pt.equity)
        dd = (peak - pt.equity) / peak if peak > 0 else 0.0
        dd_values.append(max(dd, 0.0))
        max_dd = max(max_dd, dd)
    return max_dd, dd_values


def _sharpe_sortino(equity_curve: list[Any]) -> tuple[float | None, float | None]:
    """Compute per-bar Sharpe and Sortino ratios."""
    if len(equity_curve) < 3:
        return None, None
    returns = []
    for a, b in zip(equity_curve[:-1], equity_curve[1:]):
        if a.equity > 0:
            returns.append((b.equity - a.equity) / a.equity)
    if len(returns) < 3:
        return None, None
    mean_r = float(np.mean(returns))
    std = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
    sharpe = mean_r / std * (len(returns) ** 0.5) if std > 0 else None

    downside = [r for r in returns if r < 0]
    if len(downside) < 2:
        sortino = None
    else:
        dstd = float(np.std(downside, ddof=1))
        sortino = mean_r / dstd * (len(returns) ** 0.5) if dstd > 0 else None
    return sharpe, sortino


def monthly_returns(
    trades: list[Any], symbol: str, timeframe: str
) -> pd.DataFrame:
    """Aggregate realized P&L by calendar month."""
    if not trades:
        return pd.DataFrame(
            columns=["month", "symbol", "timeframe", "net_pnl", "trade_count"]
        )
    rows = []
    for t in trades:
        month = t.exit_time.strftime("%Y-%m")
        rows.append(
            {
                "month": month,
                "symbol": symbol,
                "timeframe": timeframe,
                "net_pnl": t.net_pnl,
                "trade_count": 1,
            }
        )
    df = pd.DataFrame(rows)
    grouped = (
        df.groupby(["month", "symbol", "timeframe"])
        .agg({"net_pnl": "sum", "trade_count": "sum"})
        .reset_index()
    )
    return grouped.sort_values("month")


def yearly_returns(
    trades: list[Any], symbol: str, timeframe: str
) -> pd.DataFrame:
    """Aggregate realized P&L by calendar year."""
    if not trades:
        return pd.DataFrame(
            columns=["year", "symbol", "timeframe", "net_pnl", "trade_count"]
        )
    rows = []
    for t in trades:
        year = t.exit_time.strftime("%Y")
        rows.append(
            {
                "year": year,
                "symbol": symbol,
                "timeframe": timeframe,
                "net_pnl": t.net_pnl,
                "trade_count": 1,
            }
        )
    df = pd.DataFrame(rows)
    grouped = (
        df.groupby(["year", "symbol", "timeframe"])
        .agg({"net_pnl": "sum", "trade_count": "sum"})
        .reset_index()
    )
    return grouped.sort_values("year")


def regime_performance(trade_log: pd.DataFrame) -> pd.DataFrame:
    """Per-regime performance from the compact trade log.

    The trade log stores one row per signal with ``result`` and ``r_multiple``.
    Regime performance groups by ``regime`` (from the causal snapshot).
    """
    if trade_log is None or trade_log.empty or "result" not in trade_log.columns:
        return pd.DataFrame(
            columns=["regime", "trade_count", "wins", "losses", "net_r", "win_rate"]
        )
    df = trade_log.copy()
    df["is_win"] = df["result"] == "win"
    df["is_loss"] = df["result"] == "loss"
    grouped = (
        df.groupby("regime")
        .agg(
            trade_count=("result", "count"),
            wins=("is_win", "sum"),
            losses=("is_loss", "sum"),
            net_r=("r_multiple", "sum"),
        )
        .reset_index()
    )
    grouped["win_rate"] = grouped["wins"] / grouped["trade_count"].clip(lower=1)
    return grouped.sort_values("trade_count", ascending=False)


def session_performance(trade_log: pd.DataFrame) -> pd.DataFrame:
    """Per-session performance from the compact trade log."""
    if trade_log is None or trade_log.empty or "result" not in trade_log.columns:
        return pd.DataFrame(
            columns=["session", "trade_count", "wins", "losses", "net_r", "win_rate"]
        )
    df = trade_log.copy()
    df["is_win"] = df["result"] == "win"
    df["is_loss"] = df["result"] == "loss"
    grouped = (
        df.groupby("session")
        .agg(
            trade_count=("result", "count"),
            wins=("is_win", "sum"),
            losses=("is_loss", "sum"),
            net_r=("r_multiple", "sum"),
        )
        .reset_index()
    )
    grouped["win_rate"] = grouped["wins"] / grouped["trade_count"].clip(lower=1)
    return grouped.sort_values("trade_count", ascending=False)


def symbol_performance(trade_log: pd.DataFrame) -> pd.DataFrame:
    """Per-symbol performance from the compact trade log."""
    if trade_log is None or trade_log.empty or "result" not in trade_log.columns:
        return pd.DataFrame(
            columns=["symbol", "trade_count", "wins", "losses", "net_r", "win_rate"]
        )
    df = trade_log.copy()
    df["is_win"] = df["result"] == "win"
    df["is_loss"] = df["result"] == "loss"
    grouped = (
        df.groupby("symbol")
        .agg(
            trade_count=("result", "count"),
            wins=("is_win", "sum"),
            losses=("is_loss", "sum"),
            net_r=("r_multiple", "sum"),
        )
        .reset_index()
    )
    grouped["win_rate"] = grouped["wins"] / grouped["trade_count"].clip(lower=1)
    return grouped.sort_values("trade_count", ascending=False)