"""Structured backtest report generation."""

from app.backtest.models import BacktestResult


def render_report(result: BacktestResult) -> str:
    """Render a human-readable backtest report.

    Simulated spread/slippage/commission are explicit assumptions because the
    underlying OHLC dataset has no historical bid/ask — stated in the report.
    """
    m = result.metadata
    metrics = result.metrics
    initial = float(result.config_dump.get("initial_balance", 0.0) or 0.0)
    final_equity = (
        result.equity_curve[-1].equity if result.equity_curve else initial
    )
    lines = [
        "BACKTEST",
        "──────────────",
        f"Symbol: {m.symbol}",
        f"Timeframe: {m.timeframe}",
        f"Period: {m.start.date()} → {m.end.date()}",
        f"Initial capital: {initial:,.2f} {result.config_dump.get('account_currency', '?')}",
        f"Final equity: {final_equity:,.2f}",
        f"Net P&L: {metrics.net_pnl:+,.6f}",
        f"Total return: {(metrics.total_return or 0.0):+.4%}",
        f"Max drawdown: {metrics.max_drawdown:.4%}",
        f"Drawdown duration (bars): {metrics.drawdown_duration_bars}",
        f"Trades: {metrics.trade_count}",
        f"Win rate: {(metrics.win_rate or 0.0):.2%}",
        f"Profit factor: {(metrics.profit_factor or 0.0):.4f}",
        f"Expectancy: {(metrics.expectancy or 0.0):+.6f}",
        f"Sharpe (per bar): {(metrics.sharpe or 0.0):.4f}",
        f"Sortino (per bar): {(metrics.sortino or 0.0):.4f}",
        "",
        (
            "NOTE: Simulated costs (spread/slippage/commission) are ASSUMPTIONS — "
            "the underlying OHLC dataset contains no historical bid/ask. "
            "These numbers describe the simulation engine, NOT real execution."
        ),
    ]
    if metrics.insufficient_data:
        lines.append("")
        lines.append("Data limitations:")
        lines.extend(f"  - {note}" for note in metrics.insufficient_data)
    return "\n".join(lines)
