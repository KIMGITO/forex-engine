"""Event-driven backtest engine.

CORE INVARIANT: the backtest processes bars strictly in chronological order
and exposes to the strategy ONLY information available at the current bar.

A strategy receives a restricted :class:`BacktestContext`:
- the current candle (from the indexable frame, never a future row)
- historical candles ``up to`` the current bar (``history(t)`` is capped at
  the current index)
- trailing features computed on the full (causal) series, sliced at the
  current bar
- market-structure events filtered by ``available_from <= now``
- news context similarly filtered by availability
- regime observations filtered by ``available_from <= now``
- current portfolio values

It is architecturally impossible for the strategy to access future candles
through the public API: the context holds only ``.iloc[:i+1]`` views and
timestamp-filtered event lists.
"""

from itertools import pairwise
from typing import Any

import pandas as pd

from app.backtest.clock import BacktestClock
from app.backtest.config import BacktestConfig
from app.backtest.costs import (
    CommissionModel,
    FixedPerTradeCommissionModel,
    FixedSlippageModel,
    FixedSpreadModel,
    NoSwapModel,
)
from app.backtest.execution import ExecutionSimulator
from app.backtest.models import (
    BacktestConfigMeta,
    BacktestResult,
    EquityPoint,
    Order,
    OrderIntent,
    OrderSide,
    PerformanceMetrics,
    Position,
)
from app.backtest.orders import fill_order, intent_to_order
from app.backtest.portfolio import Portfolio
from app.market_structure.models import MarketStructureResult

__all__ = ["BacktestContext", "EventBacktester", "NoOpStrategy", "Strategy"]


def _normalize_backtest_ts(value) -> Any:
    """Normalize a (possibly pandas) timestamp to a hashable UTC key."""
    if hasattr(value, "to_pydatetime"):
        return value.to_pydatetime()
    return value


class Strategy:
    """Abstract strategy interface.

    A strategy receives a BacktestContext and returns a list of OrderIntent.
    It must not attempt to access future data; the context does not expose it.
    """

    name: str = "abstract"

    def __init__(self) -> None:
        self.pending_orders: list[Order] = []

    def on_bar(self, context: "BacktestContext") -> list[OrderIntent]:
        raise NotImplementedError


class NoOpStrategy(Strategy):
    """Trivial strategy that never trades (deterministic baseline)."""

    name = "noop"

    def on_bar(self, context: "BacktestContext") -> list[OrderIntent]:
        return []


class BacktestContext:
    """Restricted, causal view of the market for a strategy.

    Only current and past data are reachable. The context is constructed by
    the engine at each bar and is not intended to be constructed by strategies.
    """

    def __init__(
        self,
        symbol: str,
        timeframe: str,
        clock: BacktestClock,
        frame: pd.DataFrame,
        current_index: int,
        features: pd.DataFrame | None = None,
        structure: MarketStructureResult | None = None,
        news_events: list[Any] | None = None,
        regime_observations: list[Any] | None = None,
        portfolio: Portfolio | None = None,
        now: pd.Timestamp | None = None,
        mtf: Any | None = None,
    ) -> None:
        self.symbol = symbol
        self.timeframe = timeframe
        self._clock = clock
        self._frame = frame
        self._i = current_index
        self._features = features
        self._structure = structure
        self._news_events = news_events
        self._regime_observations = regime_observations
        self._portfolio = portfolio
        self._mtf = mtf
        self.now = now if now is not None else frame.index[current_index]

    # ── current state (only this bar) ────────────────────────────────────────

    def current_candle(self) -> pd.Series:
        """Current bar only. Never a future bar."""
        return self._frame.iloc[self._i]

    def current_features(self) -> pd.Series | None:
        if self._features is None:
            return None
        return self._features.iloc[self._i]

    # ── historical state (strictly <= current) ───────────────────────────────

    def history(self, bars: int | None = None) -> pd.DataFrame:
        """Return historical + current candles, capped at the current bar.

        ``bars`` limits the lookback window; default returns everything from
        start through the current bar. Never includes future rows.
        """
        if bars is None:
            return self._frame.iloc[: self._i + 1]
        return self._frame.iloc[max(0, self._i + 1 - bars) : self._i + 1]

    def features_history(self) -> pd.DataFrame | None:
        if self._features is None:
            return None
        return self._features.iloc[: self._i + 1]

    # ── market structure (available up to now) ───────────────────────────────

    def structure_events(self) -> list[Any]:
        """Structure events with available_from <= now (or all if none set)."""
        if self._structure is None:
            return []
        out: list[Any] = []
        for p in self._structure.structure:
            if p.available_from is None or p.available_from <= self.now:
                out.append(p)
        for r in self._structure.ranges:
            if r.available_from is None or r.available_from <= self.now:
                out.append(r)
        return out

    def news_available(self) -> list[Any]:
        """News/economic events available at or before now."""
        if not self._news_events:
            return []
        return [
            e
            for e in self._news_events
            if getattr(e, "available_from", None) is None
            or e.available_from <= self.now
        ]

    def regime_available(self) -> list[Any]:
        """Regime observations with available_from <= now."""
        if not self._regime_observations:
            return []
        return [
            r for r in self._regime_observations if r.available_from <= self.now
        ]

    # ── portfolio (current values only) ──────────────────────────────────────

    def mtf_context(self):
        """Return the causal MTF context for this bar, or None when MTF is not
        enabled/provided. The MTF context carries its own ``available_from``
        invariant: a strategy may only ever act on MTF tiers whose candle was
        fully completed before this bar."""
        return self._mtf

    def equity(self, mid: float) -> float:
        assert self._portfolio is not None
        return self._portfolio.equity(mid)

    def free_margin(self, mid: float) -> float:
        assert self._portfolio is not None
        return self._portfolio.free_margin(mid)

    def open_positions(self) -> list[Position]:
        assert self._portfolio is not None
        return list(self._portfolio.positions.values())


class EventBacktester:
    """Deterministic, causal backtest orchestrator."""

    def __init__(self, config: BacktestConfig) -> None:
        self.config = config

    def run(
        self,
        frame: pd.DataFrame,
        strategy: Strategy,
        *,
        features: pd.DataFrame | None = None,
        market_structure: MarketStructureResult | None = None,
        news_events: list[Any] | None = None,
        regime_observations: list[Any] | None = None,
        mtf_contexts: list | None = None,
        provider: str = "unknown",
        source_type: str = "historical",
    ) -> BacktestResult:
        """Run a backtest over ``frame`` (tz-aware, sorted).

        Parameters are intentionally typed as the causal inputs the engine
        accepts; the strategy can only ever view slices capped at the current
        bar via BacktestContext.

        ``mtf_contexts`` (optional) provides one MtfContext per bar (ordered
        same as the frame index). When provided, each per-bar BacktestContext
        exposes ``mtf_context()`` with only the MTF tiers whose ``available_from``
        is at or before that bar. When omitted, the engine behaves exactly as
        before (no MTF wiring) — fully backward compatible.
        """
        if frame.empty or "close" not in frame.columns:
            raise ValueError("frame must be a non-empty OHLC DataFrame")

        sorted_frame = frame.sort_index()
        clock = BacktestClock(list(sorted_frame.index))

        # Slice to configured [start, end] if provided.
        if self.config.start is not None:
            sorted_frame = sorted_frame.loc[self.config.start :]
        if self.config.end is not None:
            sorted_frame = sorted_frame.loc[: self.config.end]
        clock = BacktestClock(list(sorted_frame.index))
        if len(clock) == 0:
            raise ValueError("No bars in configured time range")

        pip = self.config.pip_size
        from app.backtest.costs import pip_size_for_symbol

        pip = pip or pip_size_for_symbol(self.config.symbol)

        spread_model = FixedSpreadModel(self.config.spread_pips, pip)
        slippage_model = FixedSlippageModel(self.config.slippage_pips, pip)
        commission_model: CommissionModel
        if self.config.commission_model == "zero":
            from app.backtest.costs import ZeroCommissionModel

            commission_model = ZeroCommissionModel()
        elif self.config.commission_model == "fixed":
            commission_model = FixedPerTradeCommissionModel(
                self.config.commission_per_trade
            )
        elif self.config.commission_model == "percentage":
            from app.backtest.costs import PercentageCommissionModel

            commission_model = PercentageCommissionModel(self.config.commission_percent)
        else:
            raise ValueError(f"Unknown commission model: {self.config.commission_model}")

        swap_model = NoSwapModel()
        execution = ExecutionSimulator(
            spread_model, slippage_model, self.config.fill_policy
        )
        portfolio = Portfolio(
            initial_balance=self.config.initial_balance,
            account_currency=self.config.account_currency,
            leverage=self.config.leverage,
            max_position_size=self.config.max_position_size,
        )

        equity_curve: list[EquityPoint] = []
        trades: list = []
        portfolio_states: list = []
        all_orders: list[Order] = []

        # Map MTF contexts by observation timestamp for causal per-bar lookup.
        mtf_by_ts = None
        if mtf_contexts is not None:
            mtf_by_ts = {
                _normalize_backtest_ts(mtf.timestamp): mtf for mtf in mtf_contexts
            }

        for i, ts in clock:
            bar = sorted_frame.iloc[i]
            mid = float(bar["close"])

            context = BacktestContext(
                symbol=self.config.symbol,
                timeframe=self.config.timeframe,
                clock=clock,
                frame=sorted_frame,
                current_index=i,
                features=features,
                structure=market_structure,
                news_events=news_events,
                regime_observations=regime_observations,
                portfolio=portfolio,
                now=ts,
                mtf=(
                    mtf_by_ts.get(_normalize_backtest_ts(ts))
                    if mtf_by_ts is not None
                    else None
                ),
            )

            # 1. Strategy decisions (causal context only).
            intents = strategy.on_bar(context)

            # 2. Convert intents to orders + execute.
            for intent in intents:
                order = intent_to_order(intent)
                all_orders.append(order)
                fill_result = execution.evaluate_entry(
                    order,
                    bar_mid=mid,
                    bar_open=float(bar["open"]),
                    bar_high=float(bar["high"]),
                    bar_low=float(bar["low"]),
                )
                if not fill_result.filled or fill_result.price is None:
                    continue
                filled, fill = fill_order(
                    order,
                    price=fill_result.price,
                    filled_at=ts,
                    slippage_applied=fill_result.slippage_applied,
                    gross_value=fill_result.price * order.quantity,
                )
                all_orders[-1] = filled
                # Position management
                portfolio.open_position(
                    fill,
                    stop_loss=intent.stop_loss,
                    take_profit=intent.take_profit,
                )
                notional = fill.price * order.quantity
                commission = commission_model.commission(notional, order.quantity)
                if commission:
                    portfolio.apply_commission(commission)

            # 3. SL/TP resolution for open positions (causal, same bar).
            for symbol, pos in list(portfolio.positions.items()):
                result = execution.resolve_stop_take_profit(
                    pos,
                    bar_mid=mid,
                    bar_open=float(bar["open"]),
                    bar_high=float(bar["high"]),
                    bar_low=float(bar["low"]),
                )
                if result is not None and result.price is not None:
                    trade = portfolio.close_position_via_sl_tp(
                        symbol, result.price, ts
                    )
                    if trade:
                        trades.append(trade)
                        notional = pos.quantity * result.price
                        commission = commission_model.commission(
                            notional, pos.quantity
                        )
                        if commission:
                            portfolio.apply_commission(commission)

            # 4. Daily/bar swap financing (NoSwap baseline).
            financing = swap_model.financing(
                sum(p.quantity * p.average_entry for p in portfolio.positions.values()),
                holding_bars=1,
            )
            if financing:
                portfolio.apply_financing(financing)

            # 5. Increment holding bars.
            for sym, pos in list(portfolio.positions.items()):
                portfolio.positions[sym] = pos.model_copy(
                    update={"holding_bars": pos.holding_bars + 1}
                )

            # 6. Record equity point + portfolio snapshot.
            eq = portfolio.equity(mid)
            equity_curve.append(
                EquityPoint(
                    timestamp=ts,
                    equity=eq,
                    balance=portfolio.balance,
                    unrealized_pnl=portfolio.unrealized_pnl(mid),
                )
            )
            portfolio_states.append(portfolio.snapshot(ts, mid))

        # 7. Final mark-to-market: close any remaining positions at last close.
        last_ts = sorted_frame.index[-1]
        last_mid = float(sorted_frame.iloc[-1]["close"])
        for symbol, pos in list(portfolio.positions.items()):
            final_fill_px = execution.spread.bid_ask(last_mid)[0] if pos.side == OrderSide.SELL else execution.spread.bid_ask(last_mid)[1]
            trade = portfolio.close_position_via_sl_tp(symbol, final_fill_px, last_ts)
            if trade:
                trades.append(trade)

        metrics = self._compute_metrics(portfolio, equity_curve, trades, portfolio_states)

        return BacktestResult(
            metadata=BacktestConfigMeta(
                provider=provider,
                symbol=self.config.symbol,
                timeframe=self.config.timeframe,
                start=clock.start.to_pydatetime(),
                end=clock.end.to_pydatetime(),
                source_type=source_type,
                strategy=strategy.name,
            ),
            equity_curve=equity_curve,
            trades=portfolio.trades,
            portfolio_states=portfolio_states,
            metrics=metrics,
            config_dump=self.config.to_dict(),
        )

    def _compute_metrics(
        self,
        portfolio: Portfolio,
        equity_curve: list[EquityPoint],
        trades: list,
        portfolio_states: list,
    ) -> PerformanceMetrics:
        """Compute research metrics from the completed run."""
        if not equity_curve:
            return PerformanceMetrics(trade_count=len(trades), insufficient_data=["no equity curve"])

        initial = portfolio.initial_balance
        final_equity = equity_curve[-1].equity
        net = final_equity - initial
        total_return = net / initial if initial else None

        gross_profit = sum(max(t.net_pnl, 0.0) for t in portfolio.trades)
        gross_loss = abs(sum(min(t.net_pnl, 0.0) for t in portfolio.trades))

        win_trades = [t for t in portfolio.trades if t.net_pnl > 0]
        loss_trades = [t for t in portfolio.trades if t.net_pnl < 0]
        n = len(portfolio.trades)

        win_rate = len(win_trades) / n if n else None
        loss_rate = len(loss_trades) / n if n else None
        avg_win = sum(t.net_pnl for t in win_trades) / len(win_trades) if win_trades else None
        avg_loss = sum(t.net_pnl for t in loss_trades) / len(loss_trades) if loss_trades else None
        profit_factor = gross_profit / gross_loss if gross_loss else None
        expectancy = (sum(t.net_pnl for t in portfolio.trades) / n) if n else None

        # Drawdown.
        peak = -1.0
        max_dd = 0.0
        current_dd_start: int | None = None
        max_dd_duration = 0
        for idx, pt in enumerate(equity_curve):
            if pt.equity > peak:
                peak = pt.equity
                current_dd_start = None
            else:
                dd = (peak - pt.equity) / peak if peak > 0 else 0.0
                if dd > max_dd:
                    max_dd = dd
                    if current_dd_start is None:
                        current_dd_start = idx
            if current_dd_start is not None:
                max_dd_duration = max(max_dd_duration, idx - current_dd_start + 1)

        # Returns for Sharpe/Sortino (per bar).
        returns = []
        for a, b in pairwise(equity_curve):
            if a.equity == 0:
                continue
            returns.append((b.equity - a.equity) / a.equity)
        sharpe = None
        sortino = None
        if len(returns) >= 3:
            mean_r = sum(returns) / len(returns)
            var = sum((r - mean_r) ** 2 for r in returns) / (len(returns) - 1)
            std = var ** 0.5
            if std > 0:
                sharpe = mean_r / std * (len(returns) ** 0.5)
            downside = [r for r in returns if r < 0]
            if downside and len(downside) >= 2:
                # Sample variance requires >= 2 observations; with a single
                # downside return (or none) Sortino is reported as None
                # (insufficient data), never via a division by zero.
                dvari = sum(r * r for r in downside) / (len(downside) - 1)
                dstd = dvari ** 0.5
                if dstd > 0:
                    sortino = mean_r / dstd * (len(returns) ** 0.5)

        exposure_fraction = (
            sum(1 for s in portfolio_states if s.open_positions) / len(portfolio_states)
            if portfolio_states
            else 0.0
        )
        avg_holding = (
            sum(t.holding_bars for t in portfolio.trades) / len(portfolio.trades)
            if portfolio.trades
            else None
        )

        insufficient = []
        if n < 2:
            insufficient.append("fewer than 2 trades: win/loss rates, profit factor, expectancy are unreliable")
        if len(returns) < 3:
            insufficient.append("fewer than 3 equity observations: Sharpe/Sortino omitted")

        return PerformanceMetrics(
            total_return=total_return,
            net_pnl=net,
            gross_profit=gross_profit,
            gross_loss=gross_loss,
            win_rate=win_rate,
            loss_rate=loss_rate,
            profit_factor=profit_factor,
            average_win=avg_win,
            average_loss=avg_loss,
            expectancy=expectancy,
            max_drawdown=max_dd,
            drawdown_duration_bars=max_dd_duration,
            sharpe=sharpe,
            sortino=sortino,
            trade_count=n,
            exposure_fraction=exposure_fraction,
            average_holding_bars=avg_holding,
            insufficient_data=insufficient,
        )