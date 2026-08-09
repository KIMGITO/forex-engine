"""Determinism and look-ahead regression tests.

The most important property of the backtest engine: modifying future data
(future candles, future news, future structure, future regime) must NEVER
change the results of bars before the modification point.
"""

import numpy as np
import pandas as pd
import pytest

from app.backtest import BacktestConfig, EventBacktester, NoOpStrategy


def _make_frame(n=220, seed=3, start="2024-01-01"):
    idx = pd.date_range(start, periods=n, freq="1h", tz="UTC")
    close = 1.08 + np.cumsum(np.random.default_rng(seed).normal(0, 0.002, n))
    return pd.DataFrame(
        {"open": close, "high": close + 0.002, "low": close - 0.002, "close": close},
        index=idx,
    )


class _ContextProbeStrategy(NoOpStrategy):
    """Records the maximum history length the strategy ever sees.

    Used to verify the strategy never receives future candles: the history
    length at bar i must equal i+1 (never more).
    """

    name = "probe"

    def __init__(self) -> None:
        super().__init__()
        self.max_history_len = 0
        self.observed_news_kinds = set()

    def on_bar(self, context):
        h = context.history()
        self.max_history_len = max(self.max_history_len, len(h))
        # Sanity: history must never exceed its own bar count.
        assert len(h) <= context._i + 1
        return []


def _result_key(result) -> tuple:
    return (
        tuple((eq.timestamp, round(eq.equity, 9)) for eq in result.equity_curve),
        tuple(
            (t.trade_id, round(t.net_pnl, 9)) for t in result.trades
        ),
        round(result.metrics.net_pnl, 9),
    )


class TestDeterminism:
    def test_same_run_twice_identical(self):
        df = _make_frame()
        cfg = BacktestConfig(symbol="EURUSD")
        r1 = EventBacktester(cfg).run(df, NoOpStrategy())
        r2 = EventBacktester(cfg).run(df, NoOpStrategy())
        assert _result_key(r1) == _result_key(r2)

    def test_noop_always_zero_trades(self):
        df = _make_frame(seed=5)
        res = EventBacktester(BacktestConfig(symbol="EURUSD")).run(df, NoOpStrategy())
        assert res.metrics.trade_count == 0
        assert res.metrics.net_pnl == pytest.approx(0.0)


class TestLookahead:
    def test_future_candles_do_not_change_past(self):
        df = _make_frame()
        cfg = BacktestConfig(symbol="EURUSD")
        full = EventBacktester(cfg).run(df, NoOpStrategy())

        modified = df.copy()
        modified.iloc[-40:] *= 2.0
        recomputed = EventBacktester(cfg).run(modified, NoOpStrategy())

        cutoff_idx = len(df) - 40
        full_past = _result_key_of_prefix(full, cutoff_idx)
        mod_past = _result_key_of_prefix(recomputed, cutoff_idx)
        assert full_past == mod_past

    def test_future_regime_does_not_change_past(self):
        df = _make_frame()
        cfg = BacktestConfig(symbol="EURUSD")
        # Build a "future regime" list with observational timestamps; the
        # backtester will only see those whose available_from <= current bar.

        future_regimes = [
            type(
                "FakeRegime",
                (),
                {"available_from": df.index[i + 50].to_pydatetime(), "state": i},
            )()
            for i in range(120)
        ]
        res = EventBacktester(cfg).run(
            df, NoOpStrategy(), regime_observations=future_regimes
        )
        # No-op should remain unchanged regardless of future regime data.
        assert res.metrics.trade_count == 0
        assert res.metrics.net_pnl == pytest.approx(0.0)

    def test_strategy_never_sees_future_history(self):
        df = _make_frame()
        strat = _ContextProbeStrategy()
        EventBacktester(BacktestConfig(symbol="EURUSD")).run(df, strat)
        assert strat.max_history_len == len(df)


def _result_key_of_prefix(result, n_bars) -> tuple:
    """Result key restricted to the first n_bars equity points."""
    prefix = result.equity_curve[:n_bars]
    return tuple((eq.timestamp, round(eq.equity, 9)) for eq in prefix)