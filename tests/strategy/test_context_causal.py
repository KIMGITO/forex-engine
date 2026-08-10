"""Tests that StrategyContext never exposes future data."""

import numpy as np
import pandas as pd

from app.strategy.config import StrategyConfig
from app.strategy.context import StrategyContext


def _frame(n=100):
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    close = 1.08 + np.cumsum(np.random.default_rng(9).normal(0, 0.001, n))
    return pd.DataFrame(
        {"open": close, "high": close + 0.001, "low": close - 0.001, "close": close},
        index=idx,
    )


class TestContextCausal:
    def test_history_never_exceeds_current(self):
        df = _frame()
        for i in range(10, len(df), 10):
            ctx = StrategyContext(
                symbol="EURUSD",
                timeframe="1h",
                now=df.index[i],
                frame=df,
                current_index=i,
                config=StrategyConfig(),
            )
            h = ctx.history()
            assert len(h) == i + 1

    def test_limited_history(self):
        df = _frame()
        ctx = StrategyContext(
            symbol="EURUSD", timeframe="1h", now=df.index[50],
            frame=df, current_index=50, config=StrategyConfig(),
        )
        assert len(ctx.history(bars=5)) == 5

    def test_current_candle_is_current_row(self):
        df = _frame()
        ctx = StrategyContext(
            symbol="EURUSD", timeframe="1h", now=df.index[7],
            frame=df, current_index=7, config=StrategyConfig(),
        )
        assert ctx.current_candle()["close"] == df.iloc[7]["close"]
