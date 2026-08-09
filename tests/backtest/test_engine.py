"""Tests for the backtest engine mechanics."""

import numpy as np
import pandas as pd
import pytest

from app.backtest import BacktestConfig, EventBacktester, NoOpStrategy
from app.backtest.models import OrderIntent, OrderSide


def _make_frame(n=120, seed=42, start="2024-01-01"):
    idx = pd.date_range(start, periods=n, freq="1h", tz="UTC")
    close = 1.0800 + np.cumsum(np.random.default_rng(seed).normal(0, 0.001, n))
    return pd.DataFrame(
        {"open": close, "high": close + 0.001, "low": close - 0.001, "close": close},
        index=idx,
    )


class _BuyOnFirstBarStrategy(NoOpStrategy):
    """Deterministic engine-mechanics validation strategy (not profitable)."""

    name = "buy_first"

    def __init__(self) -> None:
        super().__init__()
        self._traded = False

    def on_bar(self, context):
        if self._traded:
            return []
        self._traded = True
        return [
            OrderIntent(
                order_id="buy1",
                symbol=context.symbol,
                side=OrderSide.BUY,
                quantity=1000.0,
                timestamp=context.now.to_pydatetime(),
            )
        ]


class TestEventBacktester:
    def test_noop_unchanged(self):
        df = _make_frame()
        res = EventBacktester(BacktestConfig(symbol="EURUSD")).run(df, NoOpStrategy())
        assert len(res.trades) == 0
        assert res.metrics.net_pnl == 0.0
        assert res.equity_curve[-1].equity == pytest.approx(10000.0)

    def test_buy_first_opens_position(self):
        df = _make_frame()
        res = EventBacktester(BacktestConfig(symbol="EURUSD")).run(
            df, _BuyOnFirstBarStrategy()
        )
        assert len(res.trades) == 1
        t = res.trades[0]
        assert t.side == OrderSide.BUY

    def test_equity_curve_length(self):
        df = _make_frame(n=200)
        res = EventBacktester(BacktestConfig(symbol="EURUSD")).run(df, NoOpStrategy())
        assert len(res.equity_curve) == 200

    def test_start_end_slicing(self):
        df = _make_frame(n=200)
        cfg = BacktestConfig(
            symbol="EURUSD",
            start=df.index[50].to_pydatetime(),
            end=df.index[150].to_pydatetime(),
        )
        res = EventBacktester(cfg).run(df, NoOpStrategy())
        assert len(res.equity_curve) == 101

    def test_empty_frame_raises(self):
        with pytest.raises(ValueError):
            EventBacktester(BacktestConfig()).run(pd.DataFrame(), NoOpStrategy())

    def test_reproducible_metadata(self):
        df = _make_frame(n=60)
        res = EventBacktester(BacktestConfig(symbol="EURUSD")).run(
            df, NoOpStrategy(), provider="twelvedata"
        )
        assert res.metadata.provider == "twelvedata"
        assert res.metadata.symbol == "EURUSD"
        assert res.metadata.strategy == "noop"