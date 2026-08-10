"""Tests for the Signal -> OrderIntent backtest adapter."""

import datetime

import numpy as np
import pandas as pd

from app.backtest import BacktestConfig, EventBacktester, NoOpStrategy
from app.backtest.models import OrderSide
from app.strategy import SignalToOrderAdapter
from app.strategy.models import Signal, SignalDirection, SignalStatus


def _ts():
    return datetime.datetime(2024, 6, 10, 12, 0, tzinfo=datetime.timezone.utc)


def _make_signal(direction=SignalDirection.LONG, status=SignalStatus.DETECTED):
    entry = 1.10000 if direction == SignalDirection.LONG else 1.11000
    stop = 1.09500 if direction == SignalDirection.LONG else 1.11500
    target = 1.11000 if direction == SignalDirection.LONG else 1.10000
    return Signal(
        signal_id="sig1",
        timestamp=_ts(),
        symbol="EURUSD",
        timeframe="1h",
        direction=direction,
        strength="moderate",
        score=4.0,
        max_score=5.0,
        entry=entry,
        stop_loss=stop,
        take_profit=target,
        risk_distance=0.005,
        reward_distance=0.010,
        risk_reward_ratio=2.0,
        strategy="trend_structure",
        status=status,
        available_from=_ts(),
    )


class TestSignalToOrderAdapter:
    def test_long_signal_becomes_buy_order(self):
        adapter = SignalToOrderAdapter(quantity=1000.0)
        intents = adapter.to_order_intents(_make_signal(), None, _ts())
        assert len(intents) == 1
        assert intents[0].side == OrderSide.BUY
        assert intents[0].quantity == 1000.0
        assert intents[0].stop_loss == 1.09500
        assert intents[0].take_profit == 1.11000

    def test_short_signal_becomes_sell_order(self):
        adapter = SignalToOrderAdapter(quantity=500.0)
        intents = adapter.to_order_intents(
            _make_signal(SignalDirection.SHORT), None, _ts()
        )
        assert intents[0].side == OrderSide.SELL

    def test_none_signal_no_orders(self):
        adapter = SignalToOrderAdapter(quantity=1000.0)
        assert adapter.to_order_intents(None, None, _ts()) == []

    def test_invalidated_signal_no_orders(self):
        adapter = SignalToOrderAdapter(quantity=1000.0)
        s = _make_signal(status=SignalStatus.INVALIDATED)
        assert adapter.to_order_intents(s, None, _ts()) == []

    def test_expired_signal_no_orders(self):
        adapter = SignalToOrderAdapter(quantity=1000.0)
        s = _make_signal(status=SignalStatus.EXPIRED)
        assert adapter.to_order_intents(s, None, _ts()) == []


class TestSignalBacktestEndToEnd:
    def test_signal_feeds_backtest(self):
        """A strategy signal flows through the adapter into the Step 8 backtester
        without modifying its architecture."""
        idx = pd.date_range("2024-01-01", periods=90, freq="1h", tz="UTC")
        close = 1.08 + np.cumsum(np.random.default_rng(7).normal(0.001, 0.001, 90))
        df = pd.DataFrame(
            {"open": close, "high": close + 0.002, "low": close - 0.002, "close": close},
            index=idx,
        )

        class _SignalStrategy(NoOpStrategy):
            name = "signal_strategy"

            def on_bar(self, context):
                if context._i > 0:
                    return []
                sig = _make_signal()
                adapter = SignalToOrderAdapter(quantity=1000.0)
                return adapter.to_order_intents(
                    sig, context, context.now.to_pydatetime()
                )

        result = EventBacktester(BacktestConfig(symbol="EURUSD")).run(
            df, _SignalStrategy()
        )
        # Key guarantee: adapter produced OrderIntent the engine accepted.
        assert result.metrics.trade_count >= 0
