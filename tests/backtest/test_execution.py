"""Tests for the execution simulator (fills, SL/TP, gaps)."""

from datetime import datetime, timezone

import pytest

from app.backtest.costs import FixedSlippageModel, FixedSpreadModel
from app.backtest.execution import ExecutionSimulator
from app.backtest.models import FillPolicy, Order, OrderSide, OrderType, Position


def _make_order(
    side=OrderSide.BUY,
    order_type=OrderType.MARKET,
    quantity=1000.0,
    requested=None,
    sl=None,
    tp=None,
) -> Order:
    return Order(
        order_id="o1",
        symbol="EURUSD",
        side=side,
        quantity=quantity,
        order_type=order_type,
        requested_price=requested,
        stop_loss=sl,
        take_profit=tp,
        timestamp=datetime(2024, 6, 10, 12, 0, tzinfo=timezone.utc),
    )


def _make_position(side=OrderSide.BUY, entry=1.10000, sl=1.09500, tp=1.11000) -> Position:
    return Position(
        symbol="EURUSD",
        side=side,
        quantity=1000.0,
        average_entry=entry,
        stop_loss=sl,
        take_profit=tp,
        opened_at=datetime(2024, 6, 10, 11, 0, tzinfo=timezone.utc),
    )


def _simulator():
    return ExecutionSimulator(
        FixedSpreadModel(1.0, 0.0001),
        FixedSlippageModel(0.0, 0.0001),
        FillPolicy.CONSERVATIVE_SL_FIRST,
    )


class TestMarketFill:
    def test_buy_at_ask(self):
        sim = _simulator()
        res = sim.evaluate_entry(
            _make_order(OrderSide.BUY), bar_mid=1.10000, bar_open=1.1, bar_high=1.11, bar_low=1.09
        )
        assert res.filled
        assert res.price == pytest.approx(1.10005)  # mid + spread/2

    def test_sell_at_bid(self):
        sim = _simulator()
        res = sim.evaluate_entry(
            _make_order(OrderSide.SELL), bar_mid=1.10000, bar_open=1.1, bar_high=1.11, bar_low=1.09
        )
        assert res.filled
        assert res.price == pytest.approx(1.09995)  # mid - spread/2


class TestLimitStop:
    def test_limit_buy_touched(self):
        sim = _simulator()
        res = sim.evaluate_entry(
            _make_order(OrderSide.BUY, OrderType.LIMIT, requested=1.09800),
            bar_mid=1.1, bar_open=1.1, bar_high=1.11, bar_low=1.09700,
        )
        assert res.filled
        assert res.price == pytest.approx(1.09800)

    def test_limit_not_touched(self):
        sim = _simulator()
        res = sim.evaluate_entry(
            _make_order(OrderSide.BUY, OrderType.LIMIT, requested=1.09500),
            bar_mid=1.1, bar_open=1.1, bar_high=1.11, bar_low=1.09700,
        )
        assert not res.filled

    def test_stop_gap_through(self):
        sim = _simulator()
        # Bar opens above the stop level (gap) — fill at open, not the level.
        res = sim.evaluate_entry(
            _make_order(OrderSide.BUY, OrderType.STOP, requested=1.09500),
            bar_mid=1.1, bar_open=1.09800, bar_high=1.11, bar_low=1.09600,
        )
        assert res.filled
        assert res.price == pytest.approx(1.09800)


class TestSLTPResolution:
    def test_sl_hit(self):
        sim = _simulator()
        pos = _make_position(sl=1.09500, tp=1.11000)
        res = sim.resolve_stop_take_profit(
            pos, bar_mid=1.09, bar_open=1.093, bar_high=1.096, bar_low=1.088
        )
        assert res is not None
        assert res.reason == "stop_loss"

    def test_tp_hit(self):
        sim = _simulator()
        pos = _make_position(sl=1.09500, tp=1.11000)
        res = sim.resolve_stop_take_profit(
            pos, bar_mid=1.115, bar_open=1.11, bar_high=1.12, bar_low=1.109
        )
        assert res is not None
        assert res.reason == "take_profit"

    def test_both_touched_conservative_sl(self):
        sim = _simulator()
        pos = _make_position(sl=1.09500, tp=1.11000)
        # Bar spans both levels.
        res = sim.resolve_stop_take_profit(
            pos, bar_mid=1.1, bar_open=1.1, bar_high=1.115, bar_low=1.090
        )
        assert res is not None
        assert res.reason == "sl_tp_both_touched_conservative_sl_first"
        assert res.price == pytest.approx(1.09500)

    def test_neither_touched(self):
        sim = _simulator()
        pos = _make_position(sl=1.09500, tp=1.11000)
        res = sim.resolve_stop_take_profit(
            pos, bar_mid=1.1, bar_open=1.1, bar_high=1.103, bar_low=1.097
        )
        assert res is None

    def test_sl_gap_worse_open(self):
        sim = _simulator()
        pos = _make_position(sl=1.09500, tp=1.11000)
        # Bar gaps below SL → fill at open (worse for long).
        res = sim.resolve_stop_take_profit(
            pos, bar_mid=1.09, bar_open=1.09200, bar_high=1.096, bar_low=1.085
        )
        assert res is not None
        assert res.price == pytest.approx(1.09200)