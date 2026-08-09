"""Tests for portfolio accounting and position lifecycle."""

from datetime import datetime, timezone

import pytest

from app.backtest.models import Fill, OrderSide
from app.backtest.portfolio import Portfolio, pnl_for_sides


def _fill(side, qty, price, ts):
    return Fill(
        order_id="o",
        symbol="EURUSD",
        side=side,
        quantity=qty,
        price=price,
        timestamp=ts,
        gross_value=price * qty,
    )


def _ts(hour=12):
    return datetime(2024, 6, 10, hour, 0, tzinfo=timezone.utc)


class TestPnlForSides:
    def test_long_up(self):
        assert pnl_for_sides(OrderSide.BUY, 1000, 1.1000, 1.1100) == pytest.approx(10.0)

    def test_long_down(self):
        assert pnl_for_sides(OrderSide.BUY, 1000, 1.1000, 1.0900) == pytest.approx(-10.0)

    def test_short_down_up(self):
        assert pnl_for_sides(OrderSide.SELL, 1000, 1.1000, 1.0900) == pytest.approx(10.0)


class TestPortfolio:
    def test_initial_state(self):
        p = Portfolio(10000.0)
        assert p.balance == 10000.0
        assert p.equity(1.0) == 10000.0

    def test_open_long_then_close_profit(self):
        p = Portfolio(10000.0)
        t0, t1 = _ts(10), _ts(11)
        p.open_position(_fill(OrderSide.BUY, 1000, 1.1000, t0))
        assert p.unrealized_pnl(1.1100) == pytest.approx(10.0)
        p.open_position(_fill(OrderSide.SELL, 1000, 1.1100, t1))
        assert len(p.positions) == 0
        assert p.balance == pytest.approx(10010.0)

    def test_open_short_then_close_profit(self):
        p = Portfolio(10000.0)
        t0, t1 = _ts(10), _ts(11)
        p.open_position(_fill(OrderSide.SELL, 1000, 1.1000, t0))
        assert p.unrealized_pnl(1.0900) == pytest.approx(10.0)
        p.open_position(_fill(OrderSide.BUY, 1000, 1.0900, t1))
        assert len(p.positions) == 0
        assert p.balance == pytest.approx(10010.0)

    def test_margin_leverage(self):
        p = Portfolio(10000.0, leverage=10)
        p.open_position(_fill(OrderSide.BUY, 1000, 1.1000, _ts(10)))
        assert p.used_margin() == pytest.approx(110.0)

    def test_commission_applied(self):
        p = Portfolio(10000.0)
        p.apply_commission(5.0)
        assert p.balance == pytest.approx(9995.0)
        assert p.fees == pytest.approx(5.0)

    def test_equity_unrealized(self):
        p = Portfolio(10000.0)
        p.open_position(_fill(OrderSide.BUY, 1000, 1.1000, _ts(10)))
        assert p.equity(1.1050) == pytest.approx(10005.0)

    def test_fifo_average(self):
        p = Portfolio(10000.0)
        t0, t1 = _ts(10), _ts(11)
        p.open_position(_fill(OrderSide.BUY, 1000, 1.1000, t0))
        p.open_position(_fill(OrderSide.BUY, 1000, 1.1100, t1))
        pos = p.positions["EURUSD"]
        assert pos.quantity == 2000
        assert pos.average_entry == pytest.approx(1.1050)