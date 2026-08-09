"""Tests for cost models and pip utilities."""

import pytest

from app.backtest.costs import (
    FixedPerTradeCommissionModel,
    FixedSlippageModel,
    FixedSpreadModel,
    NoSwapModel,
    PercentageCommissionModel,
    ZeroCommissionModel,
    pip_distance,
    pip_size_for_symbol,
)


class TestPipUtilities:
    def test_eurusd_pip_size(self):
        assert pip_size_for_symbol("EURUSD") == 0.0001

    def test_jpy_pip_size(self):
        assert pip_size_for_symbol("USDJPY") == 0.01

    def test_pip_distance(self):
        assert pip_distance(1.10000, 1.10010, 0.0001) == pytest.approx(1.0)


class TestFixedSpreadModel:
    def test_bid_ask_symmetric(self):
        model = FixedSpreadModel(spread_pips=1.0, pip_size=0.0001)
        bid, ask = model.bid_ask(1.10000)
        assert ask - bid == pytest.approx(0.0001)
        assert bid < 1.10000 < ask

    def test_zero_spread(self):
        model = FixedSpreadModel(0.0, 0.0001)
        assert model.bid_ask(1.0) == (1.0, 1.0)

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            FixedSpreadModel(-1.0, 0.0001)


class TestFixedSlippageModel:
    def test_buy_slippage_higher(self):
        model = FixedSlippageModel(slippage_pips=1.0, pip_size=0.0001)
        assert model.slippage_price(1.10000, "buy") == pytest.approx(1.10010)

    def test_sell_slippage_lower(self):
        model = FixedSlippageModel(slippage_pips=1.0, pip_size=0.0001)
        assert model.slippage_price(1.10000, "sell") == pytest.approx(1.09990)


class TestCommissionModels:
    def test_zero(self):
        assert ZeroCommissionModel().commission(1000.0, 1.0) == 0.0

    def test_fixed(self):
        assert FixedPerTradeCommissionModel(1.5).commission(1000.0, 1.0) == 1.5

    def test_percentage(self):
        assert PercentageCommissionModel(0.001).commission(1000.0, 1.0) == pytest.approx(1.0)


class TestNoSwapModel:
    def test_zero_financing(self):
        assert NoSwapModel().financing(1000.0, 10) == 0.0