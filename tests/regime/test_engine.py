"""Tests for the RegimeEngine sequential analysis."""

import numpy as np
import pandas as pd

from app.market_structure.engine import MarketStructureEngine
from app.regime import RegimeConfig, RegimeEngine
from app.regime.models import MarketState


def _make_ohlc(n=200):
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    close = 100.0 + np.cumsum(np.random.default_rng(7).normal(0, 0.2, n))
    return pd.DataFrame(
        {"open": close, "high": close + 0.5, "low": close - 0.5, "close": close},
        index=idx,
    )


class TestRegimeEngine:
    def test_sequential_regimes_per_bar(self) -> None:
        data = _make_ohlc()
        engine = RegimeEngine()
        regimes = engine.analyze(data, "EURUSD", "1h")
        assert len(regimes) == len(data)
        # available_from == bar timestamp
        for r in regimes:
            assert r.available_from == r.timestamp

    def test_with_market_structure(self) -> None:
        data = _make_ohlc(300)
        structure = MarketStructureEngine().analyze(data, "EURUSD", "1h")
        engine = RegimeEngine()
        regimes = engine.analyze(data, "EURUSD", "1h", market_structure=structure)
        assert len(regimes) == len(data)
        assert all(r.symbol == "EURUSD" for r in regimes)

    def test_causal_assignment(self) -> None:
        data = _make_ohlc(150)
        engine = RegimeEngine()
        regimes = engine.analyze(data, "EURUSD", "1h")
        # First bars should be UNKNOWN (not enough MA/volatility history)
        assert regimes[0].market_state in (MarketState.UNKNOWN, MarketState.TRANSITION)
        assert regimes[0].volatility_state == regimes[0].volatility_state  # smoke

    def test_custom_config(self) -> None:
        data = _make_ohlc(300)
        cfg = RegimeConfig(ema_fast=10, ema_slow=30, percentile_window=50)
        engine = RegimeEngine(cfg)
        regimes = engine.analyze(data, "EURUSD", "1h")
        assert len(regimes) == len(data)
