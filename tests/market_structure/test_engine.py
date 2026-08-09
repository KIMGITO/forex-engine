"""Tests for the central market-structure engine."""

import numpy as np
import pandas as pd
import pytest

from app.market_structure.engine import MarketStructureConfig, MarketStructureEngine
from app.market_structure.errors import MarketStructureError


@pytest.fixture
def engine() -> MarketStructureEngine:
    return MarketStructureEngine()


def _make_varied_ohlc(n: int = 120) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    rng = np.random.default_rng(42)
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    high = close + 0.5
    low = close - 0.5
    open_p = close - 0.1
    return pd.DataFrame(
        {"open": open_p, "high": high, "low": low, "close": close},
        index=idx,
    )


class TestMarketStructureEngine:
    def test_returns_all_sections(self, engine: MarketStructureEngine) -> None:
        data = _make_varied_ohlc()
        result = engine.analyze(data, symbol="EURUSD", timeframe="1h")
        assert result.symbol == "EURUSD"
        assert result.timeframe == "1h"
        assert isinstance(result.swings, list)
        assert isinstance(result.structure, list)
        assert isinstance(result.breaks, list)
        assert isinstance(result.liquidity_zones, list)
        assert isinstance(result.sweeps, list)
        assert isinstance(result.displacement, list)
        assert isinstance(result.ranges, list)
        # Displacement should have one event per bar.
        assert len(result.displacement) == len(data)

    def test_custom_config(self) -> None:
        cfg = MarketStructureConfig(swing_left=2, swing_right=2, confirm_bars=1)
        engine = MarketStructureEngine(cfg)
        data = _make_varied_ohlc()
        result = engine.analyze(data, symbol="GBPUSD", timeframe="4h")
        assert result.timeframe == "4h"
        assert len(result.swings) > 0

    def test_missing_columns_raise(self, engine: MarketStructureEngine) -> None:
        bad = pd.DataFrame({"close": [1.0, 2.0]})
        with pytest.raises(MarketStructureError):
            engine.analyze(bad, symbol="EURUSD", timeframe="1h")

    def test_insufficient_data_raises(self, engine: MarketStructureEngine) -> None:
        idx = pd.date_range("2024-01-01", periods=5, freq="1h", tz="UTC")
        small = pd.DataFrame(
            {
                "open": [1.0] * 5,
                "high": [2.0] * 5,
                "low": [0.5] * 5,
                "close": [1.5] * 5,
            },
            index=idx,
        )
        with pytest.raises(MarketStructureError):
            engine.analyze(small, symbol="EURUSD", timeframe="1h")

    def test_all_swings_have_confirmation(self, engine: MarketStructureEngine) -> None:
        data = _make_varied_ohlc()
        result = engine.analyze(data, symbol="EURUSD", timeframe="1h")
        for swing in result.swings:
            assert swing.confirmation_timestamp >= swing.timestamp
            assert swing.available_from >= swing.timestamp