"""Tests for trend-state classification."""

import numpy as np
import pandas as pd

from app.regime import RegimeConfig
from app.regime.models import TrendState
from app.regime.trend import classify_trend_series


def _trending_up(n=120):
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    close = 100.0 + np.arange(n) * 0.3
    return pd.DataFrame({"open": close, "high": close + 0.2, "low": close - 0.2, "close": close}, index=idx)


def _trending_down(n=120):
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    close = 100.0 + np.arange(n) * -0.3
    return pd.DataFrame({"open": close, "high": close + 0.2, "low": close - 0.2, "close": close}, index=idx)


def _flat(n=120):
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    close = np.full(n, 100.0)
    return pd.DataFrame({"open": close, "high": close + 0.2, "low": close - 0.2, "close": close}, index=idx)


class TestTrendClassification:
    def test_persistent_bullish(self) -> None:
        states = classify_trend_series(_trending_up(), RegimeConfig())
        # Later bars (after enough MA/slope history) should be mostly BULLISH
        later = [s for s in states.values[-40:]]
        assert later.count(TrendState.BULLISH) > later.count(TrendState.BEARISH)

    def test_persistent_bearish(self) -> None:
        states = classify_trend_series(_trending_down(), RegimeConfig())
        later = [s for s in states.values[-40:]]
        assert later.count(TrendState.BEARISH) > later.count(TrendState.BULLISH)

    def test_neutral_market(self) -> None:
        states = classify_trend_series(_flat(), RegimeConfig())
        later = [s for s in states.values[-40:]]
        # Flat market should never be decisively bullish/bearish in bulk
        assert later.count(TrendState.BULLISH) < 20
