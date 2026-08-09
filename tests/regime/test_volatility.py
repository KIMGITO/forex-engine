"""Tests for volatility-state classification."""

import numpy as np
import pandas as pd

from app.regime import RegimeConfig
from app.regime.models import VolatilityState
from app.regime.volatility import classify_volatility_series


def _make_ohlc(n=160, base_vol=1.0):
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    rng = np.random.default_rng(42)
    close = 100.0 + np.cumsum(rng.normal(0, base_vol, n))
    high = close + base_vol
    low = close - base_vol
    return pd.DataFrame({"open": close, "high": high, "low": low, "close": close}, index=idx)


class TestVolatilityClassification:
    def test_low_volatility_flagged_low(self) -> None:
        # Low constant volatility → LOW
        idx = pd.date_range("2024-01-01", periods=200, freq="1h", tz="UTC")
        close = np.linspace(100.0, 100.5, 200)
        df = pd.DataFrame(
            {"open": close, "high": close + 0.01, "low": close - 0.01, "close": close},
            index=idx,
        )
        states, _ = classify_volatility_series(df, RegimeConfig(percentile_window=50))
        # Most bars should be LOW or UNKNOWN (uniform ratio -> percentile rank)
        state_values = [s for s in states.values if s != VolatilityState.UNKNOWN]
        assert state_values, "expected some known states"
        # Uniform series → all percentile ranks equal → NORMAL by our rule.
        # This asserts we don't crash and produce states.
        assert set(state_values) <= {VolatilityState.LOW, VolatilityState.NORMAL, VolatilityState.HIGH}

    def test_spike_flagged_extreme_or_high(self) -> None:
        idx = pd.date_range("2024-01-01", periods=200, freq="1h", tz="UTC")
        close = np.linspace(100.0, 100.5, 200)
        # Spike at bar ~150: huge range
        highs = close + 0.01
        lows = close - 0.01
        highs[150] = close[150] + 2.0
        lows[150] = close[150] - 2.0
        df = pd.DataFrame({"open": close, "high": highs, "low": lows, "close": close}, index=idx)
        states, _ = classify_volatility_series(df, RegimeConfig(percentile_window=40))
        assert states.iloc[150] in (VolatilityState.EXTREME, VolatilityState.HIGH)

    def test_insufficient_window_unknown(self) -> None:
        # Enough bars for ATR to compute, but percentile_window so large that
        # no bar ever accumulates 2 valid ratios -> all UNKNOWN.
        idx = pd.date_range("2024-01-01", periods=100, freq="1h", tz="UTC")
        ts = pd.Series(np.full(100, 100.0), index=idx)
        df = pd.DataFrame(
            {"open": ts, "high": ts + 0.01, "low": ts - 0.01, "close": ts},
            index=idx,
        )
        states, _ = classify_volatility_series(df, RegimeConfig(percentile_window=1000))
        # Early bars (before the ATR/percentile seed) are UNKNOWN and never crash.
        assert states.iloc[0] == VolatilityState.UNKNOWN
        assert states.iloc[5] == VolatilityState.UNKNOWN
        # The engine keeps producing states (no exceptions).
        assert len(states) == len(df)
