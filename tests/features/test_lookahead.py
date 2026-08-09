"""Regression tests proving causal (no-look-ahead) feature computation."""

import numpy as np
import pandas as pd
import pytest

from app.features._lookahead import truncate_prefix
from app.features.engine import FeatureEngine
from app.features.momentum import rsi
from app.features.trend import ema, sma


@pytest.fixture
def df() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=100, freq="1h", tz="UTC")
    rng = np.random.default_rng(123)
    return pd.DataFrame(
        {
            "close": 1.0850 + np.cumsum(rng.normal(0, 0.001, 100)),
            "high": 1.0850 + np.cumsum(rng.normal(0, 0.001, 100)) + 0.001,
            "low": 1.0850 + np.cumsum(rng.normal(0, 0.001, 100)) - 0.001,
        },
        index=idx,
    )


class TestLookAheadProtection:
    """Feature values at time T must not change when future data is altered."""

    def test_rsi_unchanged_when_future_prices_modified(self, df: pd.DataFrame) -> None:
        # Full-series RSI
        full_rsi = rsi(df, period=14)

        # A "future-only" version of df where the last 10 rows are perturbed.
        modified = df.copy()
        modified.iloc[-10:, modified.columns.get_loc("close")] *= 10.0
        modified.iloc[-10:, modified.columns.get_loc("high")] *= 10.0
        modified.iloc[-10:, modified.columns.get_loc("low")] *= 10.0
        truncated_rsi = rsi(modified, period=14)

        # RSI values at all timestamps prior to the perturbed region must match.
        assert truncated_rsi.iloc[:-10].equals(full_rsi.iloc[:-10])

    def test_ema_unchanged_when_future_prices_modified(self, df: pd.DataFrame) -> None:
        full_ema = ema(df, period=20)
        modified = df.copy()
        modified.iloc[-10:, modified.columns.get_loc("close")] *= 10.0
        truncated_ema = ema(modified, period=20)
        assert truncated_ema.iloc[:-10].equals(full_ema.iloc[:-10])

    def test_sma_unchanged_when_future_prices_modified(self, df: pd.DataFrame) -> None:
        full_sma = sma(df, period=20)
        modified = df.copy()
        modified.iloc[-10:, modified.columns.get_loc("close")] *= 10.0
        truncated_sma = sma(modified, period=20)
        assert truncated_sma.iloc[:-10].equals(full_sma.iloc[:-10])

    def test_engine_features_unchanged_for_earlier_rows(
        self, df: pd.DataFrame
    ) -> None:
        engine = FeatureEngine()
        full = engine.calculate(df, features=["rsi", "sma", "atr"])
        modified = df.copy()
        modified.iloc[-10:, :] *= 10.0
        recomputed = engine.calculate(modified, features=["rsi", "sma", "atr"])
        # All rows strictly before the perturbed window must match exactly.
        assert full.iloc[:-10].equals(recomputed.iloc[:-10])

    def test_truncate_prefix_helper(self, df: pd.DataFrame) -> None:
        prefixed = truncate_prefix(df, 5)
        assert len(prefixed) == 5
        assert list(prefixed.index) == list(df.index[:5])