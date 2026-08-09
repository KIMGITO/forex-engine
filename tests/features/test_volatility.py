"""Tests for the volatility feature module."""

import numpy as np
import pandas as pd
import pytest

from app.features.errors import InsufficientDataError
from app.features.volatility import (
    annualized_volatility,
    atr,
    rolling_atr,
    rolling_std_init,
    volatility_percentile,
)


@pytest.fixture
def df() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=5, freq="1h", tz="UTC")
    return pd.DataFrame({"close": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=idx)


@pytest.fixture
def atr_df() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=4, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "high": [11.0, 11.0, 11.0, 11.0],
            "low": [9.0, 9.0, 9.0, 9.0],
            "close": [10.0, 10.0, 10.0, 10.0],
        },
        index=idx,
    )


class TestRollingStd:
    def test_known_values(self, df: pd.DataFrame) -> None:
        result = rolling_std_init(df, window=3)
        assert np.isnan(result.iloc[0])
        assert np.isnan(result.iloc[1])
        assert result.iloc[2] == pytest.approx(1.0)
        assert result.iloc[3] == pytest.approx(1.0)
        assert result.iloc[4] == pytest.approx(1.0)

    def test_insufficient_data_raises(self, df: pd.DataFrame) -> None:
        with pytest.raises(InsufficientDataError):
            rolling_std_init(df, window=10)


class TestAnnualizedVolatility:
    def test_relationship_to_rolling_std(self, df: pd.DataFrame) -> None:
        result = annualized_volatility(df, window=2, timeframe="1h")
        returns = df["close"].pct_change(fill_method=None)
        expected = returns.rolling(window=2, min_periods=2).std(ddof=1) * np.sqrt(8760.0)
        pd.testing.assert_series_equal(result, expected)

    def test_insufficient_data_raises(self, df: pd.DataFrame) -> None:
        with pytest.raises(InsufficientDataError):
            annualized_volatility(df, window=10)


class TestATR:
    def test_constant_true_range(self, atr_df: pd.DataFrame) -> None:
        result = atr(atr_df, window=2)
        assert result.iloc[1] == pytest.approx(2.0)
        assert result.iloc[2] == pytest.approx(2.0)
        assert result.iloc[3] == pytest.approx(2.0)

    def test_insufficient_data_raises(self, atr_df: pd.DataFrame) -> None:
        with pytest.raises(InsufficientDataError):
            atr(atr_df, window=10)


class TestRollingATR:
    def test_constant_true_range(self, atr_df: pd.DataFrame) -> None:
        result = rolling_atr(atr_df, window=2)
        assert result.iloc[1] == pytest.approx(2.0)
        assert result.iloc[2] == pytest.approx(2.0)
        assert result.iloc[3] == pytest.approx(2.0)

    def test_insufficient_data_raises(self, atr_df: pd.DataFrame) -> None:
        with pytest.raises(InsufficientDataError):
            rolling_atr(atr_df, window=10)


class TestVolatilityPercentile:
    def test_known_values(self, df: pd.DataFrame) -> None:
        result = volatility_percentile(df, window=2, rank_window=3)
        assert np.isnan(result.iloc[0])
        assert np.isnan(result.iloc[1])
        assert np.isnan(result.iloc[2])
        assert result.iloc[3] == pytest.approx(100.0)
        assert result.iloc[4] == pytest.approx(100.0)

    def test_insufficient_data_raises(self, df: pd.DataFrame) -> None:
        with pytest.raises(InsufficientDataError):
            volatility_percentile(df, window=2, rank_window=100)