"""Tests for the momentum feature module."""

import numpy as np
import pandas as pd
import pytest

from app.features.errors import InsufficientDataError
from app.features.momentum import price_momentum, rate_of_change, rsi


@pytest.fixture
def df() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=6, freq="1h", tz="UTC")
    # Prices: 100, 110, 121, 110, 100, 110
    return pd.DataFrame({"close": [100.0, 110.0, 121.0, 110.0, 100.0, 110.0]}, index=idx)


class TestRateOfChange:
    def test_known_values(self, df: pd.DataFrame) -> None:
        result = rate_of_change(df, period=2)
        # 100 -> 121 : (121/100 - 1)*100 = 21.0
        assert result.iloc[2] == pytest.approx(21.0)
        # 110 -> 110 : (110/110 - 1)*100 = 0.0
        assert result.iloc[3] == pytest.approx(0.0)
        # 121 -> 100 : (100/121 - 1)*100 = -17.355...
        assert result.iloc[4] == pytest.approx(-17.35537190082645)

    def test_insufficient_lookback_is_nan(self, df: pd.DataFrame) -> None:
        result = rate_of_change(df, period=2)
        assert np.isnan(result.iloc[0])
        assert np.isnan(result.iloc[1])

    def test_insufficient_data_raises(self, df: pd.DataFrame) -> None:
        with pytest.raises(InsufficientDataError):
            rate_of_change(df, period=100)


class TestPriceMomentum:
    def test_known_values(self, df: pd.DataFrame) -> None:
        result = price_momentum(df, period=2)
        # 100 -> 121 : 121 - 100 = 21
        assert result.iloc[2] == pytest.approx(21.0)
        # 110 -> 110 : 0
        assert result.iloc[3] == pytest.approx(0.0)
        # 121 -> 100 : -21
        assert result.iloc[4] == pytest.approx(-21.0)

    def test_insufficient_lookback_is_nan(self, df: pd.DataFrame) -> None:
        result = price_momentum(df, period=2)
        assert np.isnan(result.iloc[0])
        assert np.isnan(result.iloc[1])

    def test_insufficient_data_raises(self, df: pd.DataFrame) -> None:
        with pytest.raises(InsufficientDataError):
            price_momentum(df, period=100)


class TestRSI:
    def test_known_values(self) -> None:
        # Monotonic increasing prices -> RSI = 100
        idx = pd.date_range("2024-01-01", periods=20, freq="1h", tz="UTC")
        up = pd.DataFrame({"close": np.arange(1.0, 21.0)}, index=idx)
        result = rsi(up, period=14)
        assert result.iloc[14] == pytest.approx(100.0)

    def test_known_values_down(self) -> None:
        # Monotonic decreasing prices -> RSI = 0
        idx = pd.date_range("2024-01-01", periods=20, freq="1h", tz="UTC")
        down = pd.DataFrame({"close": np.arange(20.0, 0.0, -1.0)}, index=idx)
        result = rsi(down, period=14)
        assert result.iloc[14] == pytest.approx(0.0)

    def test_insufficient_lookback_is_nan(self, df: pd.DataFrame) -> None:
        result = rsi(df, period=2)
        assert np.isnan(result.iloc[0])
        assert np.isnan(result.iloc[1])

    def test_insufficient_data_raises(self, df: pd.DataFrame) -> None:
        with pytest.raises(InsufficientDataError):
            rsi(df, period=100)