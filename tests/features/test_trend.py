"""Tests for the trend feature module."""

import numpy as np
import pandas as pd
import pytest

from app.features.errors import InsufficientDataError
from app.features.trend import distance_from_ma, ema, ma_slope, sma


@pytest.fixture
def df() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=5, freq="1h", tz="UTC")
    return pd.DataFrame({"close": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=idx)


class TestSMA:
    def test_known_values(self, df: pd.DataFrame) -> None:
        result = sma(df, period=3)
        assert np.isnan(result.iloc[0])
        assert np.isnan(result.iloc[1])
        assert result.iloc[2] == pytest.approx(2.0)  # mean(1,2,3)
        assert result.iloc[3] == pytest.approx(3.0)  # mean(2,3,4)
        assert result.iloc[4] == pytest.approx(4.0)  # mean(3,4,5)

    def test_insufficient_data_raises(self, df: pd.DataFrame) -> None:
        with pytest.raises(InsufficientDataError):
            sma(df, period=10)


class TestEMA:
    def test_known_values(self, df: pd.DataFrame) -> None:
        result = ema(df, period=3)
        # First valid at index 2 (min_periods=3)
        assert np.isnan(result.iloc[0])
        assert np.isnan(result.iloc[1])
        # EMA(3) of [1,2,3] with adjust=False, span=3
        alpha = 2.0 / (3.0 + 1.0)
        expected_2 = 1.0 * (1 - alpha) ** 2 + 2.0 * alpha * (1 - alpha) + 3.0 * alpha
        assert result.iloc[2] == pytest.approx(expected_2)

    def test_insufficient_data_raises(self, df: pd.DataFrame) -> None:
        with pytest.raises(InsufficientDataError):
            ema(df, period=10)


class TestDistanceFromMA:
    def test_known_values(self, df: pd.DataFrame) -> None:
        result = distance_from_ma(df, period=3)
        # At index 4: price=5, SMA=4 -> (5/4 - 1)*100 = 25.0
        assert result.iloc[4] == pytest.approx(25.0)

    def test_ema_type(self, df: pd.DataFrame) -> None:
        result = distance_from_ma(df, period=3, ma_type="ema")
        assert np.isnan(result.iloc[0])
        assert np.isnan(result.iloc[1])
        assert not np.isnan(result.iloc[2])


class TestMASlope:
    def test_known_values(self, df: pd.DataFrame) -> None:
        result = ma_slope(df, period=3, slope_periods=1)
        # SMA at index 3 = 3.0, at index 4 = 4.0 -> (4/3 - 1)*100 = 33.33...
        assert result.iloc[4] == pytest.approx(33.33333333333333)

    def test_insufficient_data_raises(self, df: pd.DataFrame) -> None:
        with pytest.raises(InsufficientDataError):
            ma_slope(df, period=10)