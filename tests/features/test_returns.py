"""Tests for the returns feature module."""

import numpy as np
import pandas as pd
import pytest

from app.features.errors import InsufficientDataError
from app.features.returns import log_returns, pct_returns, simple_returns


@pytest.fixture
def df() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=5, freq="1h", tz="UTC")
    # Prices: 100, 102, 101, 103, 105
    return pd.DataFrame({"close": [100.0, 102.0, 101.0, 103.0, 105.0]}, index=idx)


class TestSimpleReturns:
    def test_first_obs_is_nan(self, df: pd.DataFrame) -> None:
        result = simple_returns(df)
        assert np.isnan(result.iloc[0])

    def test_known_values(self, df: pd.DataFrame) -> None:
        result = simple_returns(df)
        # 100 -> 102 : (102-100)/100 = 0.02
        assert result.iloc[1] == pytest.approx(0.02)
        # 102 -> 101 : (101-102)/102 = -0.0098039...
        assert result.iloc[2] == pytest.approx(-0.00980392156862745)

    def test_returns_preserves_index(self, df: pd.DataFrame) -> None:
        result = simple_returns(df)
        assert list(result.index) == list(df.index)

    def test_insufficient_data_raises(self) -> None:
        small = pd.DataFrame({"close": [100.0]}, index=pd.date_range("2024-01-01", periods=1, freq="1h", tz="UTC"))
        with pytest.raises(InsufficientDataError):
            simple_returns(small, min_obs=2)

    def test_missing_column_raises(self) -> None:
        with pytest.raises(ValueError):
            simple_returns(pd.DataFrame({"x": [1.0]}))


class TestPctReturns:
    def test_first_obs_is_nan(self, df: pd.DataFrame) -> None:
        result = pct_returns(df)
        assert np.isnan(result.iloc[0])

    def test_known_values(self, df: pd.DataFrame) -> None:
        result = pct_returns(df)
        # 100 -> 102 : (102/100 - 1) * 100 = 2.0
        assert result.iloc[1] == pytest.approx(2.0)
        # 102 -> 101 : (101/102 - 1) * 100 = -0.98039...
        assert result.iloc[2] == pytest.approx(-0.9803921568627451)


class TestLogReturns:
    def test_first_obs_is_nan(self, df: pd.DataFrame) -> None:
        result = log_returns(df)
        assert np.isnan(result.iloc[0])

    def test_known_values(self, df: pd.DataFrame) -> None:
        result = log_returns(df)
        # log(102/100)
        expected = np.log(102.0 / 100.0)
        assert result.iloc[1] == pytest.approx(expected)
        # log(101/102)
        expected2 = np.log(101.0 / 102.0)
        assert result.iloc[2] == pytest.approx(expected2)