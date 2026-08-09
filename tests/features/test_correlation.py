"""Tests for the correlation feature module."""

import numpy as np
import pandas as pd
import pytest

from app.features.correlation import (
    align_price_dfs,
    correlation_matrix,
    pairwise_correlation,
    rolling_correlation,
)
from app.features.errors import FeatureError, InsufficientDataError


@pytest.fixture
def aligned_df() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=30, freq="1h", tz="UTC")
    x = np.arange(1.0, 31.0)
    return pd.DataFrame(
        {
            "A": x,
            "B": 2.0 * x,
            "C": 31.0 - x,
            "D": np.random.default_rng(42).uniform(0, 10, 30),
        },
        index=idx,
    )


class TestAlignPriceDfs:
    def test_aligns_single(self) -> None:
        idx = pd.date_range("2024-01-01", periods=3, freq="1h", tz="UTC")
        data_map = {"EURUSD": pd.DataFrame({"close": [1.0, 2.0, 3.0]}, index=idx)}
        aligned = align_price_dfs(data_map)
        assert list(aligned.columns) == ["EURUSD"]
        assert len(aligned) == 3

    def test_aligns_multiple(self) -> None:
        idx1 = pd.date_range("2024-01-01", periods=3, freq="1h", tz="UTC")
        idx2 = pd.date_range("2024-01-01 01:00", periods=3, freq="1h", tz="UTC")
        data_map = {
            "A": pd.DataFrame({"close": [1.0, 2.0, 3.0]}, index=idx1),
            "B": pd.DataFrame({"close": [10.0, 20.0, 30.0]}, index=idx2),
        }
        aligned = align_price_dfs(data_map)
        assert len(aligned) == 4
        assert "A" in aligned.columns
        assert "B" in aligned.columns
        assert np.isnan(aligned.loc[aligned.index[0], "B"])

    def test_empty_map_raises(self) -> None:
        with pytest.raises(ValueError):
            align_price_dfs({})

    def test_no_overlap_raises(self) -> None:
        idx1 = pd.date_range("2024-01-01", periods=3, freq="1h", tz="UTC")
        idx2 = pd.date_range("2025-01-01", periods=3, freq="1h", tz="UTC")
        data_map = {
            "A": pd.DataFrame({"close": [1.0, 2.0, 3.0]}, index=idx1),
            "B": pd.DataFrame({"close": [10.0, 20.0, 30.0]}, index=idx2),
        }
        with pytest.raises(FeatureError):
            align_price_dfs(data_map)


class TestPairwiseCorrelation:
    def test_perfect_positive(self, aligned_df: pd.DataFrame) -> None:
        corr = pairwise_correlation(aligned_df, "A", "B")
        assert corr == pytest.approx(1.0)

    def test_perfect_negative(self, aligned_df: pd.DataFrame) -> None:
        corr = pairwise_correlation(aligned_df, "A", "C")
        assert corr == pytest.approx(-1.0)

    def test_insufficient_data_raises(self, aligned_df: pd.DataFrame) -> None:
        with pytest.raises(InsufficientDataError):
            pairwise_correlation(aligned_df, "A", "B", min_periods=100)


class TestRollingCorrelation:
    def test_perfect_positive_trailing(self, aligned_df: pd.DataFrame) -> None:
        result = rolling_correlation(aligned_df, "A", "B", window=3)
        assert np.isnan(result.iloc[0])
        assert np.isnan(result.iloc[1])
        assert result.iloc[2] == pytest.approx(1.0)
        assert result.iloc[3] == pytest.approx(1.0)
        assert result.iloc[4] == pytest.approx(1.0)
        assert result.iloc[29] == pytest.approx(1.0)


class TestCorrelationMatrix:
    def test_symmetric(self, aligned_df: pd.DataFrame) -> None:
        mat = correlation_matrix(aligned_df)
        for a in aligned_df.columns:
            for b in aligned_df.columns:
                assert mat.loc[a, b] == pytest.approx(mat.loc[b, a])

    def test_diagonal_is_one(self, aligned_df: pd.DataFrame) -> None:
        mat = correlation_matrix(aligned_df)
        for col in aligned_df.columns:
            assert mat.loc[col, col] == pytest.approx(1.0)