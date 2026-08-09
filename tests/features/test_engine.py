"""Tests for the central feature engine."""

import numpy as np
import pandas as pd
import pytest

from app.features.engine import FEATURE_REGISTRY, FeatureDefinition, FeatureEngine
from app.features.errors import UnknownFeatureError


@pytest.fixture
def engine() -> FeatureEngine:
    return FeatureEngine()


@pytest.fixture
def df() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=100, freq="1h", tz="UTC")
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "close": 1.0850 + np.cumsum(rng.normal(0, 0.001, 100)),
            "high": 1.0850 + np.cumsum(rng.normal(0, 0.001, 100)) + 0.001,
            "low": 1.0850 + np.cumsum(rng.normal(0, 0.001, 100)) - 0.001,
        },
        index=idx,
    )


class TestFeatureEngine:
    def test_known_features_returned(self, engine: FeatureEngine, df: pd.DataFrame) -> None:
        result = engine.calculate(df, features=["rsi", "atr", "sma"])
        assert list(result.columns) == ["rsi", "atr", "sma"]
        assert len(result) == len(df)

    def test_unknown_feature_raises(self, engine: FeatureEngine, df: pd.DataFrame) -> None:
        with pytest.raises(UnknownFeatureError):
            engine.calculate(df, features=["bogus_feature"])

    def test_multiple_features_calculated_together(
        self, engine: FeatureEngine, df: pd.DataFrame
    ) -> None:
        result = engine.calculate(
            df,
            features=["simple_returns", "log_returns", "pct_returns"],
        )
        assert list(result.columns) == ["simple_returns", "log_returns", "pct_returns"]
        assert np.isnan(result.iloc[0, 0])

    def test_param_overrides(self, engine: FeatureEngine, df: pd.DataFrame) -> None:
        result = engine.calculate(
            df,
            features=["sma"],
            params={"sma": {"period": 50}},
        )
        # First valid SMA at index 49 (0-based) with period=50
        assert np.isnan(result.iloc[48, 0])
        assert not np.isnan(result.iloc[49, 0])

    def test_register_feature(self, engine: FeatureEngine, df: pd.DataFrame) -> None:
        def rolling_mean(data: pd.DataFrame, **kwargs: dict) -> pd.Series:
            return data["close"].rolling(window=kwargs.get("window", 5)).mean()

        engine.register_feature(
            "custom_mean",
            FeatureDefinition(
                name="custom_mean",
                category="custom",
                description="Custom rolling mean",
                default_params={"window": 5},
            ),
            rolling_mean,
        )
        result = engine.calculate(df, features=["custom_mean"])
        assert "custom_mean" in result.columns

    def test_list_features(self, engine: FeatureEngine) -> None:
        features = engine.list_features()
        assert "rsi" in features
        assert "sma" in features
        assert isinstance(features["rsi"], FeatureDefinition)

    def test_registry_contains_all(self) -> None:
        assert "rsi" in FEATURE_REGISTRY
        assert "sma" in FEATURE_REGISTRY
        assert "atr" in FEATURE_REGISTRY