"""Tests for the regime models."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.regime.models import (
    MarketRegime,
    MarketState,
    NewsRiskState,
    TrendState,
    VolatilityState,
)


def _ts():
    return datetime(2024, 6, 10, 12, 0, tzinfo=timezone.utc)


def _make_regime(**overrides) -> MarketRegime:
    base = {
        "symbol": "EURUSD",
        "timeframe": "1h",
        "timestamp": _ts(),
        "trend_state": TrendState.BULLISH,
        "volatility_state": VolatilityState.NORMAL,
        "market_state": MarketState.TRENDING,
        "news_risk": NewsRiskState.CALM,
        "strength": 0.8,
        "metrics": {"atr_ratio": 0.004},
        "available_from": _ts(),
    }
    base.update(overrides)
    return MarketRegime(**base)


class TestMarketRegime:
    def test_valid(self) -> None:
        r = _make_regime()
        assert r.symbol == "EURUSD"
        assert r.trend_state == TrendState.BULLISH
        assert r.market_state == MarketState.TRENDING

    def test_strength_bounds(self) -> None:
        r = _make_regime(strength=1.0)
        assert r.strength == 1.0
        with pytest.raises(ValidationError):
            _make_regime(strength=1.5)
        with pytest.raises(ValidationError):
            _make_regime(strength=-0.1)

    def test_missing_required(self) -> None:
        # Omit an actually-required field (symbol) entirely.
        base = {
            "timeframe": "1h",
            "timestamp": _ts(),
            "trend_state": TrendState.BULLISH,
            "volatility_state": VolatilityState.NORMAL,
            "market_state": MarketState.TRENDING,
            "strength": 0.5,
            "available_from": _ts(),
        }
        with pytest.raises(ValidationError):
            MarketRegime(**base)
