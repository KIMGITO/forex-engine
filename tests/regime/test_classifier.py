"""Tests for the multi-factor classifier."""

from datetime import datetime, timezone

from app.regime.classifier import classify_regime
from app.regime.models import MarketState, NewsRiskState, TrendState, VolatilityState


def _ts():
    return datetime(2024, 6, 10, 12, 0, tzinfo=timezone.utc)


class TestClassifier:
    def test_trending_bullish_high_vol(self) -> None:
        """spec: strong bullish trend + HH/HL structure + high vol + no range."""
        regime = classify_regime(
            symbol="EURUSD", timeframe="1h", timestamp=_ts(),
            trend=TrendState.BULLISH,
            volatility=VolatilityState.HIGH,
            structure_bias=0.8, structure_count=5,
            range_active=False,
            transition_vol_ratio=1.1,
        )
        assert regime.market_state == MarketState.TRENDING
        assert regime.trend_state == TrendState.BULLISH
        assert regime.volatility_state == VolatilityState.HIGH

    def test_ranging_neutral_low_vol(self) -> None:
        """spec: weak trend + mixed structure + low vol + active range."""
        regime = classify_regime(
            symbol="EURUSD", timeframe="1h", timestamp=_ts(),
            trend=TrendState.NEUTRAL,
            volatility=VolatilityState.LOW,
            structure_bias=0.0, structure_count=4,
            range_active=True,
            transition_vol_ratio=1.0,
        )
        assert regime.market_state == MarketState.RANGING
        assert regime.trend_state == TrendState.NEUTRAL
        assert regime.volatility_state == VolatilityState.LOW

    def test_structure_agrees_with_trend(self) -> None:
        regime = classify_regime(
            symbol="EURUSD", timeframe="1h", timestamp=_ts(),
            trend=TrendState.BEARISH,
            volatility=VolatilityState.NORMAL,
            structure_bias=-0.75, structure_count=6,
            range_active=False,
            transition_vol_ratio=1.0,
        )
        assert regime.market_state == MarketState.TRENDING
        assert regime.strength > 0.5

    def test_structure_conflicts_with_trend_transition(self) -> None:
        regime = classify_regime(
            symbol="EURUSD", timeframe="1h", timestamp=_ts(),
            trend=TrendState.BULLISH,
            volatility=VolatilityState.NORMAL,
            structure_bias=-0.6, structure_count=4,
            range_active=False,
            transition_vol_ratio=1.0,
        )
        assert regime.market_state == MarketState.TRANSITION

    def test_volatility_expansion_transition(self) -> None:
        regime = classify_regime(
            symbol="EURUSD", timeframe="1h", timestamp=_ts(),
            trend=TrendState.BULLISH,
            volatility=VolatilityState.EXTREME,
            structure_bias=0.5, structure_count=4,
            range_active=False,
            transition_vol_ratio=1.9,
        )
        assert regime.market_state == MarketState.TRANSITION

    def test_unknown_when_insufficient(self) -> None:
        regime = classify_regime(
            symbol="EURUSD", timeframe="1h", timestamp=_ts(),
            trend=TrendState.UNKNOWN,
            volatility=VolatilityState.UNKNOWN,
            structure_bias=None, structure_count=0,
            range_active=False,
            transition_vol_ratio=float("nan"),
        )
        assert regime.market_state == MarketState.UNKNOWN

    def test_news_risk_metadata_only(self) -> None:
        """Active high-impact news must not change the direction."""
        regime = classify_regime(
            symbol="EURUSD", timeframe="1h", timestamp=_ts(),
            trend=TrendState.BULLISH,
            volatility=VolatilityState.NORMAL,
            structure_bias=0.5, structure_count=4,
            range_active=False,
            transition_vol_ratio=1.0,
            news_risk=NewsRiskState.ACTIVE_HIGH,
        )
        # Direction unchanged; news is metadata.
        assert regime.trend_state == TrendState.BULLISH
        assert regime.news_risk == NewsRiskState.ACTIVE_HIGH
