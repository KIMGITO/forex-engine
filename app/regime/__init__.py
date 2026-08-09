"""Market regime detection engine.

Classifies the current market environment (trending/ranging/transition) from
measurable information in the data, feature, market-structure, and news layers.

This is an ANALYSIS layer. It does NOT generate trade signals or execute
trades, and it does NOT use machine learning yet.
"""

from app.regime.classifier import classify_regime
from app.regime.config import RegimeConfig
from app.regime.engine import RegimeEngine
from app.regime.models import (
    MarketRegime,
    MarketState,
    NewsRiskState,
    TrendState,
    VolatilityState,
)

__all__ = [
    "MarketRegime",
    "MarketState",
    "NewsRiskState",
    "RegimeConfig",
    "RegimeEngine",
    "TrendState",
    "VolatilityState",
    "classify_regime",
]
