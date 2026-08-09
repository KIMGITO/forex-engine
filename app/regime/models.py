"""Typed models for the market-regime detection engine.

Regime classification is a *descriptive model* of the current environment. It
does NOT guarantee future price behavior and is NOT a trading signal. The
strategy/risk layers decide what to do with this information.
"""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class TrendState(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"


class VolatilityState(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    EXTREME = "extreme"
    UNKNOWN = "unknown"


class MarketState(str, Enum):
    TRENDING = "trending"
    RANGING = "ranging"
    TRANSITION = "transition"
    UNKNOWN = "unknown"


class NewsRiskState(str, Enum):
    """Contextual event-risk metadata only.

    Never a directional bias: a high-impact event window raises *uncertainty*,
    it does not imply the market will move up or down.
    """

    CALM = "calm"
    ACTIVE_MEDIUM = "active_medium"
    ACTIVE_HIGH = "active_high"
    UNKNOWN = "unknown"


class MarketRegime(BaseModel):
    """A single regime observation for a symbol/timeframe at a timestamp.

    ``strength`` is an objective 0..1 internal-agreement score (how many
    available factor signals agree with the classification). It is NOT a
    probability and NOT a prediction.
    """

    model_config = ConfigDict(frozen=True)

    symbol: str
    timeframe: str
    timestamp: datetime = Field(..., description="Bar this regime applies to (tz-aware)")
    trend_state: TrendState
    volatility_state: VolatilityState
    market_state: MarketState
    news_risk: NewsRiskState = Field(default=NewsRiskState.UNKNOWN)
    strength: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Objective internal-agreement score (NOT a probability)",
    )
    metrics: dict[str, float] = Field(
        default_factory=dict,
        description="Objective inputs that produced this classification",
    )
    available_from: datetime = Field(
        ..., description="Earliest a consumer may legally use this regime observation"
    )
