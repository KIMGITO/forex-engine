"""Typed models for the strategy and signal layer.

Signals are RESEARCH INFORMATION, not orders. A signal must never be confused
with a trade. Signal strength is a documented categorical/deterministic score
(rule agreement), never a probability or a guarantee of profit.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SignalDirection(str, Enum):
    LONG = "long"
    SHORT = "short"


class SignalStrength(str, Enum):
    """Categorical rule-agreement strength (NOT probability)."""

    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"


class SignalStatus(str, Enum):
    """Signal lifecycle. A signal is research information, not an order."""

    DETECTED = "detected"
    CONFIRMED = "confirmed"
    INVALIDATED = "invalidated"
    EXPIRED = "expired"


class SignalReason(str, Enum):
    """Explicit, documented reasons a strategy may assert."""

    REGIME_SUPPORTS_TREND = "regime_supports_trend"
    STRUCTURE_BULLISH = "structure_bullish"
    STRUCTURE_BEARISH = "structure_bearish"
    HIGHER_HIGHS_HIGHER_LOWS = "higher_highs_higher_lows"
    LOWER_HIGHS_LOWER_LOWS = "lower_highs_lower_lows"
    DISPLACEMENT_BULLISH = "displacement_bullish"
    DISPLACEMENT_BEARISH = "displacement_bearish"
    VOLATILITY_ACCEPTABLE = "volatility_acceptable"
    NEWS_RISK_ACCEPTABLE = "news_risk_acceptable"
    LIQUIDITY_SWEEP_OCCURRED = "liquidity_sweep_occurred"
    PRICE_RETURN_THROUGH_LEVEL = "price_return_through_level"
    REGIME_NEUTRAL = "regime_neutral"
    STRUCTURE_MIXED = "structure_mixed"
    NEWS_RISK_PROHIBITED = "news_risk_prohibited"
    RANGE_PRESENT = "range_present"
    NO_DISPLACEMENT = "no_displacement"
    STRATEGY_COOLDOWN = "strategy_cooldown"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class Setup(BaseModel):
    """A deterministic setup identifier (for cooldown/duplicate protection)."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    timeframe: str
    direction: SignalDirection
    strategy: str
    anchor_timestamp: datetime = Field(
        ..., description="Timestamp of the setup's primary confirming event"
    )
    anchor_price: float = Field(
        ..., gt=0.0, description="Price at the primary confirming event"
    )
    context_key: str = Field(
        ..., description="Deterministic context fingerprint (e.g. level + sweep bar)"
    )

    def identity(self) -> str:
        """Deterministic unique identifier for cooldown/duplicate prevention."""
        return (
            f"{self.symbol}|{self.timeframe}|{self.strategy}|{self.direction.value}"
            f"|{self.context_key}"
        )


class Signal(BaseModel):
    """A single strategy-generated research signal."""

    model_config = ConfigDict(frozen=True)

    signal_id: str
    timestamp: datetime = Field(..., description="Bar timestamp when evaluated (tz-aware)")
    symbol: str
    timeframe: str
    direction: SignalDirection
    strength: SignalStrength = Field(
        ..., description="Categorical rule-agreement strength (NOT probability)"
    )
    score: float = Field(
        ...,
        ge=0.0,
        description="Deterministic rule-agreement score (0..max_score), NOT probability",
    )
    max_score: float = Field(..., gt=0.0)

    entry: float = Field(..., gt=0.0)
    stop_loss: float = Field(..., gt=0.0)
    take_profit: float = Field(..., gt=0.0)
    risk_distance: float = Field(..., gt=0.0)
    reward_distance: float = Field(..., gt=0.0)
    risk_reward_ratio: float = Field(..., ge=0.0)

    strategy: str
    regime: str | None = None
    market_state: str | None = None
    reasons: list[str] = Field(default_factory=list)
    structure_evidence: list[str] = Field(default_factory=list)
    news_risk_state: str | None = None
    setup: Setup | None = None
    status: SignalStatus = SignalStatus.DETECTED
    available_from: datetime = Field(
        ..., description="Earliest a consumer may legally use this signal"
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_direction_levels(self) -> "Signal":
        """Ensure stop/target are on the correct side of entry.

        LONG: stop < entry < target.
        SHORT: target < entry < stop.
        """
        if self.direction == SignalDirection.LONG:
            if not (self.stop_loss < self.entry < self.take_profit):
                raise ValueError("LONG signal requires stop < entry < target")
        else:
            if not (self.take_profit < self.entry < self.stop_loss):
                raise ValueError("SHORT signal requires target < entry < stop")
        return self

    @model_validator(mode="after")
    def _validate_risk_reward_distance(self) -> "Signal":
        if abs(self.entry - self.stop_loss) <= 0:
            raise ValueError("risk_distance must be > 0")
        if abs(self.take_profit - self.entry) <= 0:
            raise ValueError("reward_distance must be > 0")
        expected_rr = abs(self.take_profit - self.entry) / abs(self.entry - self.stop_loss)
        if abs(expected_rr - self.risk_reward_ratio) > 1e-9:
            raise ValueError("risk_reward_ratio inconsistent with entry/stop/target")
        return self