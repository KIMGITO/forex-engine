"""Typed models for the multi-timeframe research engine.

Every MTF observation carries an explicit ``available_from`` timestamp. A
lower-timeframe observation may ONLY use higher-timeframe context whose
candle was fully completed before the observation moment.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class MtfAlignmentState(str, Enum):
    ALIGNED_LONG = "aligned_long"
    ALIGNED_SHORT = "aligned_short"
    CONFLICTED = "conflicted"
    UNKNOWN = "unknown"


class MtfWindow(BaseModel):
    """A single aligned (possibly missing) higher-timeframe candle window."""

    model_config = ConfigDict(frozen=True)

    timeframe: str
    open: datetime = Field(..., description="UTC open of the candle slot (tz-aware)")
    close: datetime = Field(..., description="UTC close = available_from (tz-aware)")
    available_from: datetime = Field(
        ..., description="Earliest a consumer may legally use this candle"
    )
    present: bool = Field(
        default=False, description="True when an actual candle exists in this slot"
    )


class TimeframeContext(BaseModel):
    """Per-timeframe analytical context available at an observation moment."""

    model_config = ConfigDict(frozen=True)

    timeframe: str
    timestamp: datetime = Field(..., description="Observation moment (UTC, tz-aware)")
    candle_open: datetime | None = Field(
        default=None, description="Open of the completed candle used (UTC)"
    )
    candle_close: datetime | None = Field(
        default=None, description="Close (= available_from) of the completed candle (UTC)"
    )

    trend_state: str | None = None
    volatility_state: str | None = None
    market_state: str | None = None
    structural_bias: str | None = Field(
        default=None, description="'bullish' | 'bearish' | 'neutral' | None"
    )
    liquidity_zones: list[Any] = Field(default_factory=list)
    sweeps: list[Any] = Field(default_factory=list)
    news_risk_max: str | None = Field(
        default=None, description="Highest active news-risk state at timestamp"
    )

    strength: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Alignment agreement (NOT probability)",
    )
    present: bool = Field(
        default=True,
        description="False when the timeframe had no completed candle at timestamp",
    )
    available_from: datetime = Field(
        ..., description="Earliest a consumer may legally use this context"
    )


class MtfContext(BaseModel):
    """Unified multi-timeframe context for a single observation moment."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    base_timeframe: str
    timestamp: datetime = Field(..., description="Base observation moment (UTC)")
    hierarchy: list[TimeframeContext] = Field(default_factory=list)
    alignment: MtfAlignmentState = MtfAlignmentState.UNKNOWN
    alignment_reasons: list[str] = Field(default_factory=list)
    min_aligned: float = 0.0
    news_risk_max: str | None = None
    metadata: dict = Field(default_factory=dict)
    available_from: datetime = Field(
        ..., description="Earliest a consumer may legally use this MTF context"
    )