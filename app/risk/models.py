"""Typed domain models for the broker-independent Risk Management Engine.

The risk engine sits between a generated signal and the order/execution layer:

    Signal -> Risk Engine -> RiskDecision (approved/rejected) -> Order

It never decides whether a signal is profitable. Its only responsibility is
controlling exposure and deciding whether a proposed trade is allowed.

Design notes
------------
* Every model is immutable (``frozen=True``), matching the backend/strategy
  layer conventions and preventing accidental mutation of risk state.
* All monetary values are plain ``float`` (the existing backtest/portfolio
  layer uses float throughout). Determinism is guaranteed by the engine
  performing identical arithmetic on identical inputs, not by Decimal (the
  project has no Decimal usage). Precision-sensitive quantisation (lot size,
  price rounding) is delegated to the instrument specification, never to the
  core risk rules.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.backtest.models import OrderSide


class RiskDecisionType(str, Enum):
    """Outcome of a risk decision."""

    APPROVED = "approved"
    REJECTED = "rejected"


class RejectionReason(str, Enum):
    """Structured rejection reason (never an arbitrary string)."""

    INVALID_TRADE = "invalid_trade"
    INVALID_INSTRUMENT = "invalid_instrument"
    MISSING_STOP_LOSS = "missing_stop_loss"
    STOP_ON_WRONG_SIDE = "stop_on_wrong_side"
    POSITION_SIZE_TOO_SMALL = "position_size_too_small"
    POSITION_SIZE_TOO_LARGE = "position_size_too_large"
    ZERO_STOP_DISTANCE = "zero_stop_distance"
    PER_TRADE_RISK_EXCEEDED = "per_trade_risk_exceeded"
    DAILY_LOSS_LIMIT_EXCEEDED = "daily_loss_limit_exceeded"
    DRAWDOWN_LIMIT_EXCEEDED = "drawdown_limit_exceeded"
    MAX_OPEN_POSITIONS_REACHED = "max_open_positions_reached"
    SYMBOL_EXPOSURE_EXCEEDED = "symbol_exposure_exceeded"
    TOTAL_EXPOSURE_EXCEEDED = "total_exposure_exceeded"
    DUPLICATE_POSITION = "duplicate_position"
    EMERGENCY_STOP = "emergency_stop"
    INSUFFICIENT_MARGIN = "insufficient_margin"
    EXPOSURE_GROUP_EXCEEDED = "exposure_group_exceeded"
    ACCOUNT_INVALID = "account_invalid"
    RISK_CONFIG_INVALID = "risk_config_invalid"


class PositionSide(str, Enum):
    """Side of a proposed trade (mirrors backtest OrderSide without coupling)."""

    BUY = "buy"
    SELL = "sell"

    @classmethod
    def from_order_side(cls, side: OrderSide) -> "PositionSide":
        return cls.BUY if side == OrderSide.BUY else cls.SELL


class ExposureGroup(str, Enum):
    """Conservative currency-exposure groups.

    This is a *conservative* grouping mechanism based on the base/quote
    currency of each pair. It is deliberately NOT a statistical correlation
    model — currency grouping is not equivalent to historical correlation.
    True statistical correlation comes from ``app.features.correlation``;
    the risk engine only uses these coarse groups as a conservative
    exposure cap when the caller opts in (``max_exposure_per_group``).
    """

    USD_QUOTE = "usd_quote"      # USD is the quote (EURUSD, GBPUSD, AUDUSD, USDCAD)
    USD_BASE_JPY = "usd_base_jpy"  # USD base, JPY quote (USDJPY)
    USD_BASE_CHF = "usd_base_chf"  # USD base, CHF quote (USDCHF)
    UNKNOWN = "unknown"

    @classmethod
    def from_symbol(cls, symbol: str) -> "ExposureGroup":
        """Derive the exposure group for a 6-letter FX symbol.

        Only the six configured pairs are recognised; anything else maps to
        UNKNOWN and is subject to an explicit ``allow_unknown_symbols``
        configuration flag.
        """
        s = symbol.upper().replace("/", "").replace("_", "")
        if len(s) != 6:
            return cls.UNKNOWN
        base, quote = s[:3], s[3:]
        if quote == "JPY":
            return cls.USD_BASE_JPY
        if quote == "CHF":
            return cls.USD_BASE_CHF
        if quote == "USD":
            return cls.USD_QUOTE
        return cls.UNKNOWN


class AccountState(BaseModel):
    """Account snapshot required by the risk engine.

    Only fields the engine needs are present. No broker-specific fields.
    """

    model_config = ConfigDict(frozen=True)

    balance: float = Field(..., description="Current account balance (account currency)")
    equity: float = Field(..., ge=0.0, description="Current equity = balance + unrealized P&L")
    used_margin: float = Field(default=0.0, ge=0.0)
    available_margin: float = Field(default=0.0, ge=0.0)
    peak_equity: float = Field(default=0.0, ge=0.0)
    daily_pnl: float = Field(default=0.0, description="Realized+unrealized P&L since start of day")
    open_positions: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Open positions as lightweight dicts: each must contain "
            "``symbol``, ``side`` ('buy'|'sell'), ``quantity``, and "
            "``entry_price``. Kept untyped to avoid coupling to the backtest "
            "Position model."
        ),
    )
    drawdown_pct: float | None = Field(
        default=None,
        ge=0.0,
        description="Current drawdown as a fraction of peak equity (0..1). If None, computed.",
    )
    exposure: float | None = Field(
        default=None,
        ge=0.0,
        description="Current total notional exposure in account currency. If None, computed.",
    )


class ProposedTrade(BaseModel):
    """A proposed trade from a signal/strategy, before risk screening."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    side: PositionSide
    entry_price: float = Field(..., gt=0.0)
    stop_loss: float = Field(..., gt=0.0)
    take_profit: float | None = Field(default=None, gt=0.0)
    timeframe: str | None = None
    quantity: float | None = Field(
        default=None, gt=0.0, description="Optional pre-supplied lot/unit size"
    )
    signal_id: str | None = None

    @property
    def risk_distance(self) -> float:
        """Absolute distance from entry to stop (validation is engine-side)."""
        return abs(self.entry_price - self.stop_loss)


class RiskDecision(BaseModel):
    """Result of the risk engine evaluation."""

    model_config = ConfigDict(frozen=True)

    type: RiskDecisionType
    reason: RejectionReason | None = None
    message: str = ""
    position_size: float | None = Field(default=None, ge=0.0)
    monetary_risk: float | None = Field(default=None, ge=0.0)
    risk_percent: float | None = Field(default=None, ge=0.0)
    exposure_after: float | None = Field(default=None, ge=0.0)
    limits: dict[str, Any] = Field(default_factory=dict)

    @property
    def approved(self) -> bool:
        return self.type == RiskDecisionType.APPROVED