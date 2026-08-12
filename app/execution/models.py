"""Typed models for the broker-independent execution layer (Step 16).

Architecture:

    Signal -> ProposedTrade -> RiskEngine -> ExecutionEngine -> BrokerAdapter

Models in this module are broker-independent. They describe orders, executions,
positions, and audit records. No broker credentials, API calls, or live order
submission ever appear here.

Conventions follow the existing repository: immutable Pydantic models, enums
instead of magic strings, explicit validation, and deterministic behavior.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.risk.models import PositionSide, RiskDecision


class OrderType(str, Enum):
    """Type of an order (market only for the paper broker)."""

    MARKET = "market"


class OrderStatus(str, Enum):
    """Lifecycle state of an order."""

    SUBMITTED = "submitted"
    FILLED = "filled"
    REJECTED = "rejected"
    CANCELLED = "cancelled"
    MODIFIED = "modified"


class PositionStatus(str, Enum):
    """Lifecycle state of an open position."""

    OPEN = "open"
    CLOSED = "closed"


class RejectionCode(str, Enum):
    """Structured reason an order was not executed."""

    INVALID_REQUEST = "invalid_request"
    INVALID_SYMBOL = "invalid_symbol"
    INVALID_QUANTITY = "invalid_quantity"
    INVALID_STOP = "invalid_stop"
    RISK_REJECTED = "risk_rejected"
    EMERGENCY_STOP = "emergency_stop"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    DRAWDOWN_LIMIT = "drawdown_limit"
    MAX_OPEN_POSITIONS = "max_open_positions"
    DUPLICATE_POSITION = "duplicate_position"
    PER_TRADE_RISK = "per_trade_risk"
    POSITION_NOT_FOUND = "position_not_found"
    ORDER_NOT_FOUND = "order_not_found"


class OrderRequest(BaseModel):
    """A market order request submitted by the execution engine.

    ``risk_decision`` is populated by the ExecutionEngine AFTER the risk gate
    runs; the BrokerAdapter must never accept a request that bypassed risk.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    order_id: str = Field(..., description="Deterministic broker-generated order ID")
    client_order_id: str | None = Field(
        default=None, description="Caller correlation ID (e.g. signal_id)"
    )
    symbol: str = Field(..., description="Normalised 6-letter FX symbol")
    side: PositionSide
    quantity: float = Field(..., gt=0.0, description="Quantity in base units")
    requested_price: float = Field(
        ..., gt=0.0, description="Reference/market price at submission"
    )
    stop_loss: float | None = Field(default=None, gt=0.0)
    take_profit: float | None = Field(default=None, gt=0.0)
    timestamp: datetime = Field(..., description="Order submission time (tz-aware)")
    risk_decision: RiskDecision | None = Field(
        default=None,
        description="RiskEngine decision proving the gate was passed",
    )


class OrderResult(BaseModel):
    """Outcome of submitting/cancelling/modifying an order."""

    model_config = ConfigDict(frozen=True)

    order_id: str
    status: OrderStatus
    symbol: str
    side: PositionSide
    quantity: float
    requested_price: float
    executed_price: float | None = Field(default=None, gt=0.0)
    executed_at: datetime | None = None
    rejection_code: RejectionCode | None = None
    rejection_reason: str | None = None
    position_id: str | None = Field(
        default=None, description="Position ID created by a fill (if any)"
    )
    timestamp: datetime


class CancelRequest(BaseModel):
    """Request to cancel an order."""

    model_config = ConfigDict(frozen=True)

    order_id: str
    timestamp: datetime


class ModifyRequest(BaseModel):
    """Request to modify an order's stop-loss / take-profit."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    order_id: str
    stop_loss: float | None = Field(default=None, gt=0.0)
    take_profit: float | None = Field(default=None, gt=0.0)
    timestamp: datetime

    @model_validator(mode="after")
    def _at_least_one_change(self) -> ModifyRequest:
        if self.stop_loss is None and self.take_profit is None:
            raise ValueError("modify request must change stop_loss or take_profit")
        return self


class ExecutionPoint(BaseModel):
    """A point on the filled-order audit trail (one per lifecycle event)."""

    model_config = ConfigDict(frozen=True)

    timestamp: datetime
    event: str  # "submit" | "fill" | "reject" | "cancel" | "modify"
    order_id: str
    symbol: str
    side: PositionSide
    quantity: float
    requested_price: float
    executed_price: float | None = None
    risk_approved: bool | None = None
    status: OrderStatus
    rejection_code: RejectionCode | None = None
    rejection_reason: str | None = None
    position_id: str | None = None


class ExecutionAudit(BaseModel):
    """Audit ledger entry produced by ExecutionEngine for every order."""

    model_config = ConfigDict(frozen=True)

    timestamp: datetime
    order_id: str
    client_order_id: str | None = None
    symbol: str
    side: PositionSide
    quantity: float
    requested_price: float
    executed_price: float | None = None
    risk_decision: RiskDecision | None = None
    status: OrderStatus
    rejection_code: RejectionCode | None = None
    rejection_reason: str | None = None
    position_id: str | None = None


class Position(BaseModel):
    """An open or recently closed position held by the paper broker."""

    model_config = ConfigDict(frozen=True)

    position_id: str
    order_id: str
    client_order_id: str | None = None
    symbol: str
    side: PositionSide
    quantity: float = Field(..., gt=0.0)
    entry_price: float = Field(..., gt=0.0)
    stop_loss: float | None = Field(default=None, gt=0.0)
    take_profit: float | None = Field(default=None, gt=0.0)
    opened_at: datetime
    status: PositionStatus = PositionStatus.OPEN
    closed_at: datetime | None = None
    realized_pnl: float = 0.0
    closed_quantity: float = 0.0

    def dict_for_risk(self) -> dict[str, Any]:
        """Shape required by app.risk.models.AccountState.open_positions."""
        return {
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
        }