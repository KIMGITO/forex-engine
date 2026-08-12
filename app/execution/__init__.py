"""Broker-independent paper execution engine (Step 16).

Architecture:

    Signal -> ProposedTrade -> RiskEngine -> ExecutionEngine -> BrokerAdapter

This package provides:
- ``PaperBroker`` — deterministic in-memory broker (no real trades).
- ``ExecutionEngine`` — orchestrates the risk gate and broker submission.
- Strongly typed order/position/audit models.
"""

from app.execution.broker import BrokerAdapter
from app.execution.engine import ExecutionEngine
from app.execution.errors import (
    ExecutionError,
    InvalidOrderError,
    OrderNotFoundError,
    PositionNotFoundError,
    RiskGateViolationError,
)
from app.execution.models import (
    CancelRequest,
    ExecutionAudit,
    ExecutionPoint,
    ModifyRequest,
    OrderRequest,
    OrderResult,
    OrderStatus,
    OrderType,
    Position,
    PositionStatus,
    RejectionCode,
)
from app.execution.paper import PaperBroker

__all__ = [
    "BrokerAdapter",
    "CancelRequest",
    "ExecutionAudit",
    "ExecutionEngine",
    "ExecutionError",
    "ExecutionPoint",
    "InvalidOrderError",
    "ModifyRequest",
    "OrderNotFoundError",
    "OrderRequest",
    "OrderResult",
    "OrderStatus",
    "OrderType",
    "PaperBroker",
    "Position",
    "PositionNotFoundError",
    "PositionStatus",
    "RejectionCode",
    "RiskGateViolationError",
]