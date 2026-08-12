"""Broker-independent broker adapter interface (Step 16).

The adapter is the boundary between the execution engine and an actual (paper,
demo, or live) broker. Only the paper broker is implemented in this repository.
A real broker adapter must implement this interface without changing the
execution engine.

No broker credentials, HTTP calls, or live order submission belong here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.execution.models import (
    CancelRequest,
    ModifyRequest,
    OrderRequest,
    OrderResult,
    Position,
)
from app.risk.models import AccountState


class BrokerAdapter(ABC):
    """Abstract interface implemented by paper/demo/live brokers.

    The execution engine (``ExecutionEngine``) calls only these operations.
    Each adapter is responsible for its own deterministic order IDs, fills,
    position lifecycle, and account bookkeeping.
    """

    @abstractmethod
    def submit_order(self, request: OrderRequest) -> OrderResult:
        """Submit an order and return the execution outcome.

        ``request`` MUST carry a non-None ``risk_decision`` that is approved;
        a compliant adapter rejects (or raises) any order that bypassed the
        risk gate.
        """
        raise NotImplementedError

    @abstractmethod
    def cancel_order(self, request: CancelRequest) -> OrderResult:
        """Cancel a submitted-but-not-yet-filled order."""
        raise NotImplementedError

    @abstractmethod
    def modify_order(self, request: ModifyRequest) -> OrderResult:
        """Modify an order's stop-loss / take-profit."""
        raise NotImplementedError

    @abstractmethod
    def get_order(self, order_id: str) -> OrderResult | None:
        """Return the latest outcome for an order, or None if unknown."""
        raise NotImplementedError

    @abstractmethod
    def get_open_positions(self) -> list[Position]:
        """Return all currently open positions."""
        raise NotImplementedError

    @abstractmethod
    def get_account_state(self, mid_prices: dict[str, float] | None = None) -> AccountState:
        """Return the account snapshot required by the risk engine.

        ``mid_prices`` maps symbol -> current mid price so the broker can mark
        positions to market. Cross-currency conversions are the broker's
        responsibility; the engine must never invent rates.
        """
        raise NotImplementedError