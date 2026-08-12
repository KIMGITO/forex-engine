"""Execution engine (Step 16).

The execution engine is the only component allowed to touch the broker
adapter. It is a pure orchestrator:

    1. Convert a ProposedTrade into an OrderRequest (deterministic order ID).
    2. ALWAYS call RiskEngine.evaluate(...) first.
    3. REJECTED  -> ExecutionAudit(REJECTED) and never reach the broker.
    4. APPROVED  -> BrokerAdapter.submit_order(...) and record the audit.

The execution engine contains NO risk rules; it only invokes the risk engine.
It never places real trades and never knows about broker credentials.

Architecture:
    Signal -> ProposedTrade -> RiskEngine -> ExecutionEngine -> BrokerAdapter
"""

from __future__ import annotations

from datetime import datetime, timezone

from app.execution.errors import ExecutionError
from app.execution.models import (
    OrderRequest,
    OrderResult,
    OrderStatus,
    RejectionCode,
)
from app.risk.engine import RiskEngine
from app.risk.models import (
    AccountState,
    ProposedTrade,
    RejectionReason,
    RiskDecision,
)


class ExecutionEngine:
    """Orchestrates risk gating and broker submission.

    Parameters
    ----------
    risk_engine : RiskEngine
        The risk gate. Must never be bypassed.
    broker : BrokerAdapter
        Any adapter implementing the paper/demo/live interface.
    id_prefix : str
        Prefix for deterministic order IDs (default "ord").
    """

    def __init__(
        self,
        risk_engine: RiskEngine,
        broker,
        id_prefix: str = "ord",
    ) -> None:
        self.risk = risk_engine
        self.broker = broker
        self._counter = 0
        self._prefix = id_prefix
        # Caller-supplied mid prices for marking positions (paper broker).
        self._mid_prices: dict[str, float] = {}
        self._audit: list[dict] = []

    # ── Deterministic order IDs ───────────────────────────────────────────────

    def _next_order_id(self) -> str:
        self._counter += 1
        return f"{self._prefix}-{self._counter}"

    # ── Price marking ─────────────────────────────────────────────────────────

    def set_mid_prices(self, prices: dict[str, float]) -> None:
        """Supply latest mid prices for marking positions to market."""
        self._mid_prices.update(prices)

    # ── Public API ────────────────────────────────────────────────────────────

    def place_order(
        self,
        trade: ProposedTrade,
        account: AccountState,
        timestamp: datetime | None = None,
    ) -> dict:
        """Submit a proposed trade through the risk gate and broker.

        Returns a dictionary containing:
          - ``order_id``
          - ``risk_decision`` (the RiskDecision object)
          - ``order_result`` (OrderResult from the broker, or None when
            rejected by risk)
          - ``approved`` (bool)
          - ``rejection_code`` / ``rejection_reason`` when rejected
          - ``position_id`` when filled

        The risk gate is ALWAYS consulted before the broker sees the order.
        """
        now = timestamp or datetime.now(timezone.utc)
        order_id = self._next_order_id()

        # 1. Risk gate (must never be bypassed).
        decision = self.risk.evaluate(trade, account)
        if not decision.approved:
            code = self._rejection_code_for(decision.reason)
            entry = self._audit_entry(
                order_id=order_id,
                trade=trade,
                decision=decision,
                status=OrderStatus.REJECTED,
                rejection_code=code,
                rejection_reason=(
                    decision.message
                    or (decision.reason.value if decision.reason else "risk rejected")
                ),
                now=now,
            )
            self._audit.append(entry)
            return {
                "order_id": order_id,
                "risk_decision": decision,
                "order_result": None,
                "approved": False,
                "rejection_code": code.value,
                "rejection_reason": entry["rejection_reason"],
                "position_id": None,
            }

        # 2. Build the approved order request.
        quantity = trade.quantity if trade.quantity is not None else (
            decision.position_size or 0.0
        )
        if quantity <= 0:
            entry = self._audit_entry(
                order_id=order_id,
                trade=trade,
                decision=decision,
                status=OrderStatus.REJECTED,
                rejection_code=RejectionCode.INVALID_QUANTITY,
                rejection_reason="approved risk decision produced no position size",
                now=now,
            )
            self._audit.append(entry)
            return {
                "order_id": order_id,
                "risk_decision": decision,
                "order_result": None,
                "approved": False,
                "rejection_code": RejectionCode.INVALID_QUANTITY.value,
                "rejection_reason": entry["rejection_reason"],
                "position_id": None,
            }

        request = OrderRequest(
            order_id=order_id,
            client_order_id=trade.signal_id,
            symbol=trade.symbol,
            side=trade.side,
            quantity=quantity,
            requested_price=trade.entry_price,
            stop_loss=trade.stop_loss,
            take_profit=trade.take_profit,
            timestamp=now,
            risk_decision=decision,
        )

        # 3. Broker submission.
        try:
            result: OrderResult = self.broker.submit_order(request)
        except ExecutionError as exc:
            entry = self._audit_entry(
                order_id=order_id,
                trade=trade,
                decision=decision,
                status=OrderStatus.REJECTED,
                rejection_code=RejectionCode.INVALID_REQUEST,
                rejection_reason=str(exc),
                now=now,
            )
            self._audit.append(entry)
            return {
                "order_id": order_id,
                "risk_decision": decision,
                "order_result": None,
                "approved": False,
                "rejection_code": RejectionCode.INVALID_REQUEST.value,
                "rejection_reason": str(exc),
                "position_id": None,
            }

        # 4. Audit.
        entry = self._audit_entry(
            order_id=order_id,
            trade=trade,
            decision=decision,
            status=result.status,
            rejection_code=result.rejection_code,
            rejection_reason=result.rejection_reason,
            executed_price=result.executed_price,
            position_id=result.position_id,
            now=now,
        )
        self._audit.append(entry)

        return {
            "order_id": order_id,
            "risk_decision": decision,
            "order_result": result,
            "approved": result.status == OrderStatus.FILLED and result.rejection_code is None,
            "rejection_code": (
                result.rejection_code.value if result.rejection_code else None
            ),
            "rejection_reason": result.rejection_reason,
            "position_id": result.position_id,
        }

    def close_position(self, position_id: str, exit_price: float, timestamp: datetime) -> dict:
        """Close a position through the broker."""
        pos = self.broker.close_position(position_id, exit_price, timestamp)
        return {
            "position_id": position_id,
            "closed": pos.status.value == "closed",
            "realized_pnl": pos.realized_pnl,
            "position_status": pos.status.value,
        }

    def get_account_state(self) -> AccountState:
        """Return the broker's current account state (for the risk gate)."""
        return self.broker.get_account_state(self._mid_prices)

    def get_audit(self) -> list[dict]:
        """Return a copy of the execution audit trail (oldest first)."""
        return list(self._audit)

    # ── Internals ─────────────────────────────────────────────────────────────

    @staticmethod
    def _rejection_code_for(reason: RejectionReason | None) -> RejectionCode:
        """Map a risk rejection reason to a structured execution code."""
        if reason is RejectionReason.EMERGENCY_STOP:
            return RejectionCode.EMERGENCY_STOP
        if reason is RejectionReason.DAILY_LOSS_LIMIT_EXCEEDED:
            return RejectionCode.DAILY_LOSS_LIMIT
        if reason is RejectionReason.DRAWDOWN_LIMIT_EXCEEDED:
            return RejectionCode.DRAWDOWN_LIMIT
        if reason is RejectionReason.MAX_OPEN_POSITIONS_REACHED:
            return RejectionCode.MAX_OPEN_POSITIONS
        if reason is RejectionReason.DUPLICATE_POSITION:
            return RejectionCode.DUPLICATE_POSITION
        if reason is RejectionReason.PER_TRADE_RISK_EXCEEDED:
            return RejectionCode.PER_TRADE_RISK
        if reason in (
            RejectionReason.POSITION_SIZE_TOO_LARGE,
            RejectionReason.POSITION_SIZE_TOO_SMALL,
        ):
            return RejectionCode.INVALID_QUANTITY
        return RejectionCode.RISK_REJECTED

    def _audit_entry(
        self,
        order_id: str,
        trade: ProposedTrade,
        decision: RiskDecision,
        status: OrderStatus,
        now: datetime,
        rejection_code: RejectionCode | None = None,
        rejection_reason: str | None = None,
        executed_price: float | None = None,
        position_id: str | None = None,
    ) -> dict:
        return {
            "timestamp": now,
            "order_id": order_id,
            "client_order_id": trade.signal_id,
            "symbol": trade.symbol,
            "side": trade.side.value,
            "quantity": trade.quantity if trade.quantity is not None else (decision.position_size or 0.0),
            "requested_price": trade.entry_price,
            "executed_price": executed_price,
            "risk_decision": decision,
            "status": status.value,
            "rejection_code": rejection_code.value if rejection_code else None,
            "rejection_reason": rejection_reason,
            "position_id": position_id,
        }