"""In-memory PaperBroker (Step 16).

Deliberately simple and deterministic:

* Market orders fill immediately at the requested price (no spread/slippage
  simulation; the paper broker is a test harness, not an FX model).
* Order/position IDs are deterministic: ``ord-<n>`` / ``pos-<n>``.
* Positions are tracked per position_id; partial close is supported.
* SL/TP are resolved by the ExecutionEngine against caller-supplied ticks via
  :meth:`mark_position_sl_tp`; the broker never invents prices.
* Unrealized PnL requires the caller to pass mid prices; symbols without a
  provided price are marked at entry (zero unrealized PnL).
"""

from __future__ import annotations

from datetime import datetime

from app.execution.errors import (
    InvalidOrderError,
    OrderNotFoundError,
    PositionNotFoundError,
    RiskGateViolationError,
)
from app.execution.models import (
    CancelRequest,
    ModifyRequest,
    OrderRequest,
    OrderResult,
    OrderStatus,
    Position,
    PositionStatus,
    RejectionCode,
)
from app.risk.models import AccountState, PositionSide


class PaperBroker:
    """Deterministic in-memory paper broker implementing :class:`BrokerAdapter`.

    Parameters
    ----------
    initial_balance : float
        Starting account balance in account currency (must be > 0).
    account_currency : str
        Account denomination (default USD). Used only in reports.
    quote_to_account : dict[str, float]
        Optional symbol -> quote/account conversion for JPY/CHF pairs. When
        absent, USD-quote pairs use 1.0; cross-currency pairs without a factor
        are marked at entry (zero unrealized PnL) rather than guessed.
    """

    def __init__(
        self,
        initial_balance: float = 10_000.0,
        account_currency: str = "USD",
        quote_to_account: dict[str, float] | None = None,
    ) -> None:
        if initial_balance <= 0:
            raise ValueError("initial_balance must be > 0")
        self.initial_balance = initial_balance
        self.account_currency = account_currency
        self.balance = initial_balance
        self.quote_to_account = quote_to_account or {}

        self._orders: dict[str, OrderResult] = {}
        self._positions: dict[str, Position] = {}
        self._order_counter = 0
        self._position_counter = 0
        self._peak_equity = initial_balance
        self._daily_start_equity = initial_balance

    # ── ID generation (deterministic) ─────────────────────────────────────────

    def _next_order_id(self) -> str:
        self._order_counter += 1
        return f"ord-{self._order_counter}"

    def _next_position_id(self) -> str:
        self._position_counter += 1
        return f"pos-{self._position_counter}"

    # ── BrokerAdapter interface ───────────────────────────────────────────────

    def submit_order(self, request: OrderRequest) -> OrderResult:
        """Fill a market order immediately at the requested price.

        Rejects structurally invalid requests and any request that did not pass
        the risk gate (missing or non-approved risk decision).
        """
        if request.risk_decision is None or not request.risk_decision.approved:
            raise RiskGateViolationError(
                "paper broker refuses an order without an approved risk decision"
            )

        err = self._validate_request(request)
        if err is not None:
            code, reason = err
            result = self._reject(request, code, reason)
            self._orders[request.order_id] = result
            return result

        fill_price = request.requested_price
        position_id = self._next_position_id()
        now = request.timestamp

        self._positions[position_id] = Position(
            position_id=position_id,
            order_id=request.order_id,
            client_order_id=request.client_order_id,
            symbol=request.symbol,
            side=request.side,
            quantity=request.quantity,
            entry_price=fill_price,
            stop_loss=request.stop_loss,
            take_profit=request.take_profit,
            opened_at=now,
        )

        result = OrderResult(
            order_id=request.order_id,
            status=OrderStatus.FILLED,
            symbol=request.symbol,
            side=request.side,
            quantity=request.quantity,
            requested_price=request.requested_price,
            executed_price=fill_price,
            executed_at=now,
            position_id=position_id,
            timestamp=now,
        )
        self._orders[request.order_id] = result
        return result

    def cancel_order(self, request: CancelRequest) -> OrderResult:
        existing = self._orders.get(request.order_id)
        if existing is None:
            raise OrderNotFoundError(f"order {request.order_id} not found")
        if existing.status == OrderStatus.FILLED:
            return OrderResult(
                order_id=request.order_id,
                status=OrderStatus.CANCELLED,
                symbol=existing.symbol,
                side=existing.side,
                quantity=existing.quantity,
                requested_price=existing.requested_price,
                executed_price=existing.executed_price,
                rejection_code=RejectionCode.INVALID_REQUEST,
                rejection_reason="order already filled; cannot cancel a filled order",
                timestamp=request.timestamp,
            )
        return existing

    def modify_order(self, request: ModifyRequest) -> OrderResult:
        existing = self._orders.get(request.order_id)
        if existing is None:
            raise OrderNotFoundError(f"order {request.order_id} not found")
        if existing.status != OrderStatus.FILLED or existing.position_id is None:
            return OrderResult(
                order_id=request.order_id,
                status=OrderStatus.MODIFIED,
                symbol=existing.symbol,
                side=existing.side,
                quantity=existing.quantity,
                requested_price=existing.requested_price,
                executed_price=existing.executed_price,
                rejection_code=RejectionCode.INVALID_REQUEST,
                rejection_reason="only filled orders can be modified (SL/TP update)",
                timestamp=request.timestamp,
            )

        pos = self._positions.get(existing.position_id)
        if pos is None:
            raise PositionNotFoundError(f"position for order {request.order_id} not found")
        if request.stop_loss is not None and not self._stop_on_correct_side(
            pos.side, pos.entry_price, request.stop_loss
        ):
            return OrderResult(
                order_id=request.order_id,
                status=OrderStatus.MODIFIED,
                symbol=pos.symbol,
                side=pos.side,
                quantity=pos.quantity,
                requested_price=existing.requested_price,
                executed_price=existing.executed_price,
                rejection_code=RejectionCode.INVALID_STOP,
                rejection_reason="modified stop-loss is on the wrong side of entry",
                timestamp=request.timestamp,
            )

        new_sl = request.stop_loss if request.stop_loss is not None else pos.stop_loss
        new_tp = request.take_profit if request.take_profit is not None else pos.take_profit
        self._positions[existing.position_id] = pos.model_copy(
            update={"stop_loss": new_sl, "take_profit": new_tp}
        )
        return OrderResult(
            order_id=request.order_id,
            status=OrderStatus.MODIFIED,
            symbol=pos.symbol,
            side=pos.side,
            quantity=pos.quantity,
            requested_price=existing.requested_price,
            executed_price=existing.executed_price,
            position_id=existing.position_id,
            timestamp=request.timestamp,
        )

    def get_order(self, order_id: str) -> OrderResult | None:
        return self._orders.get(order_id)

    def get_open_positions(self) -> list[Position]:
        return [p for p in self._positions.values() if p.status == PositionStatus.OPEN]

    def get_account_state(self, mid_prices: dict[str, float] | None = None) -> AccountState:
        mid_prices = mid_prices or {}
        unrealized = 0.0
        open_positions: list[dict] = []
        total_used_margin = 0.0
        for pos in self.get_open_positions():
            open_positions.append(pos.dict_for_risk())
            mid = mid_prices.get(pos.symbol)
            if mid is not None:
                factor = self.quote_to_account.get(pos.symbol, 1.0)
                if pos.side == PositionSide.BUY:
                    unrealized += (mid - pos.entry_price) * pos.quantity * factor
                else:
                    unrealized += (pos.entry_price - mid) * pos.quantity * factor
            total_used_margin += pos.quantity * pos.entry_price / 30.0

        equity = self.balance + unrealized
        self._peak_equity = max(self._peak_equity, equity)
        drawdown = (
            (self._peak_equity - equity) / self._peak_equity if self._peak_equity > 0 else 0.0
        )
        return AccountState(
            balance=self.balance,
            equity=equity,
            used_margin=total_used_margin,
            available_margin=max(0.0, equity - total_used_margin),
            peak_equity=self._peak_equity,
            daily_pnl=equity - self._daily_start_equity,
            open_positions=open_positions,
            drawdown_pct=drawdown,
            exposure=sum(p.quantity * p.entry_price for p in self.get_open_positions()),
        )

    # ── Position lifecycle helpers ────────────────────────────────────────────

    def close_position(
        self,
        position_id: str,
        exit_price: float,
        exit_time: datetime,
        quantity: float | None = None,
    ) -> Position:
        """Close (fully or partially) an open position and realize PnL."""
        pos = self._positions.get(position_id)
        if pos is None:
            raise PositionNotFoundError(f"position {position_id} not found")
        if pos.status != PositionStatus.OPEN:
            raise InvalidOrderError(f"position {position_id} is not open")

        close_qty = (
            quantity if quantity is not None and 0 < quantity < pos.quantity else pos.quantity
        )
        factor = self.quote_to_account.get(pos.symbol, 1.0)
        if pos.side == PositionSide.BUY:
            pnl = (exit_price - pos.entry_price) * close_qty * factor
        else:
            pnl = (pos.entry_price - exit_price) * close_qty * factor

        remaining = pos.quantity - close_qty
        if remaining <= 0:
            closed = pos.model_copy(
                update={
                    "status": PositionStatus.CLOSED,
                    "closed_at": exit_time,
                    "realized_pnl": pnl,
                    "closed_quantity": pos.quantity,
                }
            )
            self._positions[position_id] = closed
            self.balance += pnl
            return closed

        partial = pos.model_copy(
            update={
                "quantity": remaining,
                "realized_pnl": pos.realized_pnl + pnl,
                "closed_quantity": pos.closed_quantity + close_qty,
            }
        )
        self._positions[position_id] = partial
        self.balance += pnl
        return partial

    def mark_position_sl_tp(
        self,
        position_id: str,
        tick_high: float,
        tick_low: float,
        tick_time: datetime,
    ) -> Position | None:
        """Resolve SL/TP for one position against one tick bar (SL-first)."""
        pos = self._positions.get(position_id)
        if pos is None or pos.status != PositionStatus.OPEN:
            return pos
        sl, tp = pos.stop_loss, pos.take_profit
        if sl is None and tp is None:
            return pos

        if pos.side == PositionSide.BUY:
            sl_hit = sl is not None and tick_low <= sl
            tp_hit = tp is not None and tick_high >= tp
        else:
            sl_hit = sl is not None and tick_high >= sl
            tp_hit = tp is not None and tick_low <= tp

        if sl_hit:
            return self.close_position(position_id, sl, tick_time)
        if tp_hit:
            return self.close_position(position_id, tp, tick_time)
        return pos

    def get_all_positions(self) -> list[Position]:
        return list(self._positions.values())

    def get_order_history(self) -> list[OrderResult]:
        return list(self._orders.values())

    # ── Internals ─────────────────────────────────────────────────────────────

    def _validate_request(self, request: OrderRequest) -> tuple[RejectionCode, str] | None:
        if not request.symbol or len(request.symbol.replace("/", "").replace("_", "")) != 6:
            return RejectionCode.INVALID_SYMBOL, f"invalid symbol {request.symbol!r}"
        if request.quantity <= 0:
            return RejectionCode.INVALID_QUANTITY, "quantity must be > 0"
        if request.stop_loss is not None and not self._stop_on_correct_side(
            request.side, request.requested_price, request.stop_loss
        ):
            return RejectionCode.INVALID_STOP, "stop-loss is on the wrong side of the entry"
        return None

    @staticmethod
    def _stop_on_correct_side(side: PositionSide, entry: float, stop: float) -> bool:
        if side == PositionSide.BUY:
            return stop < entry
        return stop > entry

    def _reject(self, request: OrderRequest, code: RejectionCode, reason: str) -> OrderResult:
        return OrderResult(
            order_id=request.order_id,
            status=OrderStatus.REJECTED,
            symbol=request.symbol,
            side=request.side,
            quantity=request.quantity,
            requested_price=request.requested_price,
            rejection_code=code,
            rejection_reason=reason,
            timestamp=request.timestamp,
        )