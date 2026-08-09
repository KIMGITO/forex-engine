"""Order intent → submitted order → fill lifecycle for the backtest engine."""

from datetime import datetime

from app.backtest.models import (
    Fill,
    Order,
    OrderIntent,
    OrderStatus,
)

__all__ = ["accept_order", "fill_order", "intent_to_order", "reject_order"]


def intent_to_order(intent: OrderIntent) -> Order:
    """Convert a strategy OrderIntent into a submitted (pending) order."""
    return Order(
        order_id=intent.order_id,
        symbol=intent.symbol,
        side=intent.side,
        quantity=intent.quantity,
        order_type=intent.order_type,
        requested_price=intent.requested_price,
        stop_loss=intent.stop_loss,
        take_profit=intent.take_profit,
        timestamp=intent.timestamp,
        status=OrderStatus.PENDING,
    )


def accept_order(order: Order) -> Order:
    """Transition to SUBMITTED (research simplification of ACCEPTED)."""
    return order.model_copy(update={"status": OrderStatus.ACCEPTED})


def reject_order(order: Order, reason: str) -> Order:
    """Transition to REJECTED with a documented reason."""
    return order.model_copy(
        update={"status": OrderStatus.REJECTED, "reject_reason": reason}
    )


def fill_order(
    order: Order,
    price: float,
    filled_at: datetime,
    slippage_applied: float,
    gross_value: float,
) -> tuple:
    """Produce (filled_order, fill). Order transitions to FILLED."""
    filled = order.model_copy(
        update={
            "status": OrderStatus.FILLED,
            "filled_price": price,
            "filled_at": filled_at,
        }
    )
    fill = Fill(
        order_id=order.order_id,
        symbol=order.symbol,
        side=order.side,
        quantity=order.quantity,
        price=price,
        timestamp=filled_at,
        slippage_applied=slippage_applied,
        gross_value=gross_value,
    )
    return filled, fill