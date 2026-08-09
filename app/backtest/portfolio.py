"""Portfolio accounting: balance, equity, margin, position P&L.

Supports long and short positions with FIFO-average entry, realized and
unrealized P&L (correctly directional per Forex side), fees, financing, and
leverage-based margin. Leverage affects usable margin only — never
profitability.
"""

from datetime import datetime

from app.backtest.models import (
    Fill,
    OrderSide,
    PortfolioState,
    Position,
    Trade,
)

__all__ = ["Portfolio"]


def pnl_for_sides(
    side: OrderSide,
    quantity: float,
    entry_price: float,
    exit_price: float,
) -> float:
    """Realized P&L for a long/short pair in quote units."""
    if side == OrderSide.BUY:
        return (exit_price - entry_price) * quantity
    return (entry_price - exit_price) * quantity


class Portfolio:
    """Mutable account state for a backtest run."""

    def __init__(
        self,
        initial_balance: float,
        account_currency: str = "USD",
        leverage: int = 30,
        max_position_size: float | None = None,
    ) -> None:
        if initial_balance <= 0:
            raise ValueError("initial_balance must be > 0")
        if leverage <= 0:
            raise ValueError("leverage must be > 0")
        self.initial_balance = initial_balance
        self.account_currency = account_currency
        self.leverage = leverage
        self.max_position_size = max_position_size
        self.balance = initial_balance
        self.positions: dict[str, Position] = {}
        self.realized_pnl = 0.0
        self.fees = 0.0
        self.financing = 0.0
        self.trades: list[Trade] = []
        self._trade_counter = 0

    # ── P&L ──────────────────────────────────────────────────────────────────

    def unrealized_pnl(self, mid: float, symbol: str | None = None) -> float:
        total = 0.0
        for sym, pos in self.positions.items():
            if symbol is not None and sym != symbol:
                continue
            if pos.side == OrderSide.BUY:
                total += (mid - pos.average_entry) * pos.quantity
            else:
                total += (pos.average_entry - mid) * pos.quantity
        return total

    # ── Margin ───────────────────────────────────────────────────────────────

    def used_margin(self) -> float:
        total = 0.0
        for pos in self.positions.values():
            notional = pos.quantity * pos.average_entry
            total += notional / self.leverage
        return total

    def free_margin(self, mid: float) -> float:
        return self.equity(mid) - self.used_margin()

    def equity(self, mid: float) -> float:
        return self.balance + self.unrealized_pnl(mid)

    # ── Position lifecycle ───────────────────────────────────────────────────

    def open_position(
        self,
        fill: Fill,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> Position:
        """Open (or add to) a position."""
        existing = self.positions.get(fill.symbol)
        if existing is None:
            pos = Position(
                symbol=fill.symbol,
                side=fill.side,
                quantity=fill.quantity,
                average_entry=fill.price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                opened_at=fill.timestamp,
                holding_bars=0,
            )
            self.positions[fill.symbol] = pos
            return pos

        # Same-side add: FIFO-average entry price.
        if existing.side == fill.side:
            total_qty = existing.quantity + fill.quantity
            avg = (
                existing.average_entry * existing.quantity
                + fill.price * fill.quantity
            ) / total_qty
            new_pos = existing.model_copy(
                update={
                    "quantity": total_qty,
                    "average_entry": avg,
                    "holding_bars": existing.holding_bars + 1,
                }
            )
            self.positions[fill.symbol] = new_pos
            return new_pos

        # Opposite-side fill: reduce or close existing position.
        return self._reduce_position(fill, existing)

    def _reduce_position(self, fill: Fill, existing: Position) -> Position:
        if fill.quantity >= existing.quantity:
            # Full close.
            pnl = pnl_for_sides(
                existing.side, existing.quantity, existing.average_entry, fill.price
            )
            self.realized_pnl += pnl
            self.balance += pnl
            self._record_trade(existing, fill, pnl, fill.quantity)
            del self.positions[fill.symbol]
            return existing
        # Partial close (research simplification: reduce quantity).
        pnl = pnl_for_sides(
            existing.side, fill.quantity, existing.average_entry, fill.price
        )
        self.realized_pnl += pnl
        self.balance += pnl
        reduced = existing.model_copy(
            update={
                "quantity": existing.quantity - fill.quantity,
                "holding_bars": existing.holding_bars + 1,
            }
        )
        self.positions[fill.symbol] = reduced
        return reduced

    def _record_trade(
        self,
        pos: Position,
        exit_fill: Fill,
        pnl: float,
        closed_qty: float,
    ) -> None:
        self._trade_counter += 1
        self.trades.append(
            Trade(
                trade_id=f"t{self._trade_counter}",
                symbol=pos.symbol,
                side=pos.side,
                entry_time=pos.opened_at,
                exit_time=exit_fill.timestamp,
                entry_price=pos.average_entry,
                exit_price=exit_fill.price,
                quantity=closed_qty,
                net_pnl=pnl - pos.fees - pos.financing,
                fees=pos.fees,
                financing=pos.financing,
                holding_bars=pos.holding_bars,
            )
        )

    def apply_commission(self, amount: float) -> None:
        self.balance -= amount
        self.fees += amount
        self.realized_pnl -= amount

    def apply_financing(self, amount: float) -> None:
        self.balance -= amount
        self.financing += amount
        self.realized_pnl -= amount

    def close_position_via_sl_tp(
        self,
        symbol: str,
        exit_price: float,
        exit_time: datetime,
    ) -> Trade | None:
        pos = self.positions.get(symbol)
        if pos is None:
            return None
        pnl = pnl_for_sides(pos.side, pos.quantity, pos.average_entry, exit_price)
        self.realized_pnl += pnl
        self.balance += pnl
        self._record_trade(
            pos,
            Fill(
                order_id=f"sl_tp_{pos.symbol}",
                symbol=pos.symbol,
                side=pos.side,
                quantity=pos.quantity,
                price=exit_price,
                timestamp=exit_time,
                gross_value=exit_price * pos.quantity,
            ),
            pnl,
            pos.quantity,
        )
        del self.positions[symbol]
        return self.trades[-1]

    def snapshot(self, timestamp: datetime, mid: float) -> PortfolioState:
        eq = self.equity(mid)
        um = self.used_margin()
        return PortfolioState(
            timestamp=timestamp,
            balance=self.balance,
            equity=eq,
            margin=um * self.leverage,
            free_margin=self.free_margin(mid),
            used_margin=um,
            open_positions=list(self.positions.values()),
            realized_pnl=self.realized_pnl,
            unrealized_pnl=self.unrealized_pnl(mid),
            fees=self.fees,
            financing=self.financing,
        )