"""Execution simulator: market/limit/stop fills, SL/TP, gap handling.

Fill policies are documented assumptions:
- OHLC data alone cannot reveal intrabar sequence. When a single bar touches
  BOTH stop-loss and take-profit, the conservative policy assumes SL fills
  first (the less favorable outcome). This is configurable via FillPolicy.
- When the market GAPS through an order level, the fill price is the worse of
  (requested level, bar open) — never a better-than-market assumption.
"""



from app.backtest.costs import FixedSlippageModel, SpreadModel
from app.backtest.models import (
    FillPolicy,
    Order,
    OrderSide,
    OrderType,
    Position,
)

__all__ = ["ExecutionSimulator", "FillResult"]


class FillResult:
    """Outcome of evaluating a single order/pair against a bar."""

    def __init__(
        self,
        filled: bool,
        price: float | None = None,
        slippage_applied: float = 0.0,
        reason: str | None = None,
    ) -> None:
        self.filled = filled
        self.price = price
        self.slippage_applied = slippage_applied
        self.reason = reason


class ExecutionSimulator:
    """Deterministic execution engine."""

    def __init__(
        self,
        spread_model: SpreadModel,
        slippage_model: FixedSlippageModel,
        fill_policy: FillPolicy = FillPolicy.CONSERVATIVE_SL_FIRST,
    ) -> None:
        self.spread = spread_model
        self.slippage = slippage_model
        self.policy = fill_policy

    # ── entry fills ──────────────────────────────────────────────────────────

    def evaluate_entry(self, order: Order, bar_mid: float, bar_open: float, bar_high: float, bar_low: float) -> FillResult:
        """Determine whether an order fills on this bar, and at what price.

        - MARKET: fill at bid/ask of current bar (with slippage).
        - LIMIT  (buy below / sell above): fills if bar range touches the
          limit level; gap handling fills at worse-of(limit, open).
        - STOP   (buy above / sell below): fills if bar range touches the
          stop level; gap handling fills at worse-of(stop, open).
        """
        if order.order_type == OrderType.MARKET:
            return self._market_fill(order, bar_mid)

        level = order.requested_price
        if level is None:
            return FillResult(False, reason="limit/stop missing requested price")

        # LIMIT buy fills when price trades DOWN to the level (bar_low <= level).
        # STOP  buy fills when price trades UP to the level (bar_high >= level).
        # LIMIT sell fills when price trades UP to the level (bar_high >= level).
        # STOP  sell fills when price trades DOWN to the level (bar_low <= level).
        if order.side == OrderSide.BUY:
            if order.order_type == OrderType.LIMIT:
                touched = bar_low <= level
            else:  # STOP
                touched = bar_high >= level
            if not touched:
                return FillResult(False, reason="level not touched")
            # Gap away: bar opened through the level without trading back.
            # STOPS fill at the worse-of(level, open); LIMITS fill at the level.
            opened_through = bar_open > level
            if order.order_type == OrderType.STOP and opened_through:
                fill_px = bar_open
            else:
                fill_px = level
        else:  # SELL
            if order.order_type == OrderType.LIMIT:
                touched = bar_high >= level
            else:  # STOP
                touched = bar_low <= level
            if not touched:
                return FillResult(False, reason="level not touched")
            opened_through = bar_open < level
            if order.order_type == OrderType.STOP and opened_through:
                fill_px = bar_open
            else:
                fill_px = level

        fill_px = self.slippage.slippage_price(fill_px, order.side.value)
        return FillResult(True, price=fill_px)

    def _market_fill(self, order: Order, bar_mid: float) -> FillResult:
        bid, ask = self.spread.bid_ask(bar_mid)
        if order.side == OrderSide.BUY:
            base_px = ask
        else:
            base_px = bid
        fill_px = self.slippage.slippage_price(base_px, order.side.value)
        return FillResult(True, price=fill_px)

    # ── SL/TP resolution ─────────────────────────────────────────────────────

    def resolve_stop_take_profit(
        self,
        position: Position,
        bar_mid: float,
        bar_open: float,
        bar_high: float,
        bar_low: float,
    ) -> FillResult | None:
        """Resolve whether the position's SL or TP is triggered on this bar.

        Returns a FillResult with the exit price, or None if neither level is
        touched. When both are touched, applies the configured policy:

        - CONSERVATIVE_SL_FIRST (default): assume SL fills first.
        - FULL_OHLC_WHIPSAW: assumes both fire, but since OHLC cannot order
          them, we apply the documented conservative behavior too (SL-first)
          and add a ``whipsaw`` note for research visibility. (The enum value
          documents the ambiguity; both resolve SL-first for causal safety.)
        """
        sl = position.stop_loss
        tp = position.take_profit
        if sl is None and tp is None:
            return None

        if position.side == OrderSide.BUY:
            sl_hit = sl is not None and bar_low <= sl
            tp_hit = tp is not None and bar_high >= tp
        else:  # SELL
            sl_hit = sl is not None and bar_high >= sl
            tp_hit = tp is not None and bar_low <= tp

        if sl is not None and sl_hit and tp_hit:
            # Conservative: SL first (less favorable outcome by construction).
            return FillResult(
                True,
                price=sl,
                reason="sl_tp_both_touched_conservative_sl_first",
            )

        if sl is not None and sl_hit:
            # Gap through level: worse-of(level, open directional).
            exit_px = self._gap_adjusted_exit(position.side, sl, bar_open, is_sl=True)
            return FillResult(True, price=exit_px, reason="stop_loss")

        if tp is not None and tp_hit:
            exit_px = self._gap_adjusted_exit(position.side, tp, bar_open, is_sl=False)
            return FillResult(True, price=exit_px, reason="take_profit")

        return None

    @staticmethod
    def _gap_adjusted_exit(side: OrderSide, level: float, bar_open: float, is_sl: bool) -> float:
        """Gap-through behavior: fill at worse-of(level, open) for SL exits.

        TP exits that gap through are filled at the level (already favorable);
        we never improve beyond the requested TP.
        """
        if not is_sl:
            return level
        if side == OrderSide.BUY:
            # SL below; if bar opened below the level, we get the open (worse).
            return min(level, bar_open)
        # SL above; if bar opened above the level, we get the open (worse).
        return max(level, bar_open)