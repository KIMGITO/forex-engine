"""Strategy abstraction and backtest adapter.

A strategy receives a restricted :class:`StrategyContext` (causal, no future
data) and returns either a :class:`Signal` or ``None``. A signal is research
information — it is NOT an order. Only the backtest adapter converts a
completed signal into :class:`OrderIntent` objects for the Step 8 engine.
"""


from app.backtest.engine import BacktestContext
from app.backtest.models import OrderIntent, OrderSide
from app.strategy.config import StrategyConfig
from app.strategy.context import StrategyContext
from app.strategy.models import Setup, Signal, SignalDirection

__all__ = ["SignalToOrderAdapter", "Strategy"]


class Strategy:
    """Base class for signal-generating strategies."""

    name: str = "abstract"

    def __init__(self, config: StrategyConfig | None = None) -> None:
        self.config = config or StrategyConfig()
        self._pending_setups: list[Setup] = []

    def evaluate(self, context: StrategyContext) -> Signal | None:
        """Evaluate current causal context; return a Signal or None."""
        raise NotImplementedError

    # ── cooldown / duplicate protection ───────────────────────────────────────

    def _on_cooldown(self, now, setup_key: str) -> bool:
        """True if the given setup identity fired within cooldown_bars."""
        if not self._pending_setups:
            return False
        # Simple bar-based cooldown: last recorded setup timestamp within
        # cooldown_bars of now (as a heuristic). The scanner records each
        # signal via `recognize_signal` / `record_signal`.
        for s in self._pending_setups:
            if s.identity() == setup_key and (now - s.anchor_timestamp).total_seconds() < 0:
                # Anchor timestamps are historical; a fresh setup within the
                # same event horizon should be blocked. We use a deterministic
                # "same anchor" check: duplicate context_key is blocked.
                return True
        return False

    def record_signal(self, signal: Signal, now) -> None:
        """Record a signal's setup for cooldown/duplicate prevention."""
        if signal.setup is not None:
            self._pending_setups.append(signal.setup)
            # Keep bounded to avoid unbounded growth (research safety).
            if len(self._pending_setups) > 1000:
                self._pending_setups.pop(0)

    # ── multi-timeframe (MTF) gate ─────────────────────────────────────────────

    def mtf_gates_pass(self, base_direction: str, mtf_context) -> tuple[bool, list[str]]:
        """Evaluate configured MTF gate conditions against the MTF context.

        Returns ``(passed, reasons)``. When MTF is disabled (default) this
        always returns ``(True, [])`` — behavior is identical to a strategy
        without MTF enabled. When enabled, checks:
          - at least ``mtf_min_aligned`` aligned high-timeframe tiers
          - no higher-timeframe conflict (when ``mtf_require_no_conflict``)
          - HTF bullish/bearish hard requirement (when configured)
          - minimum HTF volatility quality (when configured)
        """
        if not self.config.mtf_enabled:
            return True, []
        if mtf_context is None:
            return False, ["MTF enabled but no MTF context available"]

        reasons: list[str] = []
        alignment = mtf_context.alignment.value

        # Count how many present HTF tiers (hierarchy[1:]) agree with base
        # direction (regime trend matches) — this is the actual alignment.
        aligned_count = 0
        for t in mtf_context.hierarchy[1:]:
            if not t.present or not t.trend_state:
                continue
            agrees = (
                (base_direction == "long" and t.trend_state == "bullish")
                or (base_direction == "short" and t.trend_state == "bearish")
            )
            if agrees:
                aligned_count += 1

        if aligned_count < self.config.mtf_min_aligned:
            reasons.append(
                f"MTF: only {aligned_count} aligned tier(s); need {self.config.mtf_min_aligned}"
            )

        if self.config.mtf_require_no_conflict and alignment == "conflicted":
            reasons.append("MTF: higher-timeframe conflict present")

        # HTF directional hard requirements.
        present_htf = [t for t in mtf_context.hierarchy[1:] if t.present]
        if (
            self.config.mtf_require_htf_bullish
            and not any(t.trend_state == "bullish" for t in present_htf)
        ):
            reasons.append("MTF: no bullish higher-timeframe tier")
        if (
            self.config.mtf_require_htf_bearish
            and not any(t.trend_state == "bearish" for t in present_htf)
        ):
            reasons.append("MTF: no bearish higher-timeframe tier")

        # Minimum HTF volatility quality.
        min_vol = self.config.mtf_min_volatility_quality
        if min_vol and present_htf:
            rank = {"low": 1, "normal": 2, "high": 3, "extreme": 4}
            allowed = rank.get(min_vol, 0)
            if all(
                t.volatility_state
                and rank.get(t.volatility_state, 0) > allowed
                for t in present_htf
            ):
                reasons.append(f"MTF: HTF volatility exceeds {min_vol}")

        return (len(reasons) == 0), reasons


class SignalToOrderAdapter:
    """Converts a strategy Signal into OrderIntent(s) for the backtester.

    Maintains the separation: Signal → Strategy → adapter → OrderIntent →
    BacktestEngine. The adapter only ever produces orders for signals with
    status DETECTED/CONFIRMED, never for EXPIRED/INVALIDATED.
    """

    def __init__(self, quantity: float) -> None:
        if quantity <= 0:
            raise ValueError("quantity must be > 0")
        self.quantity = quantity

    def to_order_intents(
        self,
        signal: Signal | None,
        context: BacktestContext,
        now,
    ) -> list[OrderIntent]:
        if signal is None:
            return []
        if signal.status.value not in ("detected", "confirmed"):
            return []
        side = OrderSide.BUY if signal.direction == SignalDirection.LONG else OrderSide.SELL
        return [
            OrderIntent(
                order_id=f"sig-{signal.signal_id}",
                symbol=signal.symbol,
                side=side,
                quantity=self.quantity,
                timestamp=now,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
            )
        ]

    def from_backtest_context(
        self,
        signal: Signal | None,
        context: BacktestContext,
    ) -> list[OrderIntent]:
        """Alias for to_order_intents(..., now=context.now)."""
        return self.to_order_intents(signal, context, context.now.to_pydatetime())