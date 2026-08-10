"""Strategy 2 - Liquidity Reversal.

LONG (conceptual):
  price interacts with a previously identified sell-side liquidity zone
  AND a valid sell-side liquidity sweep occurs (low sweep)
  AND price returns through the level
  AND bullish displacement/confirmation follows
  AND market/news conditions permit the setup
SHORT is the inverse.

NO VALID SETUP = NO SIGNAL.
"""


from app.strategy.base import Strategy
from app.strategy.context import StrategyContext
from app.strategy.models import (
    Setup,
    Signal,
    SignalDirection,
    SignalReason,
)
from app.strategy.rules import (
    acceptable_session,
    displacement_supports_direction,
    liquidity_zone_available,
    news_risk_acceptable,
    no_active_range,
    sweep_available,
    volatility_acceptable,
)
from app.strategy.signals import SignalScore, build_signal, calculate_risk_geometry

__all__ = ["LiquidityReversalStrategy"]


class LiquidityReversalStrategy(Strategy):
    """Liquidity reversal hypothesis (independent of trend-structure)."""

    name = "liquidity_reversal"

    def evaluate(self, context: StrategyContext) -> Signal | None:
        """Evaluate one causal bar; return a Signal or None."""
        if not acceptable_session(context):
            return None
        zones = context.liquidity_zones()
        if not zones:
            return None
        if sweep_available(context, "long"):
            return self._reversal_signal(context, SignalDirection.LONG, zones[-1])
        if sweep_available(context, "short"):
            return self._reversal_signal(context, SignalDirection.SHORT, zones[-1])
        return None

    def _reversal_signal(
        self,
        context: StrategyContext,
        direction: SignalDirection,
        zone,
    ) -> Signal | None:
        """Build a reversal signal when all confirmation conditions hold."""
        if not liquidity_zone_available(context):
            return None
        if not displacement_supports_direction(context, direction.value):
            return None

        sweeps = [
            s
            for s in context.sweeps()
            if s.available_from is None or s.available_from <= context.now
        ]
        if not sweeps:
            return None

        close = float(context.current_candle()["close"])
        if direction == SignalDirection.LONG:
            if close <= zone.upper:
                return None  # price hasn't returned above zone
        else:
            if close >= zone.lower:
                return None  # price hasn't returned below zone

        if not volatility_acceptable(context):
            return None
        if not news_risk_acceptable(context):
            return None
        if not no_active_range(context):
            return None

        feat = context.current_features()
        atr_val = float(feat["atr"]) if feat is not None and "atr" in feat else 0.0
        if atr_val <= 0:
            return None

        entry = close

        score = SignalScore()
        score.add(SignalReason.LIQUIDITY_SWEEP_OCCURRED.value, 1.0)
        score.add(SignalReason.PRICE_RETURN_THROUGH_LEVEL.value, 1.0)
        score.add(
            SignalReason.DISPLACEMENT_BULLISH.value
            if direction == SignalDirection.LONG
            else SignalReason.DISPLACEMENT_BEARISH.value,
            2.0,
        )
        score.add(SignalReason.VOLATILITY_ACCEPTABLE.value, 1.0)

        last_sweep = sweeps[-1]
        anchor_ts = (
            last_sweep.available_from
            if last_sweep.available_from is not None
            else context.now
        )
        geo = calculate_risk_geometry(
            direction,
            entry=entry,
            atr_value=atr_val,
            stop_distance_atr=context.config.stop_distance_atr,
            reward_risk_target=context.config.reward_risk_target,
        )

        setup = Setup(
            symbol=context.symbol,
            timeframe=context.timeframe,
            direction=direction,
            strategy=self.name,
            anchor_timestamp=anchor_ts,
            anchor_price=entry,
            context_key=f"zone_{context.now.strftime('%Y%m%d')}_{last_sweep.sweep_type.value}",
        )
        if self._on_cooldown(context.now, setup.identity()):
            return None

        regime = context.latest_regime()
        return build_signal(
            signal_id=f"{self.name}-{context.now.strftime('%Y%m%dT%H%M%S')}",
            timestamp=context.now,
            symbol=context.symbol,
            timeframe=context.timeframe,
            direction=direction,
            entry=geo["entry"],
            stop=geo["stop"],
            target=geo["target"],
            risk_distance=geo["risk_distance"],
            reward_distance=geo["reward_distance"],
            strategy=self.name,
            score=score.total,
            max_score=5.0,
            reasons=[r for r, v in score.points.items() if v > 0],
            regime=regime,
            market_state=regime.market_state.value if regime is not None else None,
            structure_evidence=["liquidity_reversal"],
            news_risk_state=context.maximum_news_risk(),
            config=context.config,
            setup=setup,
        )
