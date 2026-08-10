"""Strategy 1 — Trend + Market Structure.

LONG (conceptual):
  regime supports bullish conditions
  AND market structure is bullish (recent HH/HL)
  AND volatility is acceptable
  AND no prohibited high-impact news window
  AND a valid bullish displacement/confirmation exists
SHORT is the inverse.

When conditions are incomplete, the strategy returns NO SIGNAL rather than
manufacturing a setup. Every condition is documented and deterministic.
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
    news_risk_acceptable,
    no_active_range,
    regime_supports_direction,
    structure_supports_direction,
    volatility_acceptable,
)
from app.strategy.signals import (
    SignalScore,
    build_signal,
    calculate_risk_geometry,
)

__all__ = ["TrendStructureStrategy"]


class TrendStructureStrategy(Strategy):
    """Baseline trend + structure strategy (engine validation, not optimized)."""

    name = "trend_structure"

    def evaluate(self, context: StrategyContext) -> Signal | None:
        """Evaluate one causal bar; return a Signal or None."""
        if not acceptable_session(context):
            return None

        # ── LONG path ──────────────────────────────────────────────────────────
        if regime_supports_direction(context, "long") and (
            structure_supports_direction(context, "long")
        ):
            return self._directional_signal(
                context, SignalDirection.LONG, "HHHL", context.current_candle()["close"]
            )

        # ── SHORT path ─────────────────────────────────────────────────────────
        if regime_supports_direction(context, "short") and (
            structure_supports_direction(context, "short")
        ):
            return self._directional_signal(
                context, SignalDirection.SHORT, "LHLL", context.current_candle()["close"]
            )

        return None

    def _directional_signal(
        self,
        context: StrategyContext,
        direction: SignalDirection,
        structure_key: str,
        entry: float,
    ) -> Signal | None:
        """Build signal only when all confirmation conditions hold."""
        # Confirmation gates (all must pass).
        if not displacement_supports_direction(context, direction.value):
            return None
        if not volatility_acceptable(context):
            return None
        if not news_risk_acceptable(context):
            return None
        if not no_active_range(context):
            return None

        # ATR for risk geometry from the causal features slice.
        feat = context.current_features()
        atr_val = float(feat["atr"]) if feat is not None and "atr" in feat else 0.0
        if atr_val <= 0:
            return None

        score = SignalScore()
        # Rule-agreement points (documented categories):
        # 1 = regime direction, 1 = structure, 2 = displacement, 1 = volatility
        score.add(SignalReason.REGIME_SUPPORTS_TREND.value, 1.0)
        score.add(
            SignalReason.HIGHER_HIGHS_HIGHER_LOWS.value
            if direction == SignalDirection.LONG
            else SignalReason.LOWER_HIGHS_LOWER_LOWS.value,
            1.0,
        )
        score.add(
            SignalReason.DISPLACEMENT_BULLISH.value
            if direction == SignalDirection.LONG
            else SignalReason.DISPLACEMENT_BEARISH.value,
            2.0,
        )
        score.add(SignalReason.VOLATILITY_ACCEPTABLE.value, 1.0)

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
            anchor_timestamp=context.now,
            anchor_price=entry,
            context_key=f"{structure_key}_{context.now.strftime('%Y%m%dT%H%M')}",
        )
        # Cooldown: block duplicate setup identity within cooldown window.
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
            reasons=list(score.points.keys()),
            regime=regime,
            market_state=regime.market_state.value if regime is not None else None,
            structure_evidence=[structure_key],
            news_risk_state=context.maximum_news_risk(),
            config=context.config,
            setup=setup,
        )