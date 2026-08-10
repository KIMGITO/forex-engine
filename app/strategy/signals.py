"""Signal scoring and risk-geometry utilities.

Scoring is a deterministic RULE-AGREEMENT model, explicitly NOT a probability
and NOT a prediction. Scores accumulate points for each documented condition
that holds; thresholds map to WEAK/MODERATE/STRONG.
"""


from app.strategy.config import StrategyConfig
from app.strategy.errors import SignalValidationError
from app.strategy.models import (
    Setup,
    Signal,
    SignalDirection,
    SignalStrength,
)

__all__ = [
    "SignalScore",
    "build_signal",
    "calculate_risk_geometry",
    "classify_strength",
]


class SignalScore:
    """Accumulates rule-agreement points for a candidate signal."""

    def __init__(self) -> None:
        self.points: dict[str, float] = {}
        self.total = 0.0

    def add(self, reason: str, value: float) -> None:
        self.points[reason] = value
        self.total += value

    def has(self, reason: str) -> bool:
        return bool(self.points.get(reason, 0.0) > 0)


def classify_strength(score: float, config: StrategyConfig) -> SignalStrength:
    """Map a rule-agreement score to a documented categorical strength."""
    if score >= config.strong_score_threshold:
        return SignalStrength.STRONG
    if score >= config.moderate_score_threshold:
        return SignalStrength.MODERATE
    return SignalStrength.WEAK


def calculate_risk_geometry(
    direction: SignalDirection,
    entry: float,
    atr_value: float,
    stop_distance_atr: float,
    reward_risk_target: float,
) -> dict[str, float]:
    """Compute entry/stop/target distances and R:R.

    LONG:  stop = entry - stop_dist_atr * ATR; target = entry + reward_risk*risk.
    SHORT: stop = entry + stop_dist_atr * ATR; target = entry - rr*risk.
    """
    if atr_value <= 0:
        raise SignalValidationError("ATR must be > 0 to compute risk geometry")
    if entry <= 0:
        raise SignalValidationError("entry must be > 0")
    risk_distance = stop_distance_atr * atr_value
    reward_distance = reward_risk_target * risk_distance
    if direction == SignalDirection.LONG:
        stop = entry - risk_distance
        target = entry + reward_distance
    else:
        stop = entry + risk_distance
        target = entry - reward_distance
    if stop <= 0 or target <= 0:
        raise SignalValidationError("stop/target must be > 0")
    return {
        "entry": entry,
        "stop": stop,
        "target": target,
        "risk_distance": risk_distance,
        "reward_distance": reward_distance,
        "r_r": reward_risk_target,
    }


def build_signal(
    *,
    signal_id: str,
    timestamp,
    symbol: str,
    timeframe: str,
    direction: SignalDirection,
    entry: float,
    stop: float,
    target: float,
    risk_distance: float,
    reward_distance: float,
    strategy: str,
    score: float,
    max_score: float,
    reasons: list,
    regime,
    market_state,
    structure_evidence,
    news_risk_state,
    config: StrategyConfig,
    setup: Setup | None = None,
    metadata: dict | None = None,
) -> Signal:
    """Assemble a validated Signal (frozen, all invariants enforced)."""
    strength = classify_strength(score, config)
    return Signal(
        signal_id=signal_id,
        timestamp=timestamp,
        symbol=symbol,
        timeframe=timeframe,
        direction=direction,
        strength=strength,
        score=score,
        max_score=max_score,
        entry=entry,
        stop_loss=stop,
        take_profit=target,
        risk_distance=risk_distance,
        reward_distance=reward_distance,
        risk_reward_ratio=reward_distance / risk_distance if risk_distance > 0 else 0.0,
        strategy=strategy,
        regime=str(regime) if regime else None,
        market_state=str(market_state) if market_state else None,
        reasons=[r for r in reasons],
        structure_evidence=list(structure_evidence),
        news_risk_state=news_risk_state,
        setup=setup,
        available_from=timestamp,
        metadata=metadata or {},
    )
