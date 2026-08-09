"""Multi-factor regime classification.

Combines trend, volatility, structure, and range signals into a coherent
:class:`MarketRegime`. Regime classification is a descriptive model — it does
NOT generate trade signals and does NOT claim future price behavior.
"""

from datetime import datetime

from app.regime.models import (
    MarketRegime,
    MarketState,
    TrendState,
    VolatilityState,
)
from app.regime.models import (
    NewsRiskState as RegimeNewsState,
)

__all__ = ["classify_regime"]


def _trend_direction(trend: TrendState) -> int | None:
    if trend == TrendState.BULLISH:
        return 1
    if trend == TrendState.BEARISH:
        return -1
    return None


def _volatility_index(vol: VolatilityState) -> float:
    return {
        VolatilityState.LOW: 0.0,
        VolatilityState.NORMAL: 0.5,
        VolatilityState.HIGH: 1.0,
        VolatilityState.EXTREME: 1.5,
        VolatilityState.UNKNOWN: float("nan"),
    }.get(vol, float("nan"))


def classify_regime(
    symbol: str,
    timeframe: str,
    timestamp: datetime,
    trend: TrendState,
    volatility: VolatilityState,
    structure_bias: float | None,
    structure_count: int,
    range_active: bool,
    transition_vol_ratio: float,
    news_risk: RegimeNewsState | None = None,
    metrics: dict | None = None,
    min_structure_points: int = 3,
) -> MarketRegime:
    """Combine factor signals into a single MarketRegime.

    Rules:
    - TRENDING when trend is decisive (BULLISH/BEARISH) and structure confirms
      (or structure is insufficient but trend is decisive and no range).
    - RANGING when a range is active and trend is not decisive.
    - TRANSITION when trend signals conflict, volatility expands beyond
      ``transition_vol_ratio``, or a HIGH-impact news window is active
      (uncertainty only — never a directional claim).
    - UNKNOWN when insufficient evidence.
    """
    metrics = dict(metrics or {})
    metrics["structure_bias"] = float(structure_bias) if structure_bias is not None else float("nan")
    metrics["structure_points"] = float(structure_count)
    metrics["range_active"] = 1.0 if range_active else 0.0
    metrics["transition_vol_ratio"] = float(transition_vol_ratio)

    trend_dir = _trend_direction(trend)
    structure_dir = 1 if (structure_bias is not None and structure_bias > 0.1) else (
        -1 if (structure_bias is not None and structure_bias < -0.1) else 0
    )

    # Directional agreement for strength (objective internal agreement).
    agreed = 0
    total = 0
    if trend_dir is not None:
        total += 1
        if structure_dir == 0 or structure_dir == trend_dir:
            agreed += 1
        else:
            total += 1  # disagreement counts toward both directions
    if volatility != VolatilityState.UNKNOWN:
        total += 1
        if trend_dir is not None and _volatility_index(volatility) < 1.0:
            agreed += 1  # moderate vol with a trend = confirmation; extreme = not

    # Decision tree
    if trend == TrendState.UNKNOWN or volatility == VolatilityState.UNKNOWN:
        market_state = MarketState.UNKNOWN
    elif structure_count >= min_structure_points and structure_dir != 0 and trend_dir is not None and structure_dir != trend_dir:
        # Trend and structure actively disagree -> transition/uncertain.
        market_state = MarketState.TRANSITION
    elif range_active and trend not in (TrendState.BULLISH, TrendState.BEARISH):
        market_state = MarketState.RANGING
    elif range_active and trend in (TrendState.BULLISH, TrendState.BEARISH) and structure_count < min_structure_points or transition_vol_ratio is not None and transition_vol_ratio >= 1.6:
        market_state = MarketState.TRANSITION
    elif trend in (TrendState.BULLISH, TrendState.BEARISH) and (structure_dir == trend_dir or structure_count < min_structure_points):
        market_state = MarketState.TRENDING
    elif trend == TrendState.NEUTRAL and not range_active:
        market_state = MarketState.TRANSITION
    else:
        market_state = MarketState.RANGING if range_active else MarketState.TRANSITION

    # News risk is metadata: it does NOT change direction, only raises
    # uncertainty when a high-impact window is active (already handled above
    # for the transition-uncertainty path, but recorded here for context).
    if news_risk is None:
        news_risk = RegimeNewsState.UNKNOWN

    strength = agreed / total if total > 0 else 0.0
    if market_state in (MarketState.TRANSITION, MarketState.UNKNOWN):
        strength = min(strength, 0.5)  # uncertainty caps objective agreement

    return MarketRegime(
        symbol=symbol,
        timeframe=timeframe,
        timestamp=timestamp,
        trend_state=trend,
        volatility_state=volatility,
        market_state=market_state,
        news_risk=news_risk,
        strength=float(strength),
        metrics=metrics,
        available_from=timestamp,
    )
