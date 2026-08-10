"""Reusable, transparent rule checks for strategies.

Every rule below is a deterministic boolean/score contribution based only on
causal information available at the current timestamp.
"""


from app.strategy.context import StrategyContext

__all__ = [
    "acceptable_session",
    "displacement_supports_direction",
    "liquidity_zone_available",
    "news_risk_acceptable",
    "no_active_range",
    "regime_supports_direction",
    "structure_supports_direction",
    "sweep_available",
    "volatility_acceptable",
]


def regime_supports_direction(
    context: StrategyContext, direction: str
) -> bool:
    """Regime trend_state matches LONG/BEARISH direction with strength >= min.

    The regime is a DESCRIPTION of the market, never a guarantee.
    """
    regime = context.latest_regime()
    if regime is None:
        return False
    if regime.strength < context.config.min_regime_strength:
        return False
    if direction == "long":
        return regime.trend_state.value in ("bullish",)
    return regime.trend_state.value in ("bearish",)


def structure_supports_direction(context: StrategyContext, direction: str) -> bool:
    """Recent structure points agree with the direction (HH/HL for long)."""
    points = context.structure_points()
    if len(points) < context.config.min_structure_points:
        return False
    relevant = points[-context.config.min_structure_points :]
    wanted = (
        {"higher_high", "higher_low"}
        if direction == "long"
        else {"lower_high", "lower_low"}
    )
    matches = sum(1 for p in relevant if p.structure_type.value in wanted)
    return matches == len(relevant)


def displacement_supports_direction(
    context: StrategyContext, direction: str
) -> bool:
    """Recent displacement event is large/extreme in the direction."""
    events = context.displacement_events()
    if not events:
        return False
    recent = events[-1]
    min_class = context.config.min_displacement_class
    if min_class == "large":
        allowed = {"large", "extreme"}
    elif min_class == "extreme":
        allowed = {"extreme"}
    else:
        allowed = {"normal", "large", "extreme"}
    if recent.classification.value not in allowed:
        return False
    if direction == "long":
        return recent.direction == "up"
    if direction == "short":
        return recent.direction == "down"
    return False


def volatility_acceptable(context: StrategyContext) -> bool:
    """Current volatility state is at or below the configured maximum."""
    regime = context.latest_regime()
    if regime is None:
        return False
    max_state = context.config.max_volatility_state
    rank = {"low": 1, "normal": 2, "high": 3, "extreme": 4}
    current = rank.get(regime.volatility_state.value, 0)
    allowed = rank.get(max_state, 2)
    return 0 < current <= allowed


def news_risk_acceptable(context: StrategyContext) -> bool:
    """Active news-risk state at/below max (high-impact windows prohibited)."""
    max_state = context.config.max_news_risk
    rank = {"low": 1, "medium": 2, "high": 3}
    allowed = rank.get(max_state, 2)
    current_rank = 0
    for e in context.news_available():
        imp = getattr(e, "importance", None)
        if imp is None:
            continue
        current_rank = max(current_rank, rank.get(imp.value, 0))
    return current_rank <= allowed


def liquidity_zone_available(context: StrategyContext) -> bool:
    """At least the configured minimum number of liquidity zones available."""
    return len(context.liquidity_zones()) >= context.config.min_liquidity_zones


def sweep_available(context: StrategyContext, direction: str) -> bool:
    """A liquidity sweep exists with matching side within causal lookback."""
    recent = [
        s
        for s in context.sweeps()
        if s.available_from is None or s.available_from <= context.now
    ]
    if not recent:
        return False
    last = recent[-1]
    if direction == "long":
        return last.sweep_type.value == "low_sweep"
    if direction == "short":
        return last.sweep_type.value == "high_sweep"
    return False


def acceptable_session(context: StrategyContext) -> bool:
    """Session restriction: empty = always allowed."""
    allowed = context.config.allowed_hours_utc
    if not allowed:
        return True
    return context.now.hour in allowed


def no_active_range(context: StrategyContext) -> bool:
    """Requires that no range event is currently active at now."""
    for r in context.active_ranges():
        if (
            r.available_from is not None
            and r.available_from <= context.now
            and (r.end_timestamp is None or r.end_timestamp >= context.now)
        ):
            return False
    return True
