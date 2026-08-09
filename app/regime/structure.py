"""Structure-based regime signals.

Reuses :mod:`app.market_structure` (swings, structure, ranges) rather than
reimplementing swing detection. All consumed events are filtered by their
``available_from`` so a precomputed structure result cannot leak future
information into a regime snapshot.
"""

from datetime import datetime

from app.market_structure.models import MarketStructureResult, StructurePoint
from app.regime.config import RegimeConfig

__all__ = ["range_active_series", "structure_bias_series"]


def _structure_bias(points: list[StructurePoint], lookback: int) -> tuple[float | None, int]:
    """Return (directional bias in -1..1, usable point count) for recent structure.

    Bias = (HH count + HL count - LH count - LL count) / max(1, points used).
    Higher-high/higher-low are bullish; lower-high/lower-low are bearish.
    """
    recent = points[-lookback:] if lookback > 0 else points
    usable = [p for p in recent if p.available_from is not None]
    if not usable:
        return None, 0
    score = 0.0
    for p in usable:
        t = p.structure_type.value
        if t in ("higher_high", "higher_low"):
            score += 1.0
        elif t in ("lower_high", "lower_low"):
            score -= 1.0
    return score / len(usable), len(usable)


def structure_bias_series(
    structure_result: MarketStructureResult | None,
    config: RegimeConfig,
    end_ts: datetime | None = None,
) -> tuple[float | None, int]:
    """Compute the most recent structure bias usable at ``end_ts`` (or all)."""
    if structure_result is None:
        return None, 0

    points = list(structure_result.structure)
    if end_ts is not None:
        points = [p for p in points if p.available_from is not None and p.available_from <= end_ts]
    return _structure_bias(points, config.structure_lookback)


def range_active_series(
    structure_result: MarketStructureResult | None,
    end_ts: datetime | None = None,
) -> bool:
    """Return True if a range event is usable at ``end_ts``."""
    if structure_result is None:
        return False
    for r in structure_result.ranges:
        if r.available_from is None:
            continue
        if end_ts is None or r.available_from <= end_ts:
            return True
    return False
