"""Structure-based regime signals.

Reuses :mod:`app.market_structure` (swings, structure, ranges) rather than
reimplementing swing detection. All consumed events are filtered by their
``available_from`` so a precomputed structure result cannot leak future
information into a regime snapshot.
"""

from bisect import bisect_right
from datetime import datetime

from app._causal_index import build_causal_index
from app.market_structure.models import MarketStructureResult, StructurePoint
from app.regime.config import RegimeConfig

__all__ = [
    "build_structure_query_cache",
    "range_active_at",
    "range_active_series",
    "structure_bias_at",
    "structure_bias_series",
]


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


def build_structure_query_cache(
    structure_result: MarketStructureResult | None,
) -> tuple | None:
    """Precompute causal lookup indexes for structure points and ranges.

    Returns a tuple cache (or ``None`` when there is no structure result) that
    enables O(log N) per-bar ``structure_bias_at`` / ``range_active_at``
    queries instead of the previous O(N) linear scans. Build ONCE per
    ``analyze``; the per-bar loop only consumes the cache.

    The cache layout mirrors ``build_causal_index`` ordering: non-``None``
    ``available_from`` items first (sorted), ``None`` items last.
    """
    if structure_result is None:
        return None

    sorted_pts, keys = build_causal_index(structure_result.structure)
    n = len(sorted_pts)
    up_prefix = [0] * (n + 1)
    down_prefix = [0] * (n + 1)
    for i, p in enumerate(sorted_pts):
        t = p.structure_type.value
        up_prefix[i + 1] = up_prefix[i] + (
            1 if t in ("higher_high", "higher_low") else 0
        )
        down_prefix[i + 1] = down_prefix[i] + (
            1 if t in ("lower_high", "lower_low") else 0
        )
    n_non_none = sum(1 for k in keys if k is not None)

    sorted_ranges, range_keys = build_causal_index(structure_result.ranges)
    n_rng_non_none = sum(1 for k in range_keys if k is not None)

    return (
        (sorted_pts, keys, up_prefix, down_prefix, n_non_none),
        (sorted_ranges, range_keys, n_rng_non_none),
    )


def structure_bias_at(
    cache: tuple | None,
    lookback: int,
    end_ts: datetime,
) -> tuple[float | None, int]:
    """O(log N) structural bias usable at ``end_ts`` (causal).

    Identical semantics to ``structure_bias_series(..., end_ts=end_ts)`` but
    queries the precomputed prefix-sum index via binary search instead of
    scanning every structure point.
    """
    if cache is None:
        return None, 0
    (sorted_pts, keys, up_prefix, down_prefix, n_non_none), _ = cache
    if n_non_none == 0:
        return None, 0
    n_avail = bisect_right(keys, end_ts, 0, n_non_none)
    if n_avail <= 0:
        return None, 0
    k = min(lookback, n_avail) if lookback > 0 else n_avail
    up = up_prefix[n_avail] - up_prefix[n_avail - k]
    down = down_prefix[n_avail] - down_prefix[n_avail - k]
    score = up - down
    return score / k, k


def range_active_at(cache: tuple | None, end_ts: datetime) -> bool:
    """O(log N) range-active check usable at ``end_ts`` (causal)."""
    if cache is None:
        return False
    _, (sorted_ranges, range_keys, n_rng_non_none) = cache
    if n_rng_non_none == 0:
        return False
    return bisect_right(range_keys, end_ts, 0, n_rng_non_none) > 0


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