"""Liquidity: equal highs/lows -> potential liquidity zones, and sweeps.

**Important distinction:** A *potential liquidity zone* is derived purely from
price structure (clustered swing highs or lows within a tolerance). It is a
measurable statement about price, **not** a claim about actual order-book
liquidity or the location of stop orders. We cannot infer the existence of
stop orders from OHLC data alone.
"""

from datetime import datetime

import pandas as pd

from app.market_structure.errors import LiquidityError
from app.market_structure.models import (
    LiquidityZone,
    SweepEvent,
    SweepType,
    Swing,
    SwingType,
)

__all__ = ["detect_liquidity_zones", "detect_sweeps"]


def detect_liquidity_zones(
    swings: list[Swing],
    symbol: str,
    timeframe: str,
    tolerance_pct: float = 0.05,
    min_swings: int = 2,
) -> list[LiquidityZone]:
    """Group equal swing highs and equal swing lows into potential liquidity zones.

    Two swings of the same type are "equal" if their prices fall within
    ``tolerance_pct`` percent of each other. Zones require at least
    ``min_swings`` swings.
    """
    if tolerance_pct < 0:
        raise LiquidityError("tolerance_pct must be non-negative.")
    if min_swings < 2:
        raise LiquidityError("min_swings must be >= 2.")

    highs = [s for s in swings if s.swing_type == SwingType.HIGH]
    lows = [s for s in swings if s.swing_type == SwingType.LOW]

    zones: list[LiquidityZone] = []
    zones.extend(_group_equal(highs, symbol, timeframe, tolerance_pct, min_swings, "equal_highs"))
    zones.extend(_group_equal(lows, symbol, timeframe, tolerance_pct, min_swings, "equal_lows"))
    return zones


def _group_equal(
    swings: list[Swing],
    symbol: str,
    timeframe: str,
    tolerance_pct: float,
    min_swings: int,
    zone_type: str,
) -> list[LiquidityZone]:
    """Greedily group swings whose prices are within tolerance."""
    zones: list[LiquidityZone] = []
    n = len(swings)
    used = [False] * n

    for i in range(n):
        if used[i]:
            continue
        group = [swings[i]]
        used[i] = True
        for j in range(i + 1, n):
            if used[j]:
                continue
            if _within_tolerance(swings[i].price, swings[j].price, tolerance_pct):
                group.append(swings[j])
                used[j] = True

        if len(group) >= min_swings:
            prices = [g.price for g in group]
            upper = max(prices)
            lower = min(prices)
            zones.append(
                LiquidityZone(
                    symbol=symbol,
                    timeframe=timeframe,
                    zone_type=zone_type,
                    upper=upper,
                    lower=lower,
                    mid=(upper + lower) / 2.0,
                    swing_count=len(group),
                    first_timestamp=min(g.timestamp for g in group),
                    last_timestamp=max(g.timestamp for g in group),
                    # The zone is only knowable once the last grouped swing is
                    # confirmed.
                    available_from=max(g.confirmation_timestamp for g in group),
                )
            )
    return zones


def _within_tolerance(price_a: float, price_b: float, tolerance_pct: float) -> bool:
    """Return True if two prices are within ``tolerance_pct`` percent."""
    if price_a == 0:
        return price_b == 0
    return abs(price_a - price_b) / abs(price_a) * 100.0 <= tolerance_pct


def detect_sweeps(
    data: pd.DataFrame,
    zones: list[LiquidityZone],
    symbol: str,
    timeframe: str,
    sweep_bars: int = 3,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
) -> list[SweepEvent]:
    """Detect liquidity sweeps against pre-existing potential liquidity zones.

    A **high sweep** at bar ``i`` requires:
      1. A prior ``equal_highs`` zone whose ``available_from <= timestamp[i]``.
      2. ``high[i] > zone.upper`` (price trades above the level).
      3. Within ``sweep_bars`` bars, a close ``<= zone.upper`` (price returns).

    A **low sweep** is the mirror against ``equal_lows`` zones.

    A wick without a subsequent return, or a move without a prior zone, is
    **not** a sweep.
    """
    for col in (high_col, low_col, close_col):
        if col not in data.columns:
            raise LiquidityError(f"Column '{col}' not found in data.")

    sorted_data = data.sort_index()
    highs = sorted_data[high_col].to_numpy(dtype=float)
    lows = sorted_data[low_col].to_numpy(dtype=float)
    closes = sorted_data[close_col].to_numpy(dtype=float)
    timestamps = sorted_data.index.to_numpy()
    n = len(highs)

    events: list[SweepEvent] = []

    for i in range(n):
        ts = _to_datetime(timestamps[i])

        # High sweeps
        for zone in zones:
            if zone.zone_type != "equal_highs":
                continue
            if zone.available_from > ts:
                continue
            if highs[i] <= zone.upper:
                continue
            for k in range(1, sweep_bars + 1):
                if i + k >= n:
                    break
                if closes[i + k] <= zone.upper:
                    events.append(
                        SweepEvent(
                            symbol=symbol,
                            timeframe=timeframe,
                            timestamp=ts,
                            sweep_type=SweepType.HIGH_SWEEP,
                            level=zone.upper,
                            extreme_price=float(highs[i]),
                            close_price=float(closes[i + k]),
                            available_from=_to_datetime(timestamps[i + k]),
                        )
                    )
                    break

        # Low sweeps
        for zone in zones:
            if zone.zone_type != "equal_lows":
                continue
            if zone.available_from > ts:
                continue
            if lows[i] >= zone.lower:
                continue
            for k in range(1, sweep_bars + 1):
                if i + k >= n:
                    break
                if closes[i + k] >= zone.lower:
                    events.append(
                        SweepEvent(
                            symbol=symbol,
                            timeframe=timeframe,
                            timestamp=ts,
                            sweep_type=SweepType.LOW_SWEEP,
                            level=zone.lower,
                            extreme_price=float(lows[i]),
                            close_price=float(closes[i + k]),
                            available_from=_to_datetime(timestamps[i + k]),
                        )
                    )
                    break

    return events


def _to_datetime(value) -> datetime:
    if isinstance(value, datetime):
        return value
    return pd.Timestamp(value).to_pydatetime()
