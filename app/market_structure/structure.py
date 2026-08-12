"""Market structure: higher highs/lows, lower highs/lows, and breaks.

Structural relationships are derived from *confirmed* swings only. A structure
point is only knowable once the newer swing is confirmed, so its
``available_from`` equals the newer swing's confirmation timestamp.

Break-of-structure events are explicitly separated into three types:

* ``wick_breach`` — price trades through a level but closes back on the
  original side (intrabar penetration only).
* ``close_break`` — price closes beyond the level.
* ``confirmed_break`` — a close beyond the level that is sustained for
  ``confirm_bars`` subsequent bars (or, if ``confirm_bars == 0``, a close
  break with a minimum move ``min_move_pct``).
"""

from datetime import datetime

import pandas as pd

from app.market_structure.errors import StructureError
from app.market_structure.models import (
    BreakEvent,
    BreakType,
    StructurePoint,
    StructureType,
    Swing,
    SwingType,
)

__all__ = ["build_structure", "detect_breaks"]


def build_structure(
    swings: list[Swing],
    symbol: str,
    timeframe: str,
) -> list[StructurePoint]:
    """Build a chronological structure sequence from confirmed swings.

    For each new confirmed swing, compare it to the most recent prior swing of
    the *same type*:

    * Swing high: higher than prior high -> ``higher_high``; lower -> ``lower_high``.
    * Swing low: higher than prior low -> ``higher_low``; lower -> ``lower_low``.

    Returns structure points ordered by the newer swing's timestamp.
    """
    if not swings:
        return []

    # Sort by confirmation timestamp so structure is only built from swings
    # that are actually knowable in sequence.
    ordered = sorted(swings, key=lambda s: s.confirmation_timestamp)
    last_high: Swing | None = None
    last_low: Swing | None = None
    points: list[StructurePoint] = []

    for swing in ordered:
        if swing.swing_type == SwingType.HIGH:
            if last_high is not None:
                if swing.price > last_high.price:
                    stype = StructureType.HIGHER_HIGH
                else:
                    stype = StructureType.LOWER_HIGH
                points.append(
                    StructurePoint(
                        symbol=symbol,
                        timeframe=timeframe,
                        timestamp=swing.timestamp,
                        structure_type=stype,
                        price=swing.price,
                        prior_price=last_high.price,
                        available_from=swing.confirmation_timestamp,
                    )
                )
            last_high = swing
        else:
            if last_low is not None:
                if swing.price > last_low.price:
                    stype = StructureType.HIGHER_LOW
                else:
                    stype = StructureType.LOWER_LOW
                points.append(
                    StructurePoint(
                        symbol=symbol,
                        timeframe=timeframe,
                        timestamp=swing.timestamp,
                        structure_type=stype,
                        price=swing.price,
                        prior_price=last_low.price,
                        available_from=swing.confirmation_timestamp,
                    )
                )
            last_low = swing

    return points


def detect_breaks(
    data: pd.DataFrame,
    swings: list[Swing],
    symbol: str,
    timeframe: str,
    confirm_bars: int = 2,
    min_move_pct: float = 0.0,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
) -> list[BreakEvent]:
    """Detect structural breaks against significant prior swing levels.

    A "significant" level is the most recent confirmed swing high (for upward
    breaks) and the most recent confirmed swing low (for downward breaks) at
    the time of each bar.

    Parameters
    ----------
    data : pd.DataFrame
        OHLC data indexed by UTC timestamps.
    swings : List[Swing]
        Confirmed swings (from :func:`app.market_structure.swings.detect_swings`).
    confirm_bars : int
        Number of subsequent bars a close must be sustained beyond the level
        for a ``confirmed_break``. If ``0``, a close break with a move of at
        least ``min_move_pct`` percent counts as confirmed.
    min_move_pct : float
        Minimum percent move beyond the level for a confirmed break when
        ``confirm_bars == 0``.

    Returns
    -------
    List[BreakEvent]
        Break events ordered by timestamp.
    """
    for col in (high_col, low_col, close_col):
        if col not in data.columns:
            raise StructureError(f"Column '{col}' not found in data.")

    sorted_data = data.sort_index()
    highs = sorted_data[high_col].to_numpy(dtype=float)
    lows = sorted_data[low_col].to_numpy(dtype=float)
    closes = sorted_data[close_col].to_numpy(dtype=float)
    timestamps = sorted_data.index.to_numpy()
    n = len(highs)

    # Build a per-bar map of the most recent confirmed swing high/low level.
    # A swing is "active" from its confirmation bar onward.
    #
    # Optimised: instead of O(swings × bars) nested loops, collect
    # (confirmation_index, price) pairs, sort by index descending, then do a
    # single right-to-left scan.  This collapses the level-filling from
    # quadratic to O(swings·log(swings) + bars).
    ts_to_idx = {pd.Timestamp(ts): idx for idx, ts in enumerate(timestamps)}

    high_events: list[tuple[int, float]] = []
    low_events: list[tuple[int, float]] = []
    for swing in swings:
        conf_idx = ts_to_idx.get(pd.Timestamp(swing.confirmation_timestamp))
        if conf_idx is None:
            continue
        if swing.swing_type == SwingType.HIGH:
            high_events.append((conf_idx, swing.price))
        else:
            low_events.append((conf_idx, swing.price))

    # Sort ascending by confirmation index.  A swing confirmed at bar k
    # applies to all bars >= k (future), so a single left-to-right pass
    # accumulating the running max high / min low is equivalent to the
    # original O(swings × bars) nested fill.
    high_events.sort(key=lambda x: x[0])
    low_events.sort(key=lambda x: x[0])

    swing_high_levels: list[float | None] = [None] * n
    swing_low_levels: list[float | None] = [None] * n

    hi_ptr = 0
    lo_ptr = 0
    cur_high: float | None = None
    cur_low: float | None = None
    for i in range(n):
        # Activate any swing confirmed exactly at bar i.
        while hi_ptr < len(high_events) and high_events[hi_ptr][0] == i:
            val = high_events[hi_ptr][1]
            if cur_high is None or val > cur_high:
                cur_high = val
            hi_ptr += 1
        while lo_ptr < len(low_events) and low_events[lo_ptr][0] == i:
            val = low_events[lo_ptr][1]
            if cur_low is None or val < cur_low:
                cur_low = val
            lo_ptr += 1
        swing_high_levels[i] = cur_high
        swing_low_levels[i] = cur_low

    events: list[BreakEvent] = []

    for i in range(n):
        ts = _to_datetime(timestamps[i])

        # Upward breaks against the most recent significant swing high
        level = swing_high_levels[i]
        if level is not None:
            if highs[i] > level and closes[i] <= level:
                events.append(
                    BreakEvent(
                        symbol=symbol,
                        timeframe=timeframe,
                        timestamp=ts,
                        break_type=BreakType.WICK_BREACH,
                        level=level,
                        direction="up",
                        confirmation_timestamp=ts,
                        available_from=ts,
                    )
                )
            elif closes[i] > level:
                btype, conf_ts = _classify_close_break(
                    closes, i, level, confirm_bars, min_move_pct, timestamps, direction="up"
                )
                events.append(
                    BreakEvent(
                        symbol=symbol,
                        timeframe=timeframe,
                        timestamp=ts,
                        break_type=btype,
                        level=level,
                        direction="up",
                        confirmation_timestamp=conf_ts,
                        available_from=conf_ts or ts,
                    )
                )

        # Downward breaks against the most recent significant swing low
        level = swing_low_levels[i]
        if level is not None:
            if lows[i] < level and closes[i] >= level:
                events.append(
                    BreakEvent(
                        symbol=symbol,
                        timeframe=timeframe,
                        timestamp=ts,
                        break_type=BreakType.WICK_BREACH,
                        level=level,
                        direction="down",
                        confirmation_timestamp=ts,
                        available_from=ts,
                    )
                )
            elif closes[i] < level:
                btype, conf_ts = _classify_close_break(
                    closes, i, level, confirm_bars, min_move_pct, timestamps, direction="down"
                )
                events.append(
                    BreakEvent(
                        symbol=symbol,
                        timeframe=timeframe,
                        timestamp=ts,
                        break_type=btype,
                        level=level,
                        direction="down",
                        confirmation_timestamp=conf_ts,
                        available_from=conf_ts or ts,
                    )
                )

    return events


def _classify_close_break(
    closes,
    i: int,
    level: float,
    confirm_bars: int,
    min_move_pct: float,
    timestamps,
    direction: str,
):
    """Classify a close beyond a level as close_break or confirmed_break."""
    n = len(closes)
    if confirm_bars > 0:
        # Need confirm_bars subsequent closes to remain beyond the level.
        sustained = True
        for k in range(1, confirm_bars + 1):
            if i + k >= n:
                sustained = False
                break
            if direction == "up" and closes[i + k] <= level:
                sustained = False
                break
            if direction == "down" and closes[i + k] >= level:
                sustained = False
                break
        if sustained and i + confirm_bars < n:
            conf_ts = _to_datetime(timestamps[i + confirm_bars])
            return BreakType.CONFIRMED_BREAK, conf_ts
        return BreakType.CLOSE_BREAK, None

    # confirm_bars == 0: require a minimum move beyond the level.
    if direction == "up":
        move_pct = (closes[i] - level) / level * 100.0
    else:
        move_pct = (level - closes[i]) / level * 100.0
    if move_pct >= min_move_pct:
        return BreakType.CONFIRMED_BREAK, _to_datetime(timestamps[i])
    return BreakType.CLOSE_BREAK, None


def _index_of(timestamps, target: datetime) -> int | None:
    """Return the integer index of ``target`` in ``timestamps``, or None."""
    for idx, ts in enumerate(timestamps):
        if pd.Timestamp(ts) == pd.Timestamp(target):
            return idx
    return None


def _to_datetime(value) -> datetime:
    if isinstance(value, datetime):
        return value
    return pd.Timestamp(value).to_pydatetime()