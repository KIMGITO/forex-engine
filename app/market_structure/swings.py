"""Swing high/low detection.

A swing high at bar ``i`` is a bar whose ``high`` is strictly greater than the
``high`` of every bar in the neighborhood ``[i - left, i + right]`` (excluding
``i`` itself). A swing low is the mirror using ``low``.

**Look-ahead note:** Swing detection requires ``right`` future bars to confirm.
It is therefore **not causal in real-time** — it is intended for historical
analysis / replay only. Every returned :class:`Swing` carries an explicit
``confirmation_timestamp`` (the bar at which the swing becomes knowable) and an
``available_from`` field. A consumer must never act on a swing before
``available_from``.
"""

from datetime import datetime

import pandas as pd

from app.market_structure.errors import SwingDetectionError
from app.market_structure.models import Swing, SwingType

__all__ = ["detect_swings"]


def detect_swings(
    data: pd.DataFrame,
    symbol: str,
    timeframe: str,
    left: int = 3,
    right: int = 3,
    high_col: str = "high",
    low_col: str = "low",
) -> list[Swing]:
    """Detect confirmed swing highs and swing lows.

    Parameters
    ----------
    data : pd.DataFrame
        OHLC data indexed by UTC timestamps. Must contain ``high_col`` and
        ``low_col``.
    symbol, timeframe : str
        Metadata propagated to each :class:`Swing`.
    left, right : int
        Lookback and lookforward window sizes. A swing at bar ``i`` requires
        ``left`` bars before and ``right`` bars after it.
    high_col, low_col : str
        Column names for highs and lows.

    Returns
    -------
    List[Swing]
        Confirmed swings, ordered by timestamp. Each swing's
        ``confirmation_timestamp`` is the bar at ``i + right``.
    """
    if left < 1 or right < 1:
        raise SwingDetectionError("left and right must both be >= 1.")
    for col in (high_col, low_col):
        if col not in data.columns:
            raise SwingDetectionError(f"Column '{col}' not found in data.")

    sorted_data = data.sort_index()
    highs = sorted_data[high_col].to_numpy(dtype=float)
    lows = sorted_data[low_col].to_numpy(dtype=float)
    timestamps = sorted_data.index.to_numpy()
    n = len(highs)

    if n < left + right + 1:
        raise SwingDetectionError(
            f"Need at least {left + right + 1} bars for swing detection; got {n}."
        )

    swings: list[Swing] = []

    for i in range(left, n - right):
        lo = i - left
        hi = i + right

        # Swing high: high[i] > all highs in neighborhood (excluding i)
        if highs[i] > highs[lo:i].max() and highs[i] > highs[i + 1 : hi + 1].max():
            ts = _to_datetime(timestamps[i])
            conf_ts = _to_datetime(timestamps[i + right])
            swings.append(
                Swing(
                    symbol=symbol,
                    timeframe=timeframe,
                    timestamp=ts,
                    swing_type=SwingType.HIGH,
                    price=float(highs[i]),
                    confirmation_timestamp=conf_ts,
                    available_from=conf_ts,
                    left=left,
                    right=right,
                )
            )

        # Swing low: low[i] < all lows in neighborhood (excluding i)
        if lows[i] < lows[lo:i].min() and lows[i] < lows[i + 1 : hi + 1].min():
            ts = _to_datetime(timestamps[i])
            conf_ts = _to_datetime(timestamps[i + right])
            swings.append(
                Swing(
                    symbol=symbol,
                    timeframe=timeframe,
                    timestamp=ts,
                    swing_type=SwingType.LOW,
                    price=float(lows[i]),
                    confirmation_timestamp=conf_ts,
                    available_from=conf_ts,
                    left=left,
                    right=right,
                )
            )

    return swings


def _to_datetime(value) -> datetime:
    """Convert a pandas/numpy timestamp to a timezone-aware datetime."""
    if isinstance(value, datetime):
        return value
    return pd.Timestamp(value).to_pydatetime()