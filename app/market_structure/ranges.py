"""Range/consolidation detection.

A consolidation range is a period of compressed volatility: a trailing ATR
ratio below a threshold. This module measures price behavior only — it does
not generate trade signals.
"""

from datetime import datetime

import numpy as np
import pandas as pd

from app.features.trend import sma
from app.features.volatility import atr
from app.market_structure.errors import RangeError
from app.market_structure.models import RangeEvent

__all__ = ["detect_ranges"]


def detect_ranges(
    data: pd.DataFrame,
    symbol: str,
    timeframe: str,
    atr_window: int = 14,
    compression_threshold: float = 0.85,
    range_window: int = 30,
    min_range_bars: int = 10,
) -> list[RangeEvent]:
    """Detect consolidation/range periods from compressed volatility.

    A bar ``i`` counts as compressed when
    ``atr[i] / sma(atr, range_window)[i] <= compression_threshold``, i.e. the
    current ATR is compressed relative to its own recent average. Contiguous
    runs of at least ``min_range_bars`` compressed bars become
    :class:`RangeEvent` objects.

    Parameters
    ----------
    data : pd.DataFrame
        OHLC data indexed by UTC timestamps. Must contain ``high``, ``low``,
        ``close``.
    atr_window : int
        Window for the ATR used as the volatility denominator.
    compression_threshold : float
        Max ``atr / sma(atr)`` ratio for a bar to count as compressed.
    range_window : int
        Trailing window for the ATR average (compression reference).
    min_range_bars : int
        Minimum consecutive compressed bars required to emit a range event.

    Returns
    -------
    List[RangeEvent]
        Detected ranges, ordered chronologically.
    """
    for col in ("high", "low", "close"):
        if col not in data.columns:
            raise RangeError(f"Column '{col}' not found in data.")

    sorted_data = data.sort_index()
    n = len(sorted_data)
    if n < range_window + atr_window:
        raise RangeError(
            f"Need at least {range_window + atr_window} bars for range detection; got {n}."
        )

    a = atr(sorted_data, window=atr_window).sort_index()
    atr_sma = sma(
        sorted_data.assign(_atr=a),
        period=range_window,
        price_col="_atr",
    ).sort_index()

    compression = a / atr_sma

    timestamps = sorted_data.index.to_numpy()
    highs = sorted_data["high"].to_numpy(dtype=float)
    lows = sorted_data["low"].to_numpy(dtype=float)

    compressed = compression.notna() & (compression <= compression_threshold)
    compressed = compressed.fillna(False)

    ranges: list[RangeEvent] = []
    run_start: int | None = None

    for i in range(n):
        if bool(compressed.iloc[i]):
            if run_start is None:
                run_start = i
        else:
            if run_start is not None:
                if i - run_start >= min_range_bars:
                    ranges.append(
                        _make_range_event(
                            highs, lows, timestamps, compression,
                            run_start, i - 1, symbol, timeframe,
                        )
                    )
                run_start = None

    if run_start is not None and n - run_start >= min_range_bars:
        ranges.append(
            _make_range_event(
                highs, lows, timestamps, compression,
                run_start, n - 1, symbol, timeframe,
            )
        )

    return ranges


def _make_range_event(
    highs: np.ndarray,
    lows: np.ndarray,
    timestamps,
    compression: pd.Series,
    start: int,
    end: int,
    symbol: str,
    timeframe: str,
) -> RangeEvent:
    """Build a RangeEvent for a detected contiguous compressed run."""
    upper = float(np.nanmax(highs[start : end + 1]))
    lower = float(np.nanmin(lows[start : end + 1]))
    ratio = compression.iloc[end]
    return RangeEvent(
        symbol=symbol,
        timeframe=timeframe,
        start_timestamp=_to_datetime(timestamps[start]),
        end_timestamp=_to_datetime(timestamps[end]),
        upper=upper,
        lower=lower,
        compression_ratio=float(ratio) if not np.isnan(ratio) else float("nan"),
        available_from=_to_datetime(timestamps[end]),
    )


def _to_datetime(value) -> datetime:
    if isinstance(value, datetime):
        return value
    return pd.Timestamp(value).to_pydatetime()