"""Displacement: a per-bar metric of price movement relative to recent ATR.

Displacement measures *price behavior* only. It intentionally does **not**
attribute meaning to market participants (e.g., "institutional buying"). We can
measure how unusually large a bar is relative to recent volatility, but we
cannot infer who traded or why from OHLC data alone.
"""

from datetime import datetime

import numpy as np
import pandas as pd

from app.features.volatility import atr
from app.market_structure.errors import DisplacementError
from app.market_structure.models import DisplacementClass, DisplacementEvent

__all__ = ["compute_displacement"]


def compute_displacement(
    data: pd.DataFrame,
    symbol: str,
    timeframe: str,
    atr_window: int = 14,
    p_extreme: float = 95.0,
    p_large: float = 80.0,
    p_small: float = 20.0,
    high_col: str = "high",
    low_col: str = "low",
    open_col: str = "open",
    close_col: str = "close",
) -> list[DisplacementEvent]:
    """Compute per-bar displacement relative to recent ATR.

    For each bar ``i``:
    * ``range_i = high - low``
    * ``range_ratio = range_i / ATR[i]``  (NaN where ATR is NaN)
    * ``body_ratio = |close - open| / range_i``  (0 when range is 0)
    * ``direction`` = ``up`` | ``down`` | ``flat`` based on close vs open.

    Each bar is classified relative to the *trailing* distribution of
    ``range_ratio`` using the configurable percentiles ``p_extreme``,
    ``p_large``, and ``p_small``:

    * ``extreme`` — ratio >= ``p_extreme`` percentile of prior ratios
    * ``large``   — ratio in [``p_large``, ``p_extreme``) percentile
    * ``small``   — ratio <= ``p_small`` percentile
    * ``normal``  — otherwise

    Classification is **causal**: it uses only ratios up to and including bar
    ``i`` (trailing window). ``available_from`` equals the bar timestamp.

    Parameters
    ----------
    data : pd.DataFrame
        OHLC data indexed by UTC timestamps.
    atr_window : int
        Rolling window used for the ATR denominator.
    p_extreme, p_large, p_small : float
        Percentile thresholds (0-100) for classification.

    Returns
    -------
    List[DisplacementEvent]
        One event per bar, ordered chronologically.
    """
    for col in (high_col, low_col, open_col, close_col):
        if col not in data.columns:
            raise DisplacementError(f"Column '{col}' not found in data.")

    sorted_data = data.sort_index()
    high = sorted_data[high_col].to_numpy(dtype=float)
    low = sorted_data[low_col].to_numpy(dtype=float)
    open_p = sorted_data[open_col].to_numpy(dtype=float)
    close = sorted_data[close_col].to_numpy(dtype=float)
    timestamps = sorted_data.index.to_numpy()
    n = len(high)

    if n < atr_window + 1:
        raise DisplacementError(
            f"Need at least {atr_window + 1} bars for displacement; got {n}."
        )

    a = atr(data, window=atr_window)
    atr_vals = a.sort_index().to_numpy(dtype=float)

    events: list[DisplacementEvent] = []
    prev_ratios: list[float] = []

    # Order-statistic Fenwick tree so percentile classification is O(log V)
    # per bar instead of O(n) per bar (np.nanpercentile on the whole history).
    # This keeps exact causal semantics (percentiles over all trailing ratios
    # up to and including the current bar) while making compute_displacement
    # O(n log V) instead of O(n²).
    fenwick = _FenwickPercentiles(_RATIO_LO, _RATIO_HI, _RATIO_BINS)

    for i in range(n):
        rng = high[i] - low[i]
        if rng < 0:
            raise DisplacementError(f"Negative range at bar {i}; invalid OHLC data.")
        if rng == 0:
            body_ratio = 0.0
            direction = "flat"
        else:
            body_ratio = abs(close[i] - open_p[i]) / rng
            if close[i] > open_p[i]:
                direction = "up"
            elif close[i] < open_p[i]:
                direction = "down"
            else:
                direction = "flat"

        # range_ratio is NaN where ATR is NaN (insufficient lookback).
        if np.isnan(atr_vals[i]) or atr_vals[i] == 0:
            range_ratio = float("nan")
            classification = DisplacementClass.NORMAL
        else:
            range_ratio = rng / atr_vals[i]

        # Causal classification using trailing ratios (including current bar).
        if not np.isnan(range_ratio):
            fenwick.add(range_ratio)
            classification = _classify_ordered(
                range_ratio, fenwick, p_extreme, p_large, p_small
            )
        else:
            classification = DisplacementClass.NORMAL

        ts = _to_datetime(timestamps[i])
        events.append(
            DisplacementEvent(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=ts,
                range_ratio=range_ratio,
                body_ratio=body_ratio,
                direction=direction,
                classification=classification,
                available_from=ts,
            )
        )

    return events


# Binning range for the order-statistic Fenwick tree. range_ratio = range/ATR
# is centred near 1.0; a generous upper bound with fine resolution keeps the
# percentile lookup accurate while bounding memory.
_RATIO_LO = 0.0
_RATIO_HI = 50.0
_RATIO_BINS = 200_000


class _FenwickPercentiles:
    """Fenwick tree over binned ratio values for order-statistic queries.

    Supports O(log V) insert and O(log V) percentile lookup. Values below
    ``lo`` clamp to the first bin; values above ``hi`` clamp to the last bin.
    """

    __slots__ = ("lo", "width", "n", "tree", "total")

    def __init__(self, lo: float, hi: float, bins: int) -> None:
        self.lo = lo
        self.width = (hi - lo) / bins
        self.n = bins
        self.tree = [0] * (bins + 1)
        self.total = 0

    def _bin_index(self, value: float) -> int:
        idx = int((value - self.lo) / self.width)
        if idx < 0:
            idx = 0
        elif idx >= self.n:
            idx = self.n - 1
        return idx + 1  # 1-based for Fenwick

    def add(self, value: float) -> None:
        i = self._bin_index(value)
        self.total += 1
        n = self.n
        tree = self.tree
        while i <= n:
            tree[i] += 1
            i += i & -i

    def _prefix_sum(self, i: int) -> int:
        s = 0
        tree = self.tree
        while i > 0:
            s += tree[i]
            i -= i & -i
        return s

    def percentile(self, p: float) -> float:
        """Return the value at percentile ``p`` (0-100) of inserted values."""
        if self.total == 0:
            return 0.0
        target = self.total * p / 100.0
        # Fenwick binary lifting: find smallest index with prefix_sum >= target.
        idx = 0
        bit = 1 << (self.n.bit_length() - 1)
        while bit:
            nxt = idx + bit
            if nxt <= self.n and self.tree[nxt] < target:
                idx = nxt
                target -= self.tree[nxt]
            bit >>= 1
        idx += 1
        if idx > self.n:
            idx = self.n
        return self.lo + (idx - 1) * self.width


def _classify_ordered(
    ratio: float,
    fenwick: _FenwickPercentiles,
    p_extreme: float,
    p_large: float,
    p_small: float,
) -> DisplacementClass:
    """Classify a ratio against the trailing distribution (causal)."""
    if fenwick.total < 2:
        return DisplacementClass.NORMAL

    p_lo = fenwick.percentile(p_small)
    p_hi_large = fenwick.percentile(p_large)
    p_hi_extreme = fenwick.percentile(p_extreme)

    if ratio >= p_hi_extreme:
        return DisplacementClass.EXTREME
    if ratio >= p_hi_large:
        return DisplacementClass.LARGE
    if ratio <= p_lo:
        return DisplacementClass.SMALL
    return DisplacementClass.NORMAL


def _to_datetime(value) -> datetime:
    if isinstance(value, datetime):
        return value
    return pd.Timestamp(value).to_pydatetime()