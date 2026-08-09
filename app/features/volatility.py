"""Volatility-based features: rolling std, annualized volatility, ATR, and rank.

All windows are trailing and right-aligned (causal). Values before the first
full window are ``NaN``; no silent filling is performed.
"""


import numpy as np
import pandas as pd

from app.features.errors import InsufficientDataError

__all__ = [
    "rolling_std_init",
]

PERIODS_PER_YEAR: dict[str, float] = {
    "1m": 525_600,
    "5m": 105_120,
    "15m": 35_040,
    "30m": 17_520,
    "1h": 8_760,
    "4h": 2_190,
    "1d": 365,
    "1w": 52,
}


def _periods_per_year(timeframe: str) -> float:
    """Return periods per year for the given timeframe, defaulting to 1h."""
    return PERIODS_PER_YEAR.get(timeframe.lower(), 8_760.0)


def rolling_std_init(
    data: pd.DataFrame,
    window: int = 20,
    price_col: str = "close",
    ddof: int = 1,
) -> pd.Series:
    """Rolling standard deviation of ``price_col`` over a trailing window."""
    if price_col not in data.columns:
        raise ValueError(f"Price column '{price_col}' not found in data.")
    series = data[price_col].sort_index()
    if len(series) < window:
        raise InsufficientDataError(
            f"Need at least {window} observations for rolling std; got {len(series)}."
        )
    return series.rolling(window=window, min_periods=window).std(ddof=ddof)


def annualized_volatility(
    data: pd.DataFrame,
    window: int = 20,
    price_col: str = "close",
    timeframe: str = "1h",
    ddof: int = 1,
) -> pd.Series:
    """Annualized rolling volatility of returns.

    Computed as the rolling standard deviation of simple returns scaled by the
    square root of the number of periods per year for the given timeframe.
    """
    if price_col not in data.columns:
        raise ValueError(f"Price column '{price_col}' not found in data.")
    series = data[price_col].sort_index()
    returns = series.pct_change(fill_method=None)
    if len(returns) < window:
        raise InsufficientDataError(
            f"Need at least {window} observations for annualized volatility; got {len(returns)}."
        )
    rolling = returns.rolling(window=window, min_periods=window).std(ddof=ddof)
    return rolling * np.sqrt(_periods_per_year(timeframe))


def atr(
    data: pd.DataFrame,
    window: int = 14,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
) -> pd.Series:
    """Average True Range (Wilder smoothing).

    True Range = max(high - low, |high - prev_close|, |low - prev_close|).
    ATR uses Wilder's smoothing: ``ewm(alpha=1/window, adjust=False)``.
    """
    for col in (high_col, low_col, close_col):
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in data.")
    sorted_data = data.sort_index()
    high = sorted_data[high_col]
    low = sorted_data[low_col]
    close = sorted_data[close_col]

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    if len(tr) < window:
        raise InsufficientDataError(
            f"Need at least {window} observations for ATR; got {len(tr)}."
        )
    return tr.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()


def rolling_atr(
    data: pd.DataFrame,
    window: int = 14,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
) -> pd.Series:
    """Rolling (simple mean) ATR over a trailing window."""
    tr = _true_range(data, high_col, low_col, close_col)
    if len(tr) < window:
        raise InsufficientDataError(
            f"Need at least {window} observations for rolling ATR; got {len(tr)}."
        )
    return tr.rolling(window=window, min_periods=window).mean()


def _true_range(
    data: pd.DataFrame,
    high_col: str,
    low_col: str,
    close_col: str,
) -> pd.Series:
    for col in (high_col, low_col, close_col):
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in data.")
    sorted_data = data.sort_index()
    prev_close = sorted_data[close_col].shift(1)
    return pd.concat(
        [
            sorted_data[high_col] - sorted_data[low_col],
            (sorted_data[high_col] - prev_close).abs(),
            (sorted_data[low_col] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def volatility_percentile(
    data: pd.DataFrame,
    window: int = 20,
    rank_window: int = 100,
    price_col: str = "close",
    ddof: int = 1,
) -> pd.Series:
    """Rolling percentile rank of rolling volatility within a trailing window.

    The percentile rank at time ``T`` considers only volatility values up to and
    including ``T`` (causal), expressed in [0, 100].
    """
    rolling = rolling_std_init(data, window=window, price_col=price_col, ddof=ddof)
    if len(rolling) < rank_window:
        raise InsufficientDataError(
            f"Need at least {rank_window} observations for volatility percentile; "
            f"got {len(rolling)}."
        )
    return rolling.rolling(window=rank_window, min_periods=rank_window).apply(
        lambda x: (x <= x[-1]).mean() * 100.0, raw=True
    )