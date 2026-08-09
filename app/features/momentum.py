"""Momentum-based features: rate of change, price momentum, and RSI.

These are pure features — they are *not* trading signals. For example, an RSI
value of 72 is simply a feature; deciding whether it implies a SELL belongs to
a future strategy layer.
"""

import numpy as np
import pandas as pd

from app.features.errors import InsufficientDataError

__all__ = ["price_momentum", "rate_of_change", "rsi"]


def rate_of_change(
    data: pd.DataFrame,
    period: int = 10,
    price_col: str = "close",
) -> pd.Series:
    """Rate of change: ``(p_t / p_{t-period} - 1) * 100``, lagged by ``period``.

    Values before the first valid ``period``-back observation are ``NaN``.
    """
    if price_col not in data.columns:
        raise ValueError(f"Price column '{price_col}' not found in data.")
    series = data[price_col].sort_index()
    if len(series) < period + 1:
        raise InsufficientDataError(
            f"Need at least {period + 1} observations for ROC; got {len(series)}."
        )
    return series.pct_change(periods=period, fill_method=None) * 100.0


def price_momentum(
    data: pd.DataFrame,
    period: int = 10,
    price_col: str = "close",
) -> pd.Series:
    """Price momentum: ``p_t - p_{t-period}``, lagged by ``period``."""
    if price_col not in data.columns:
        raise ValueError(f"Price column '{price_col}' not found in data.")
    series = data[price_col].sort_index()
    if len(series) < period + 1:
        raise InsufficientDataError(
            f"Need at least {period + 1} observations for price momentum; got {len(series)}."
        )
    return series.diff(periods=period)


def rsi(
    data: pd.DataFrame,
    period: int = 14,
    price_col: str = "close",
) -> pd.Series:
    """Relative Strength Index (Wilder), in [0, 100].

    Uses Wilder's smoothing: a seed average gain/loss over the first ``period``
    changes, then recursive ``ewm(alpha=1/period, adjust=False)`` smoothing.
    Values before the seed window are ``NaN``.
    """
    if price_col not in data.columns:
        raise ValueError(f"Price column '{price_col}' not found in data.")
    series = data[price_col].sort_index()
    if len(series) < period + 1:
        raise InsufficientDataError(
            f"Need at least {period + 1} observations for RSI; got {len(series)}."
        )

    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    # Wilder seed: simple mean over the first `period` changes.
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    # RSI is undefined when there is no loss (or no gain) over the window.
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_gain / avg_loss
        rsi_value = 100.0 - (100.0 / (1.0 + rs))

    # When avg_loss == 0 but avg_gain > 0, RSI is conventionally 100.
    rsi_value = rsi_value.where(~((avg_loss == 0) & (avg_gain > 0)), 100.0)
    # When both are zero, RSI is undefined -> NaN.
    rsi_value = rsi_value.where(~((avg_loss == 0) & (avg_gain == 0)))
    return rsi_value