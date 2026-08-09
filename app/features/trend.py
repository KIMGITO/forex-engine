"""Trend-based features: SMA, EMA, distance from moving average, and slope.

All windows are trailing and right-aligned (causal). Values before the first
full window are ``NaN``; no silent filling is performed.
"""

import pandas as pd

from app.features.errors import InsufficientDataError

__all__ = ["distance_from_ma", "ema", "ma_slope", "sma"]


def sma(
    data: pd.DataFrame,
    period: int = 20,
    price_col: str = "close",
    min_periods: int | None = None,
) -> pd.Series:
    """Simple moving average of ``price_col`` over a trailing window."""
    if price_col not in data.columns:
        raise ValueError(f"Price column '{price_col}' not found in data.")
    series = data[price_col].sort_index()
    if len(series) < period:
        raise InsufficientDataError(
            f"Need at least {period} observations for SMA; got {len(series)}."
        )
    return series.rolling(window=period, min_periods=min_periods or period).mean()


def ema(
    data: pd.DataFrame,
    period: int = 20,
    price_col: str = "close",
    min_periods: int | None = None,
) -> pd.Series:
    """Exponential moving average of ``price_col`` over a trailing window.

    Uses ``adjust=False`` (exponential-weighted) for consistency with the
    standard definition used in financial analysis.
    """
    if price_col not in data.columns:
        raise ValueError(f"Price column '{price_col}' not found in data.")
    series = data[price_col].sort_index()
    if len(series) < period:
        raise InsufficientDataError(
            f"Need at least {period} observations for EMA; got {len(series)}."
        )
    return series.ewm(span=period, adjust=False, min_periods=min_periods or period).mean()


def distance_from_ma(
    data: pd.DataFrame,
    period: int = 20,
    price_col: str = "close",
    ma_type: str = "sma",
) -> pd.Series:
    """Percentage distance of price from its moving average: ``(p / MA - 1) * 100``.

    ``ma_type`` can be ``"sma"`` (default) or ``"ema"``.
    """
    ma_func = ema if ma_type.lower() == "ema" else sma
    ma = ma_func(data, period=period, price_col=price_col)
    series = data[price_col].sort_index()
    return (series / ma - 1.0) * 100.0


def ma_slope(
    data: pd.DataFrame,
    period: int = 20,
    price_col: str = "close",
    ma_type: str = "sma",
    slope_periods: int = 5,
) -> pd.Series:
    """Slope of the moving average over ``slope_periods`` trailing bars.

    Computed as the rate of change of the MA itself: ``(MA_t / MA_{t-slope_periods} - 1) * 100``.
    ``ma_type`` can be ``"sma"`` (default) or ``"ema"``.
    """
    ma_func = ema if ma_type.lower() == "ema" else sma
    ma = ma_func(data, period=period, price_col=price_col)
    return ma.pct_change(periods=slope_periods, fill_method=None) * 100.0