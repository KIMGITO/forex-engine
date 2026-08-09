"""Return-based features: simple, percentage, and logarithmic returns.

Returns are computed with a one-period lag (``shift(1)``) so that the return
classified at timestamp ``T`` reflects only prior price information. Values
before the first valid observation are ``NaN``; no silent filling is performed.
"""

import numpy as np
import pandas as pd

from app.features.errors import InsufficientDataError

__all__ = ["log_returns", "pct_returns", "simple_returns"]


def _prepare(data: pd.DataFrame, price_col: str, min_obs: int) -> pd.Series:
    """Validate and sort the input, returning the sorted price series."""
    if price_col not in data.columns:
        raise ValueError(f"Price column '{price_col}' not found in data.")
    series = data[price_col]
    if series.isna().any():
        raise ValueError("Price series contains NaN values; cannot compute returns.")
    if len(series) < min_obs:
        raise InsufficientDataError(
            f"Need at least {min_obs} observations to compute returns; got {len(series)}."
        )
    return series.sort_index()


def simple_returns(data: pd.DataFrame, price_col: str = "close", min_obs: int = 2) -> pd.Series:
    """Simple returns: ``(p_t - p_{t-1}) / p_{t-1}``, lagged by one period."""
    series = _prepare(data, price_col, min_obs)
    return series.pct_change(fill_method=None)


def pct_returns(data: pd.DataFrame, price_col: str = "close", min_obs: int = 2) -> pd.Series:
    """Percentage returns: ``(p_t / p_{t-1} - 1) * 100``, lagged by one period."""
    series = _prepare(data, price_col, min_obs)
    return series.pct_change(fill_method=None) * 100.0


def log_returns(data: pd.DataFrame, price_col: str = "close", min_obs: int = 2) -> pd.Series:
    """Logarithmic returns: ``log(p_t / p_{t-1})``, lagged by one period."""
    series = _prepare(data, price_col, min_obs)
    return (series / series.shift(1)).apply(np.log)
