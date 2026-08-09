"""Volatility-state classification.

Reuses :mod:`app.features.volatility` (ATR) rather than reimplementing it.
Classification uses a trailing, causal percentile rank of ``ATR / price``.
"""


import numpy as np
import pandas as pd

from app.features.volatility import atr
from app.regime.config import RegimeConfig
from app.regime.models import VolatilityState

__all__ = ["classify_volatility_series"]


def atr_ratio_series(data: pd.DataFrame, config: RegimeConfig) -> pd.Series:
    """Return ATR / close as a causal series (NaN where ATR is undefined)."""
    a = atr(data, window=config.atr_window).sort_index()
    close = data["close"].sort_index()
    return a / close


def classify_volatility_series(
    data: pd.DataFrame,
    config: RegimeConfig,
) -> tuple[pd.Series, pd.Series]:
    """Classify each bar's volatility state causally.

    Returns ``(states, ratios)`` where ``states`` is a Series of
    :class:`VolatilityState` (UNKNOWN where the percentile window is
    insufficient) and ``ratios`` is the underlying ATR/price ratio series.
    """
    ratios = atr_ratio_series(data, config)
    states: list = []

    rolling: list[float] = []
    for value in ratios.tolist():
        if np.isnan(value):
            rolling.append(float("nan"))
        else:
            rolling.append(value)

    for i, value in enumerate(rolling):
        if np.isnan(value):
            states.append(VolatilityState.UNKNOWN)
            continue
        window = rolling[max(0, i - config.percentile_window + 1) : i + 1]
        window = [v for v in window if not np.isnan(v)]
        if len(window) < 2:
            states.append(VolatilityState.UNKNOWN)
            continue
        pct_rank = sum(1 for v in window if v <= value) / len(window) * 100.0
        if pct_rank >= config.vol_extreme_pct:
            states.append(VolatilityState.EXTREME)
        elif pct_rank >= config.vol_high_pct:
            states.append(VolatilityState.HIGH)
        elif pct_rank <= config.vol_low_pct:
            states.append(VolatilityState.LOW)
        else:
            states.append(VolatilityState.NORMAL)

    idx = ratios.index
    return pd.Series(states, index=idx), ratios
