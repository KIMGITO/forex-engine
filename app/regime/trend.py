"""Trend-state classification from multiple measurable signals.

Signals (each causal):
1. fast EMA vs slow EMA (with a minimum margin %)
2. fast-EMA slope sign (rate of change over ``slope_periods``)
3. price distance from a moving average (sign)
4. recent simple-return sign over ``recent_return_bars``

Final state = majority vote with explicit tie-handling:
- > 50% agreement on up → BULLISH, on down → BEARISH
- exact split or 2v2 mixed → NEUTRAL
- insufficient signals → UNKNOWN (never forced)
"""


import numpy as np
import pandas as pd

from app.features.returns import simple_returns
from app.features.trend import ema
from app.regime.config import RegimeConfig
from app.regime.models import TrendState

__all__ = ["classify_trend_series", "trend_signals"]


def trend_signals(data: pd.DataFrame, config: RegimeConfig) -> pd.DataFrame:
    """Return a DataFrame of per-bar trend signals (1=up, -1=down, 0=neutral, NaN=unavailable)."""
    sorted_data = data.sort_index()
    close = sorted_data["close"]

    fast = ema(sorted_data, period=config.ema_fast, price_col="close")
    slow = ema(sorted_data, period=config.ema_slow, price_col="close")

    # Signal 1: fast vs slow EMA with margin.
    spread_pct = (fast - slow) / slow * 100.0
    s1 = pd.Series(np.nan, index=sorted_data.index)
    s1[spread_pct > config.ma_margin_pct] = 1.0
    s1[spread_pct < -config.ma_margin_pct] = -1.0
    s1[(spread_pct.abs() <= config.ma_margin_pct) & spread_pct.notna()] = 0.0

    # Signal 2: fast-EMA slope.
    slope = fast.pct_change(periods=config.slope_periods, fill_method=None)
    s2 = pd.Series(np.nan, index=sorted_data.index)
    s2[slope > 0] = 1.0
    s2[slope < 0] = -1.0
    s2[slope == 0] = 0.0

    # Signal 3: price distance from distance-MA.
    dist_period = config.distance_ma_period
    if config.distance_ma.lower() == "ema":
        ma = ema(sorted_data, period=dist_period, price_col="close")
    else:
        from app.features.trend import sma

        ma = sma(sorted_data, period=dist_period, price_col="close")
    dist = (close - ma) / ma * 100.0
    s3 = pd.Series(np.nan, index=sorted_data.index)
    s3[dist > 0] = 1.0
    s3[dist < 0] = -1.0
    s3[dist == 0] = 0.0

    # Signal 4: recent return sign.
    ret = simple_returns(sorted_data, price_col="close")
    recent = ret.rolling(window=config.recent_return_bars, min_periods=1).sum()
    s4 = pd.Series(np.nan, index=sorted_data.index)
    s4[recent > 0] = 1.0
    s4[recent < 0] = -1.0
    s4[recent == 0] = 0.0

    return pd.DataFrame({"s1": s1, "s2": s2, "s3": s3, "s4": s4}, index=sorted_data.index)


def classify_trend_series(
    data: pd.DataFrame,
    config: RegimeConfig,
) -> pd.Series:
    """Classify each bar's trend state by majority vote (causal)."""
    signals = trend_signals(data, config)
    states: list = []

    for _, row in signals.iterrows():
        valid = [v for v in row.tolist() if not np.isnan(v)]
        if len(valid) == 0:
            states.append(TrendState.UNKNOWN)
            continue
        ups = sum(1 for v in valid if v > 0)
        downs = sum(1 for v in valid if v < 0)
        if ups > downs:
            states.append(TrendState.BULLISH)
        elif downs > ups:
            states.append(TrendState.BEARISH)
        else:
            states.append(TrendState.NEUTRAL)

    return pd.Series(states, index=signals.index)
