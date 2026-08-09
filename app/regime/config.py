"""Configuration for the market-regime detection engine.

All values are DOCUMENTED DEVELOPMENT DEFAULTS only. They are not claimed to be
optimal for any instrument or strategy.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RegimeConfig:
    """Causal, configurable parameters for regime classification."""

    # ── Trend ────────────────────────────────────────────────────────────────
    ema_fast: int = 20
    ema_slow: int = 50
    # Minimum price % separation between fast and slow EMA to call a "decisive"
    # bullish/bearish signal (instead of neutral).
    ma_margin_pct: float = 0.10
    slope_periods: int = 5
    # Which MA the price-distance signal is measured against.
    distance_ma: str = "ema"
    distance_ma_period: int = 20
    recent_return_bars: int = 5
    # Earliest bar with enough MA/slope data for a trend snapshot.
    min_trend_bars: int = 55

    # ── Volatility ───────────────────────────────────────────────────────────
    atr_window: int = 14
    # Trailing causal window for the percentile rank of ATR/price.
    percentile_window: int = 100
    vol_low_pct: float = 25.0
    vol_high_pct: float = 75.0
    vol_extreme_pct: float = 95.0

    # ── Structure ────────────────────────────────────────────────────────────
    structure_lookback: int = 12
    min_structure_points: int = 3

    # ── Range ────────────────────────────────────────────────────────────────
    # Minimum range length (bars) for an active range to count as \"ranging\".
    range_min_bars: int = 10

    # ── Transition ───────────────────────────────────────────────────────────
    # ATR / SMA(ATR, range_window) above this ratio = volatility expansion.
    transition_vol_ratio: float = 1.6
    range_window: int = 30
    # Minimum number of conflicting trend signals for TRANSITION by conflict.
    transition_conflict_min: int = 2
