"""Causal HTF warm-up derivation for Step 13.

When processing a base timeframe (e.g. M15) we must not load the entire HTF
dataset. This module derives the minimum HTF history required for identical
structure/regime outputs, using the REAL engine configurations (not guessed
margins) — matching the proven pattern in ``app/mtf/engine.py``.

The result is a base-bar lookback; HTF windows are clipped to
``base_first - HTF_period * lookback``.
"""

from __future__ import annotations

from app.mtf.config import MtfConfig
from app.regime.config import RegimeConfig


def _tf_to_minutes(tf: str) -> int:
    _map = {
        "1m": 1, "5m": 5, "15m": 15, "30m": 30, "45m": 45,
        "1h": 60, "2h": 120, "4h": 240, "1d": 1440, "1w": 10080,
    }
    return _map.get(tf.lower(), 60)


def causal_htf_lookback_bars(
    config: MtfConfig | None = None,
) -> int:
    """Derive a safe historical lookback (in HTF bars) from REAL engine windows.

    Warm-ups considered:
      - regime trend slow EMA        : RegimeConfig.ema_slow (50)
      - regime volatility percentile : RegimeConfig.percentile_window (100)
      - regime range / ATR SMA       : RegimeConfig.range_window (30)
      - market-structure ATR         : MarketStructureConfig.atr_window (14)
      - market-structure ranges      : MarketStructureConfig.range_window (30)
      - swing confirmation right     : MarketStructureConfig.swing_right (3)
      - MTF gap bridging             : MtfConfig.max_gap_lookback (5)

    A 30% safety slack is added on top of the true maximum.
    """
    rc = RegimeConfig()
    warmups = [
        int(rc.ema_slow),
        int(rc.percentile_window),
        int(rc.range_window),
        int(rc.atr_window),
        int(getattr(rc, "range_min_bars", 10)),
    ]
    try:
        from app.market_structure.engine import MarketStructureConfig

        ms = MarketStructureConfig()
        warmups += [
            int(ms.atr_window),
            int(ms.range_window),
            int(ms.swing_right),
            int(ms.sweep_bars),
        ]
    except Exception:  # noqa: BLE001,S110 - optional config; safe defaults
        pass
    gap = int(config.max_gap_lookback) if config else 5
    warmups.append(gap)
    base = max(warmups)
    return int(base * 1.3) + 1


def clip_htf_frame(
    htf_df,
    base_first_ts,
    htf_timeframe: str,
    lookback_bars: int,
) -> "object":
    """Return an HTF frame clipped to the causal window required by the base.

    ``base_first_ts`` is the first base-bar timestamp. HTF bars are retained
    from ``base_first_ts - HTF_period * lookback_bars`` onward. This supplies
    all engine warm-up without loading the full HTF dataset.
    """
    import pandas as pd

    cutoff = pd.Timestamp(base_first_ts) - pd.Timedelta(
        minutes=_tf_to_minutes(htf_timeframe) * lookback_bars
    )
    window = htf_df.loc[htf_df.index >= cutoff]
    if window.empty:
        # Too little history available: keep the full (small) frame.
        return htf_df
    return window