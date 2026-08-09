"""Look-ahead regression tests for the regime engine.

The invariant: modifying future candles must never change a regime observation
whose ``available_from`` precedes the first modified bar.
"""

import numpy as np
import pandas as pd

from app.market_structure.engine import MarketStructureEngine
from app.regime import RegimeEngine


def _make_ohlc(n=220):
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    close = 100.0 + np.cumsum(np.random.default_rng(3).normal(0, 0.2, n))
    return pd.DataFrame(
        {"open": close, "high": close + 0.5, "low": close - 0.5, "close": close},
        index=idx,
    )


def _regime_key(r) -> tuple:
    return (
        r.timestamp,
        r.trend_state.value,
        r.volatility_state.value,
        r.market_state.value,
        r.news_risk.value,
        round(r.strength, 6),
        r.available_from,
    )


class TestRegimeLookAhead:
    def test_early_regimes_unchanged_when_future_altered(self) -> None:
        data = _make_ohlc()
        engine = RegimeEngine()
        structure = MarketStructureEngine().analyze(data, "EURUSD", "1h")
        full = engine.analyze(data, "EURUSD", "1h", market_structure=structure)

        # Drastically alter the last 40 bars.
        modified = data.copy()
        modified.iloc[-40:] *= 2.0
        mod_structure = MarketStructureEngine().analyze(modified, "EURUSD", "1h")
        recomputed = engine.analyze(modified, "EURUSD", "1h", market_structure=mod_structure)

        cutoff = data.index[-40]
        full_early = {_regime_key(r) for r in full if r.available_from < cutoff}
        mod_early = {_regime_key(r) for r in recomputed if r.available_from < cutoff}

        # Regimes before the cutoff must be identical.
        assert mod_early == full_early
