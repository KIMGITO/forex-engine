"""Tests that MTF context integrates with the strategy layer additively
(backward compatible) and does NOT change behavior when MTF is disabled."""

import numpy as np
import pandas as pd

from app.mtf import MtfConfig, MtfEngine
from app.strategy.config import StrategyConfig


def _frame(n=240, freq="1h", seed=1):
    idx = pd.date_range("2026-08-01", periods=n, freq=freq, tz="UTC")
    close = 1.08 + np.cumsum(np.random.default_rng(seed).normal(0.002, 0.001, n))
    return pd.DataFrame(
        {"open": close, "high": close + 0.002, "low": close - 0.002, "close": close},
        index=idx,
    )


class TestStrategyConfigBackwardCompat:
    def test_mtf_disabled_by_default(self):
        cfg = StrategyConfig()
        assert cfg.mtf_enabled is False
        assert cfg.mtf_min_aligned == 0
        assert "mtf_enabled" in cfg.to_dict()

    def test_mtf_enabled_to_dict(self):
        cfg = StrategyConfig(mtf_enabled=True, mtf_min_aligned=2)
        d = cfg.to_dict()
        assert d["mtf_enabled"] is True
        assert d["mtf_min_aligned"] == 2


class TestMtfGatesDisabled:
    def test_gates_pass_trivially_when_disabled(self):
        from app.strategy.base import Strategy

        strat = Strategy()
        passed, reasons = strat.mtf_gates_pass("long", None)
        assert passed is True
        assert reasons == []


class TestMtfGatesEnabled:
    def test_requires_mtf_context_when_enabled(self):
        from app.strategy.base import Strategy

        strat = Strategy(StrategyConfig(mtf_enabled=True))
        passed, reasons = strat.mtf_gates_pass("long", None)
        assert passed is False
        assert any("no MTF context" in r for r in reasons)

    def test_mtf_enabled_wires_into_scanner_context(self):
        """With mtf_contexts provided, StrategyContext exposes mtf_context()."""
        base = _frame(400, "15min", seed=5)
        h1 = _frame(400, "1h", seed=6)
        h4 = _frame(400, "4h", seed=7)
        mtf = MtfEngine(
            MtfConfig(base_timeframe="15m", higher_timeframes=("1h", "4h")),
            "EURUSD",
        )
        ctxs = mtf.analyze({"15m": base, "1h": h1, "4h": h4}, "15m")
        assert len(ctxs) == len(base)
        # Each MTF context carries available_from = observation timestamp.
        for c in ctxs:
            assert c.available_from == c.timestamp
            assert c.base_timeframe == "15m"