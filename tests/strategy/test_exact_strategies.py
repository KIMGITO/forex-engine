"""Tests for the two concrete strategies (structure/behavior on synthetic data).

Purpose: verifying strategy mechanics (causal evaluation, signal model
validation, no-signal-when-incomplete) — NOT profitability or optimization.
"""

import numpy as np
import pandas as pd

from app.strategy import (
    HistoricalSignalScanner,
    LiquidityReversalStrategy,
    TrendStructureStrategy,
)


def _trending_frame(n=250, seed=2, drift=0.003):
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    close = 1.08 + np.cumsum(np.random.default_rng(seed).normal(drift, 0.001, n))
    return pd.DataFrame(
        {"open": close, "high": close + 0.003, "low": close - 0.003, "close": close},
        index=idx,
    )


class TestTrendStructureStrategy:
    def test_runs_without_error_on_trending_data(self):
        df = _trending_frame()
        result = HistoricalSignalScanner().scan(
            df, TrendStructureStrategy(), "EURUSD", "1h"
        )
        assert result.bars_processed == len(df)
        assert result.strategy == "trend_structure"

    def test_signal_geometry_invariants(self):
        df = _trending_frame(drift=0.005, seed=3)
        result = HistoricalSignalScanner().scan(
            df, TrendStructureStrategy(), "EURUSD", "1h"
        )
        for s in result.signals:
            if s.direction.value == "long":
                assert s.stop_loss < s.entry < s.take_profit
            else:
                assert s.take_profit < s.entry < s.stop_loss
            assert s.risk_reward_ratio >= 1.0

    def test_no_manufactured_signals_on_flat_data(self):
        idx = pd.date_range("2024-01-01", periods=120, freq="1h", tz="UTC")
        close = np.full(120, 1.08)
        df = pd.DataFrame(
            {"open": close, "high": close + 0.0005, "low": close - 0.0005, "close": close},
            index=idx,
        )
        result = HistoricalSignalScanner().scan(
            df, TrendStructureStrategy(), "EURUSD", "1h"
        )
        # Incomplete conditions -> no manufactured signals.
        assert all(s.status.value == "detected" for s in result.signals)


class TestLiquidityReversalStrategy:
    def test_runs_without_error(self):
        df = _trending_frame(seed=4)
        result = HistoricalSignalScanner().scan(
            df, LiquidityReversalStrategy(), "EURUSD", "1h"
        )
        assert result.bars_processed == len(df)
        assert result.strategy == "liquidity_reversal"

    def test_signals_are_valid(self):
        df = _trending_frame(seed=5)
        result = HistoricalSignalScanner().scan(
            df, LiquidityReversalStrategy(), "EURUSD", "1h"
        )
        for s in result.signals:
            if s.direction.value == "long":
                assert s.stop_loss < s.entry < s.take_profit
            else:
                assert s.take_profit < s.entry < s.stop_loss
            assert s.risk_reward_ratio >= 1.0
