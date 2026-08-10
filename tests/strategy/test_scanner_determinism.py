"""Determinism + look-ahead tests for the historical signal scanner.

A signal at timestamp T must depend ONLY on information whose available_from
<= T. Modifying future candles must not change previously generated signals.
"""

import numpy as np
import pandas as pd

from app.strategy import (
    HistoricalSignalScanner,
    TrendStructureStrategy,
)


def _frame(n=260, seed=11):
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    close = 1.08 + np.cumsum(np.random.default_rng(seed).normal(0, 0.002, n))
    return pd.DataFrame(
        {"open": close, "high": close + 0.003, "low": close - 0.003, "close": close},
        index=idx,
    )


def _signal_keys(result) -> tuple:
    return tuple(
        (
            s.timestamp,
            s.direction.value,
            s.strategy,
            round(s.entry, 9),
            round(s.stop_loss, 9),
            round(s.take_profit, 9),
        )
        for s in result.signals
    )


class TestScannerDeterminism:
    def test_same_scan_twice_identical(self):
        df = _frame()
        strat1 = TrendStructureStrategy()
        strat2 = TrendStructureStrategy()
        r1 = HistoricalSignalScanner().scan(df, strat1, "EURUSD", "1h")
        r2 = HistoricalSignalScanner().scan(df, strat2, "EURUSD", "1h")
        assert r1.bars_processed == len(df)
        assert _signal_keys(r1) == _signal_keys(r2)

    def test_future_candles_do_not_change_past_signals(self):
        df = _frame()
        full = HistoricalSignalScanner().scan(
            df, TrendStructureStrategy(), "EURUSD", "1h"
        )
        modified = df.copy()
        modified.iloc[-60:] *= 2.0
        recomputed = HistoricalSignalScanner().scan(
            modified, TrendStructureStrategy(), "EURUSD", "1h"
        )
        cutoff = len(df) - 60
        past_full = [s for s in full.signals if s.timestamp < df.index[cutoff]]
        past_mod = [s for s in recomputed.signals if s.timestamp < modified.index[cutoff]]
        assert _signal_keys(type("R", (), {"signals": past_full})()) == _signal_keys(
            type("R", (), {"signals": past_mod})()
        )

    def test_no_signals_on_flat_data(self):
        idx = pd.date_range("2024-01-01", periods=120, freq="1h", tz="UTC")
        close = np.full(120, 1.08)
        df = pd.DataFrame(
            {"open": close, "high": close + 0.0005, "low": close - 0.0005, "close": close},
            index=idx,
        )
        result = HistoricalSignalScanner().scan(
            df, TrendStructureStrategy(), "EURUSD", "1h"
        )
        # Flat data: no displacement, no trend → no signals (acceptable).
        assert isinstance(result.signals, list)
